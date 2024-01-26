# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import re
from typing import List, Optional

import torch
from torch.nn import Module
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import sparseml.core.session as session_manager
from sparseml.core.recipe import Recipe, StageRunType
from sparseml.modifiers.utils.pytorch_helpers import PADDING_MASK_COLUMN_NAME
from sparseml.pytorch.model_load.helpers import (
    get_completed_stages,
    get_session_model,
    save_completed_stages,
    save_model_and_recipe,
)
from sparseml.transformers.finetune import TrainingArguments
from sparseml.transformers.finetune.data import TextGenerationDataset
from sparseml.transformers.finetune.data.data_args import DataTrainingArguments
from sparseml.transformers.finetune.data.data_helpers import (
    format_calibration_data,
    make_dataset_splits,
)
from sparseml.transformers.finetune.model_args import ModelArguments
from sparseml.utils.fsdp.context import summon_full_params_context
from sparseml.utils.fsdp.helpers import is_fsdp_model, unwrap_and_export_model
from sparseml.utils.pytorch import qat_active


_LOGGER: logging.Logger = logging.getLogger(__name__)


class StageRunner:
    """
    Launcher class for train, eval and one_shot flows. Manages data splits for each
    flow and configurations. In the future this class will also handle alternating
    between the different flows

    LifeCycle
        - populate_datasets()
        - set_trainer()
        - train() / evaluate() / predict()

    :param model_args: Arguments pertaining to model/config/tokenizer
    :param data_args: Arguments pertaining to what data to use for different flows
    :param training_args: Arguments pertaining to training loop configuration
    :model: unwrapped model to run flows on
    """

    def __init__(
        self,
        data_args: "DataTrainingArguments",
        model_args: "ModelArguments",
        training_args: "TrainingArguments",
        model: Module,
    ):
        self._data_args = data_args
        self._model_args = model_args
        self._training_args = training_args

        self.datasets = {}
        self.trainer = None
        self.tokenizer = None
        self.parent_output_dir = self._training_args.output_dir
        self._output_dir = self._training_args.output_dir

    def populate_datasets(self, tokenizer: "AutoTokenizer"):
        """
        Loads datasets for each flow based on data_args, stores a Dataset for each
        enabled flow in self.datasets

        :param tokenizer: tokenizer to use for dataset tokenization
        """
        splits = self._data_args.splits
        tokenized_datasets = {}

        def _get_split_name(inp_str):
            # strip out split name, for ex train[60%:] -> train
            match = re.match(r"(\w*)\[.*\]", inp_str)
            if match is not None:
                return match.group(1)
            return inp_str

        if splits is None:
            splits = {"all": None}
        elif isinstance(splits, str):
            splits = {_get_split_name(splits): splits}
        elif isinstance(splits, List):
            splits = {_get_split_name(s): s for s in splits}

        for split_name, split_str in splits.items():
            dataset_manager = TextGenerationDataset.load_from_registry(
                self._data_args.dataset_name,
                data_args=self._data_args,
                split=split_str,
                tokenizer=tokenizer,
            )

            store_padding_mask = False
            if self._training_args.do_oneshot and split_name == "calibration":
                store_padding_mask = True
            raw_dataset = dataset_manager.get_raw_dataset(self._model_args.cache_dir)
            tokenized_dataset = dataset_manager.tokenize_and_process(
                raw_dataset, store_padding_mask=store_padding_mask
            )
            tokenized_datasets[split_name] = tokenized_dataset

        self.datasets = make_dataset_splits(
            tokenized_datasets,
            do_train=self._training_args.do_train,
            do_eval=self._training_args.do_eval,
            do_predict=self._training_args.do_predict,
            do_oneshot=self._training_args.do_oneshot,
        )
        self.tokenizer = tokenizer

    def get_dataset_split(self, split_name: str) -> Dataset:
        """
        Retrieve a dataset split by name

        :param split_name: name of dataset split to return
        :return: dataset split labeled by split_name
        """
        return self.datasets.get(split_name)

    def one_shot(self, stage: Optional[str] = None):
        """
        Run oneshot calibration on the active model

        :param stage: which stage of the recipe to run, or None to run whole recipe
        """
        _LOGGER.info("*** One Shot ***")

        calib_data = format_calibration_data(
            tokenized_dataset=self.get_dataset_split("calibration"),
            num_calibration_samples=self._data_args.num_calibration_samples,
            accelerator=self.trainer.accelerator,
        )

        # if we don't run a forward pass after initializing the FSDP model for the
        # first time, calls to summon_full_params will fail ¯\_(ツ)_/¯
        dummy_inp = dict(next(iter(calib_data)))
        with torch.no_grad():
            dummy_inp.pop(PADDING_MASK_COLUMN_NAME, None)
            self.trainer.model(**dummy_inp)
        torch.cuda.empty_cache()

        self.trainer.one_shot(calib_data, stage=stage)

        if is_fsdp_model(self.trainer.model):
            try:
                self.trainer.save_model(output_dir=self._output_dir, _is_oneshot=True)
            except AssertionError:
                # fallback to this in the case of quantization
                unwrap_and_export_model(
                    model=self.trainer.model,
                    accelerator=self.trainer.accelerator,
                    output_dir=self._output_dir,
                    tokenizer=self.tokenizer,
                )
        else:
            save_model_and_recipe(
                model=self.trainer.model,
                save_path=self._output_dir,
                tokenizer=self.tokenizer,
            )

    def train(self, checkpoint: str, stage: Optional[str] = None):
        """
        Run trainer's training loop on train_dataset, saving the resulting model to
        output_dir

        :param checkpoint: Optional checkpoint to resume from
        :param stage: which stage of the recipe to run, or None to run whole recipe
        """
        _LOGGER.info("*** Train ***")
        train_result = self.trainer.train(
            resume_from_checkpoint=checkpoint, stage=stage
        )
        metrics = train_result.metrics
        metrics["train_samples"] = len(self.get_dataset_split("train"))
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)

        # this includes saving the state, optimizer and scheduler
        self.trainer.save_model(output_dir=self._output_dir)

    def evaluate(self):
        """
        Run trainer's evaluation loop on eval_dataset, logging the desired metrics
        """
        _LOGGER.info("*** Evaluate ***")
        metrics = self.trainer.evaluate(self.get_dataset_split("validation"))

        metrics["eval_samples"] = len(self.get_dataset_split("validation"))
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)

    def predict(self):
        """
        Run trainer's prediction loop on predict_dataset, logging the desired metrics
        """
        _LOGGER.info("*** Predict ***")
        results = self.trainer.predict(self.dataset["test"])
        metrics = results.metrics

        metrics["predict_samples"] = len(self.dataset["test"])
        self.trainer.log_metrics("predict", metrics)
        self.trainer.save_metrics("predict", metrics)

    def run_sequential_stages(self, checkpoint: Optional[str] = None):
        """
        Run the recipe stage by stage, allowing for alternating between one-shot and
        finetuning flows. Optionally save the model output at the end of each stage

        :param checkpoint: optional checkpoint to pick up a stage from
        """

        recipe_obj = Recipe.create_instance(self._training_args.recipe)
        with self.trainer.accelerator.main_process_first():
            completed_stages = get_completed_stages(self._model_args.model_name_or_path)
        self.trainer.accelerator.wait_for_everyone()

        for stage in recipe_obj.stages:
            # validate stage
            stage_name = stage.group
            run_type = stage.infer_run_type()
            if not run_type:
                raise ValueError(
                    f"a valid stage type ({[e.value for e in StageRunType]}) "
                    "must be provided in run_stages mode. Either add a run_type "
                    "attribute to each stage in the recipe or include it as part of "
                    "the stage name."
                )

            # just load structure if stage has already applied
            if stage_name in completed_stages:
                self.trainer.initialize_structure(stage=stage)
                self.trainer.accelerator.wait_for_everyone()
                continue

            # setup checkpoint dir, TODO: this should be optional
            self._output_dir = os.path.join(
                self.parent_output_dir, "stage_" + stage_name
            )
            with self.trainer.accelerator.main_process_first():
                if not os.path.exists(self._output_dir):
                    os.makedirs(self._output_dir)
                save_completed_stages(self._output_dir, completed_stages)
            self._training_args.output_dir = self._output_dir

            # run stage
            if run_type is StageRunType.ONESHOT:
                self.one_shot(stage=stage_name)
            elif run_type is StageRunType.TRAIN:
                self.train(checkpoint=checkpoint, stage=stage_name)
            checkpoint = None

            # save stage stage to checkpoint dir
            if self.trainer.accelerator.is_main_process:
                completed_stages.append(stage_name)
                save_completed_stages(self._output_dir, completed_stages)

            # setup for next stage
            session = session_manager.active_session()
            session.reset_stage()

            # log model sparsity
            with summon_full_params_context(self.trainer.model):
                if self.trainer.accelerator.is_main_process:
                    if not qat_active(self.trainer.model):
                        self.trainer.log_model_sparsification()

            # synchronize and clean up memory
            self.trainer.accelerator.wait_for_everyone()
            self.trainer.model = get_session_model()
            torch.cuda.empty_cache()
            self.trainer.accelerator.free_memory()
            self.trainer.accelerator.wait_for_everyone()
