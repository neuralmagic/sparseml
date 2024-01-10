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
from typing import List, Optional
import json

import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import AutoTokenizer

import sparseml.core.session as session_manager
from sparseml.core.recipe import Recipe, StageRunType
from sparseml.pytorch.model_load.helpers import get_session_model, save_model_and_recipe
from sparseml.transformers.finetune import TrainingArguments
from sparseml.transformers.finetune.data import TextGenerationDataset
from sparseml.transformers.finetune.data.data_args import DataTrainingArguments
from sparseml.transformers.finetune.data.data_helpers import make_dataset_splits
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
        self._output_dir = self._training_args.output_dir

    def populate_datasets(self, tokenizer: "AutoTokenizer"):
        """
        Loads datasets for each flow based on data_args, stores a Dataset for each
        enabled flow in self.datasets

        :param tokenizer: tokenizer to use for dataset tokenization
        """
        splits = self._data_args.splits
        tokenized_datasets = {}
        if splits is None:
            splits = {"all": None}
        elif isinstance(splits, str):
            splits = {splits: splits}
        elif isinstance(splits, List):
            splits = {s: s for s in splits}

        for split_name, split_str in splits.items():
            dataset_manager = TextGenerationDataset.load_from_registry(
                self._data_args.dataset_name,
                data_args=self._data_args,
                split=split_str,
                tokenizer=tokenizer,
            )
            raw_dataset = dataset_manager.get_raw_dataset(self._model_args.cache_dir)
            tokenized_dataset = dataset_manager.tokenize_and_process(raw_dataset)
            tokenized_datasets[split_name] = tokenized_dataset

        self.datasets = make_dataset_splits(
            tokenized_datasets,
            do_train=self._training_args.do_train or self._training_args.run_stages,
            do_eval=self._training_args.do_eval,
            do_predict=self._training_args.do_predict,
            do_oneshot=self._training_args.do_oneshot or self._training_args.run_stages,
        )
        self.tokenizer = tokenizer

    def get_dataset_split(self, split_name: str) -> Dataset:
        """
        Retrieve a dataset split by name

        :param split_name: name of dataset split to return
        :return: dataset split labeled by split_name
        """
        return self.datasets.get(split_name)

    def format_calibration_data(self) -> List[torch.Tensor]:
        """
        Creates a dataloader out of the calibration dataset split, trimming it to
        the desired number of calibration samples

        :return: list of trimmed calibration data tensors
        """
        oneshot_dataset = self.get_dataset_split("calibration")

        dataloader_params = {
            "batch_size": 1,
            "sampler": RandomSampler(oneshot_dataset),
            "collate_fn": self.trainer.data_collator,
        }

        calib_dataloader = DataLoader(oneshot_dataset, **dataloader_params)
        parsed_calib_data = [inp["input_ids"] for inp in calib_dataloader]
        return parsed_calib_data[
            : min(self._data_args.num_calibration_samples, len(parsed_calib_data))
        ]

    def one_shot(self, stage: Optional[str] = None):
        """
        Run oneshot calibration on the active model

        :param stage: which stage of the recipe to run, or None to run whole recipe
        """
        _LOGGER.info("*** One Shot ***")

        calib_data = self.format_calibration_data()
        self.trainer.one_shot(calib_data, stage=stage)

        if is_fsdp_model(self.trainer.model):
            try:
                self.trainer.save_model(output_dir=self._output_dir)
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

    def run_sequential_stages(self):
        """
        Run the recipe stage by stage, allowing for alternating between one-shot and
        finetuning flows. Optionally save the model output at the end of each stage
        """

        recipe_obj = Recipe.create_instance(self._training_args.recipe)
        stage_path = os.path.join(self._model_args.model_name_or_path, "completed_stages.json")
        with self.trainer.accelerator.main_process_first():
            if os.path.exists(stage_path):
                with open(stage_path) as stage_file:
                    stage_data = json.load(stage_file)
                completed_stages = stage_data['completed']
            else:
                completed_stages = []

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

            # just load structure if already applied
            if stage_name in completed_stages:
                self.trainer.initialize_structure(stage=stage)
                continue

            # setup checkpoint dir, TODO: this should be optional
            self._output_dir = os.path.join(
                self._training_args.output_dir, "stage_" + stage_name
            )
            with self.trainer.accelerator.main_process_first():
                if not os.path.exists(self._output_dir):
                    os.makedirs(self._output_dir)

            # run stage
            if run_type is StageRunType.ONESHOT:
                self.one_shot(stage=stage_name)
            elif run_type is StageRunType.TRAIN:
                if not is_fsdp_model(self.trainer.model):
                    self.trainer.model.to("cpu")
                self.train(checkpoint=None, stage=stage_name)

            # save stage stage to checkpoint dir
                
            if self.trainer.accelerator.is_main_process:
                completed_stages.append(stage_name)
                stage_path = os.path.join(self._output_dir, "completed_stages.json")
                with open(stage_path, 'w') as out_file:
                    json.dump({"completed": completed_stages}, out_file)


            # setup for next stage
            session = session_manager.active_session()
            session.reset_stage()

            with summon_full_params_context(self.trainer.model):
                if self.trainer.accelerator.is_main_process:
                    if not qat_active(self.trainer.model):
                        self.trainer.log_model_sparsification()
            self.trainer.accelerator.wait_for_everyone()
            self.trainer.model = get_session_model()
