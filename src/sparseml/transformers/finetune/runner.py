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
from typing import List

import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import AutoTokenizer

import sparseml.core.session as session_manager
from sparseml.core.framework import Framework
from sparseml.pytorch.model_load.helpers import fallback_to_cpu, save_model_and_recipe
from sparseml.transformers.finetune import Trainer, TrainingArguments
from sparseml.transformers.finetune.data import TextGenerationDataset
from sparseml.transformers.finetune.data.data_args import DataTrainingArguments
from sparseml.transformers.finetune.data.data_helpers import make_dataset_splits
from sparseml.transformers.finetune.model_args import ModelArguments


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
        self.model = model
        self.trainer = None
        self.tokenizer = None

    def populate_datasets(self, tokenizer: "AutoTokenizer"):
        """
        Loads datasets for each flow based on data_args, stores a Dataset for each
        enabled flow in self.datasets

        :param tokenizer: tokenizer to use for dataset tokenization
        """
        splits = self._data_args.splits
        tokenized_datasets = {}
        if self._data_args.splits is None:
            splits = {"all": None}
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
            self._training_args.do_train,
            self._training_args.do_eval,
            self._training_args.do_predict,
            self._training_args.do_oneshot,
        )
        self.tokenizer = tokenizer

    def set_trainer(self, trainer: Trainer):
        """
        :param trainer: update trainer
        """
        self.trainer = trainer

    def set_model(self, model: Module):
        """
        :param model: update pytorch model
        """
        self.model = model

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

    def one_shot(self):
        """
        Run oneshot calibration on the active model
        """
        _LOGGER.info("*** One Shot ***")

        calib_data = self.format_calibration_data()
        oneshot_device = fallback_to_cpu(self._training_args.oneshot_device)
        session_manager.apply(
            framework=Framework.pytorch,
            recipe=self._training_args.recipe,
            model=self.model,
            calib_data=calib_data,
            start=-1,
            device=oneshot_device,
            copy_data=False,
        )

        save_model_and_recipe(
            model=self.model,
            save_path=self._training_args.output_dir,
            tokenizer=self.tokenizer,
        )

    def train(self, checkpoint: str):
        """
        Run trainer's training loop on train_dataset, saving the resulting model to
        output_dir

        :param checkpoint: Optional checkpoint to resume from
        """
        train_result = self.trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(self.get_dataset_split("train"))
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)

        # this includes saving the state, optimizer and scheduler
        self.trainer.save_model()

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
