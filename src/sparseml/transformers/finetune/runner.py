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
from typing import Optional

from torch.nn import Module
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer

import sparseml.core.session as session_manager
from sparseml.core.framework import Framework
from sparseml.transformers.finetune import Trainer, TrainingArguments
from sparseml.transformers.finetune.data import TextGenerationDataset
from sparseml.transformers.finetune.data.data_args import DataTrainingArguments
from sparseml.transformers.finetune.data.data_helpers import make_dataset_splits
from sparseml.transformers.finetune.model_args import ModelArguments


_LOGGER: logging.Logger = logging.getLogger(__name__)


class StageRunner:
    def __init__(
        self,
        data_args: "DataTrainingArguments",
        model_args: "ModelArguments",
        training_args: "TrainingArguments",
        model: Module,
        teacher: Optional[Module] = None,
    ):
        self._data_args = data_args
        self._model_args = model_args
        self._training_args = training_args

        self.datasets = {}
        self.model = model
        self.teacher = teacher
        self.trainer = None

    def populate_datasets(self, tokenizer: "AutoTokenizer"):
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

    def set_trainer(self, trainer: Trainer):
        self.trainer = trainer

    def get_oneshot_dataloader(self) -> DataLoader:
        oneshot_dataset = self.datasets["calibration"]

        dataloader_params = {
            "batch_size": 1,
            "sampler": RandomSampler(oneshot_dataset),
            "collate_fn": self.trainer.data_collator,
        }

        return DataLoader(oneshot_dataset, **dataloader_params)

    def one_shot(self):
        _LOGGER.info("*** One Shot ***")

        calib_dataloader = self.get_oneshot_dataloader()
        calib_data = [inp["input_ids"] for inp in calib_dataloader]
        calib_data = calib_data[: self._data_args.num_calibration_samples]
        session_manager.apply(
            framework=Framework.pytorch,
            recipe=self._training_args.recipe,
            model=self.model,
            calib_data=calib_data,
            start=-1,
            device="cuda:0",  # TODO: don't hardcode
            copy_data=False,
        )

    def train(self, checkpoint: str):
        """
        Run trainer's training loop on train_dataset, saving the resulting model to
        output_dir

        :param checkpoint: Optional checkpoint to resume from
        """
        train_result = self.trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(self.datasets["train"])
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)

        # this includes saving the state, optimizer and scheduler
        self.trainer.save_model()

    def evaluate(self):
        """
        Run trainer's evaluation loop on eval_dataset, logging the desired metrics
        """
        _LOGGER.info("*** Evaluate ***")
        metrics = self.trainer.evaluate(self.datasets["validation"])

        metrics["eval_samples"] = len(self.datasets["validation"])
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
