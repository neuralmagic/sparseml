# neuralmagic: no copyright
# flake8: noqa
# fmt: off
# isort: skip_file
#!/usr/bin/env python
# coding=utf-8
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
from typing import Union

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

from transformers import Trainer, is_datasets_available, is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput

class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        target_p = inputs["labels"]
        outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        logits = outputs[0]
        loss = -torch.sum(target_p * logits.log_softmax(dim=-1), axis=-1).mean()
        if return_outputs:
            return loss, outputs
        return loss

class HardDistillationLoss(nn.Module):
    def __init__(self, teacher: nn.Module, criterion=None):
        super().__init__()
        self.teacher = teacher
        if criterion == None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion #nn.MSELoss(reduction="mean"), nn.KLDivLoss(reduction="batchmean"), 
        
    def forward(self, inputs: Tensor, outputs: Union[Tensor, Tensor], labels: Tensor) -> Tensor:
        # outputs contains booth predictions, one with the cls token and one with the dist token
        outputs_cls, outputs_dist = outputs
        base_loss = self.criterion(outputs_cls, labels)
        with torch.no_grad():
            teacher_outputs = self.teacher(inputs)
        teacher_labels = torch.argmax(teacher_outputs, dim=1)
        teacher_loss = self.criterion(outputs_dist, teacher_labels)
        return 0.5 * base_loss + 0.5 * teacher_loss

class DistillQuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, teacher=None, loss=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function
        self.teacher = teacher
        self.loss = loss
        self.criterion = HardDistillationLoss(self.teacher, self.loss)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. Modified for distilation
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        loss = self.criterion(inputs, outputs, labels)
        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                eval_dataloader,
                description="Evaluation",
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset.set_format(
                type=eval_dataset.format["type"],
                columns=list(eval_dataset.features.keys()),
            )

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(
                eval_examples, eval_dataset, output.predictions
            )
            metrics = self.compute_metrics(eval_preds)

            self.log(metrics)
        else:
            metrics = {}

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        return metrics

    def predict(self, test_dataset, test_examples, ignore_keys=None):
        test_dataloader = self.get_test_dataloader(test_dataset)
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                test_dataloader,
                description="Evaluation",
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        if isinstance(test_dataset, datasets.Dataset):
            test_dataset.set_format(
                type=test_dataset.format["type"],
                columns=list(test_dataset.features.keys()),
            )

        eval_preds = self.post_process_function(
            test_examples, test_dataset, output.predictions
        )
        metrics = self.compute_metrics(eval_preds)

        return PredictionOutput(
            predictions=eval_preds.predictions,
            label_ids=eval_preds.label_ids,
            metrics=metrics,
        )