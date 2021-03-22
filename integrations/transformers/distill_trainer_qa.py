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

from trainer_qa import QuestionAnsweringTrainer
class HardDistillationLoss(nn.Module):
    def __init__(self, teacher: nn.Module, criterion, batch_size, max_sequence_length, hardness):
        super().__init__()
        self.teacher = teacher
        self.max_sequence_length = max_sequence_length
        self.hardness = hardness
        if criterion == None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion #nn.MSELoss(reduction="mean"), nn.KLDivLoss(reduction="batchmean"), 

    def compute_loss(self, logits, positions,):
        onehot = torch.nn.functional.one_hot(positions.to(torch.int64), self.max_sequence_length)
        logprob = torch.softmax(logits, dim=-1, dtype=torch.float32)
        return -torch.mean(torch.sum(onehot * logprob))

    def forward(self, inputs: Tensor, outputs: Union[Tensor, Tensor], labels: Tensor) -> Tensor:
        base_loss = self.compute_loss(outputs["start_logits"], inputs["start_positions"])
        base_loss += self.compute_loss(outputs["end_logits"], inputs["end_positions"])
        base_loss /= 2
        teacher_loss = self.compute_loss(outputs["start_logits"], inputs["teacher_start_positions"])
        teacher_loss += self.compute_loss(outputs["end_logits"], inputs["teacher_end_positions"])
        teacher_loss /= 2
        return (1-self.hardness) * base_loss + (self.hardness * teacher_loss)

class DistillQuestionAnsweringTrainer(QuestionAnsweringTrainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, teacher=None, loss=None, batch_size=8, max_sequence_length=384,hardness =0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function
        self.loss = loss
        self.teacher = teacher
        self.batch_size = batch_size
        self.hardness = hardness
        self.max_sequence_length = max_sequence_length
        self.criterion = HardDistillationLoss(self.loss, self.teacher, self.batch_size, self.max_sequence_length, self.hardness)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. Modified for distilation
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        loss = self.criterion(inputs, outputs, labels) # Modified for distilation
        return (loss, outputs) if return_outputs else loss