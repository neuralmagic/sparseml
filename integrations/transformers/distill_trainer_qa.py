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

class DistillQuestionAnsweringTrainer(QuestionAnsweringTrainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, teacher=None, loss=None, batch_size=8, max_sequence_length=384,distill_hardness =0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function
        self.loss = loss
        self.teacher = teacher
        self.batch_size = batch_size
        self.temperature = temperature
        self.distill_hardness = distill_hardness
        self.criterion = nn.CrossEntropyLoss()
        self.max_sequence_length = max_sequence_length
        if self.teacher == None:
            self.distill_hardness = 0.0

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. Modified for Distilation using student teacher framework modified for distilation. 
        """
        input_device = inputs["input_ids"].device
        outputs = model(**inputs)
        loss = outputs['loss']
        if self.teacher is not None:
            self.teacher = self.teacher.to(input_device)
            start_logits_student = outputs["start_logits"]
            end_logits_student = outputs["end_logits"]
            start_logits_label = inputs["start_positions"]
            end_logits_label = inputs["start_positions"]
            with torch.no_grad():
                teacher_output = self.teacher(
                                input_ids=inputs["input_ids"],
                                token_type_ids=inputs["token_type_ids"],
                                attention_mask=inputs["attention_mask"],
                            )
            start_logits_teacher = teacher_output["start_logits"]
            end_logits_teacher = teacher_output["end_logits"]
            loss_start = (
                F.kl_div(
                    input=F.log_softmax(start_logits_student / self.temperature, dim=-1),
                    target=F.softmax(start_logits_teacher / self.temperature, dim=-1),
                    reduction="batchmean",
                )
                * (self.temperature ** 2)
            )
            loss_end = (
                F.kl_div(
                    input=F.log_softmax(end_logits_student / self.temperature, dim=-1),
                    target=F.softmax(end_logits_teacher / self.temperature, dim=-1),
                    reduction="batchmean",
                )
                * (self.temperature ** 2)
            )
            teacher_loss = (loss_start + loss_end) / 2.0
            loss_start = self.criterion(start_logits_student, start_logits_label)
            loss_end = self.criterion(end_logits_student, end_logits_label)
            label_loss = (loss_start + loss_end) / 2.0
            loss = ((1-self.distill_hardness) * label_loss) + (self.distill_hardness * teacher_loss)
        return (loss, outputs) if return_outputs else loss 
