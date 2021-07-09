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

class DistillGlueTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, teacher=None, loss=None, batch_size=8, max_sequence_length=384,distill_hardness=1.0, temperature=2.0, **kwargs):
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
        if self.teacher is None:
            self.distill_hardness = 0

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. Modified for Distilation using student teacher framework modified for distilation. 
        """
        outputs = model(**inputs)
        loss = outputs["loss"]
        logits_student = outputs["logits"]
        if self.teacher is not None:
            input_device = inputs["input_ids"].device
            self.teacher = self.teacher.to(input_device)
            with torch.no_grad():
                teacher_outputs = self.teacher(
                        input_ids=inputs["input_ids"],
                        token_type_ids=inputs["token_type_ids"],
                        attention_mask=inputs["attention_mask"],
                )
                logits_teacher = teacher_outputs["logits"]
            loss_distill = F.kl_div( input=logits_student, target=logits_teacher, reduction="batchmean",) * (self.temperature ** 2)
            loss = ((1-self.distill_hardness) * loss) + torch.abs((self.distill_hardness * loss_distill))
        return (loss, outputs) if return_outputs else loss   
