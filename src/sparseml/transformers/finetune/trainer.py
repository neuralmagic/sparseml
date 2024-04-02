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

from typing import Any, Callable, Dict, Optional, Union

import torch
from torch.nn import Module
from transformers import Trainer as HFTransformersTrainer

from sparseml.transformers.finetune.session_mixin import SessionManagerMixIn


__all__ = ["Trainer"]


class Trainer(SessionManagerMixIn, HFTransformersTrainer):
    """
    Training implementation for running sparsification recipes with HF Trainer.

    :param model: the model to use with the trainer and apply sparsification to
    :param recipe: the recipe, if any, to apply to the modle and training
        process
    :param recipe_args: A json string, csv key=value string, or dictionary containing
        arguments to override the root arguments within the recipe such as
        learning rate or num epochs
    :param teacher: teacher model for distillation. Set to 'self' to distill
        from the loaded model or 'disable' to turn of distillation
    :param kwargs: key word arguments passed to the parent class
    """

    def __init__(
        self,
        model: Optional[Module] = None,
        model_init: Optional[Callable] = None,
        recipe: Optional[str] = None,
        recipe_args: Optional[Union[Dict[str, Any], str]] = None,
        teacher: Optional[Union[Module, str]] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            model_init=model_init,
            recipe=recipe,
            recipe_args=recipe_args,
            teacher=teacher,
            **kwargs,
        )

    def _dummy_lr_scheduler(self):
        return torch.optim.lr_scheduler.MultiplicativeLR(
            self.optimizer,
            lambda _: 1.0,
        )
