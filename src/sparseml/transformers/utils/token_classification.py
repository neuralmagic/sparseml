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

"""
Training utilities for text classification / GLUE tasks
"""

from typing import Any, Dict, List, Optional, Union

import torch
from transformers import Trainer

from sparseml.transformers.utils.trainer import SparseMLTrainer


__all__ = ["SparseMLNERTrainer"]


class SparseMLNERTrainer(SparseMLTrainer, Trainer):
    """
    Trainer for running sparsification recipes with NER training

    :param model_name_or_path: path to model directory to be trained
    :param recipes: list of paths to recipes for model sparsification or string
        recipes for sparsification. Can also be single string path or recipe
    :param teacher: teacher model for distillation. Default is None
    :param recipe_args: Dictionary of recipe variables to override or json
        loadable string of those args. Default is None
    :param args: arguments passed into parent class
    :param kwargs: key word arguments passed to the parent class
    """

    def __init__(
        self,
        model_name_or_path: str,
        recipes: Union[str, List[str]],
        teacher: Optional[torch.nn.Module] = None,
        recipe_args: Union[Dict[str, Any], str] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            recipes=recipes,
            teacher=teacher,
            recipe_args=recipe_args,
            teacher_input_keys=None,
            *args,
            **kwargs,
        )
