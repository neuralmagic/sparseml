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

from typing import Optional

import torch

from sparseml.pytorch.utils.logger import BaseLogger
from sparseml.pytorch.utils.sparsification_info.module_sparsification_info import (
    ModuleSparsificationInfo,
)


__all__ = ["log_module_sparsification_info"]


def log_module_sparsification_info(
    module: torch.nn.Module, logger: BaseLogger, step: Optional[float] = None
):
    """
    Log the sparsification information for the given module to the given logger

    :param module: the module to log the sparsification information for
    :param logger: the logger to log the sparsification information to
    :param step: the global step for when the sparsification information
        is being logged. By default, is None
    """
    sparsification_info = ModuleSparsificationInfo.from_module(module)
    for tag, value in sparsification_info.loggable_items():
        if isinstance(value, dict):
            logger.log_scalars(tag=tag, step=step, values=value)
        else:
            logger.log_scalar(tag=tag, step=step, value=value)
