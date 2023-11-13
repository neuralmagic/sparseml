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

from itertools import cycle
from typing import Callable, List, Optional

import torch
from torch.nn import Module

from sparseml.pytorch.utils import tensors_module_forward, tensors_to_device


def run_calibration_forward(
    model: Module,
    calibration_dataloader: List,
    num_calibration_steps: Optional[int] = None,
    calibration_function: Optional[Callable] = None,
    device: Optional[str] = None,
):
    """
    Helper function used by one-shot modifiers, runs calibration data through a model to
    update modifier statistics and trigger hooks

    :param model: PyTorch model to run
    :param calibration_dataloader: data to use for calibration
    :param num_calibration_steps: number of items in calibration_dataloader to process,
    None to process all available data
    :param calibration_function: option to pass a custom forward function for model
    :param device: option to move the model to a specific device before calibration
    """
    model.eval()

    forward_fn: Callable = (
        calibration_function if calibration_function else tensors_module_forward
    )

    # move model to optional specified device if it is not already there
    model_device = next(model.parameters()).device
    if device is not None and model_device != device:
        model.to(device)
        model_device = next(model.parameters()).device
    _dataloader = (
        calibration_dataloader
        if num_calibration_steps is None
        else cycle(calibration_dataloader)
    )

    # run through the calibration data
    for batch_idx, batch in enumerate(_dataloader):
        if num_calibration_steps and batch_idx >= num_calibration_steps:
            break
        batch = tensors_to_device(batch, model_device)
        with torch.no_grad():
            forward_fn(batch, module=model)
