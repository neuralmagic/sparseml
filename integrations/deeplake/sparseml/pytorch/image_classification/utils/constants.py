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
Constants for PyTorch Image Classification Integrations
"""
from inspect import isclass

import torch


__all__ = ["OPTIMIZERS", "DEFAULT_OPTIMIZER", "AVAILABLE_DATASETS"]

OPTIMIZERS = [
    key
    for key in torch.optim.__dict__.keys()
    if isclass(torch.optim.__dict__[key]) and key != "Optimizer"
]

# Early exit if no optimizer is found
if not OPTIMIZERS:
    raise RuntimeError(
        "No optimizers found in torch.optim. "
        "Please install a torch optimizer to use this integration."
    )

DEFAULT_OPTIMIZER = "SGD" if "SGD" in OPTIMIZERS else OPTIMIZERS[0]
AVAILABLE_DATASETS = ["cifar", "imagenet", "imagenette"]
