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
Recalibration code for the PyTorch framework.
Handles things like model pruning and increasing activation sparsity.
"""

# flake8: noqa

from ..base import check_torch_install as _check_torch_install
from .analyzer_as import *
from .analyzer_module import *
from .analyzer_pruning import *
from .manager import *
from .mask_creator_pruning import *
from .mask_pruning import *
from .mask_pruning_scorer import *
from .optimizer import *
from .sensitivity_as import *
from .sensitivity_lr import *
from .sensitivity_pruning import *


_check_torch_install()  # TODO: remove once files within package load without installs
