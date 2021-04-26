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
Recalibration code for the TensorFlow framework.
Handles things like model pruning and increasing activation sparsity.
"""

# flake8: noqa

from ..base import check_keras_install as _check_keras_install
from .manager import *
from .mask_pruning import *
from .mask_pruning_creator import *
from .modifier import *
from .modifier_epoch import *
from .modifier_lr import *
from .modifier_params import *
from .modifier_pruning import *
from .utils import *


_check_keras_install()  # TODO: remove once files within package load without installs
