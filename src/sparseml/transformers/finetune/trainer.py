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
    pass