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

import sys

from torch.nn import Module


__all__ = ["compare_model"]


def compare_model(model_one: Module, model_two: Module, same: bool):
    total_dif = 0.0

    for param_one, param_two in zip(model_one.parameters(), model_two.parameters()):
        assert param_one.data.shape == param_two.data.shape
        total_dif += (param_one.data - param_two.data).abs().sum()

    if same:
        assert total_dif < sys.float_info.epsilon
    else:
        assert total_dif > sys.float_info.epsilon
