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

try:
    import keras as native_keras
except ModuleNotFoundError:
    native_keras = None

import tensorflow


__all__ = [
    "assign",
    "keras",
]


keras = native_keras if native_keras is not None else tensorflow.keras


def assign(lhs, rhs, name=None):
    if hasattr(tensorflow, "assign"):
        return tensorflow.assign(lhs, rhs, name=name)
    else:
        return lhs.assign(rhs, name=name)
