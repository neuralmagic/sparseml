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

import tensorflow as tf


__all__ = [
    "tf_compat",
    "tf_compat_div",
]


tf_compat = (
    tf
    if not hasattr(tf, "compat") or not hasattr(getattr(tf, "compat"), "v1")
    else tf.compat.v1
)  # type: tf
tf_compat_div = (
    tf_compat.div
    if not hasattr(tf_compat, "math")
    or not hasattr(getattr(tf_compat, "math"), "divide")
    else tf_compat.math.divide
)
