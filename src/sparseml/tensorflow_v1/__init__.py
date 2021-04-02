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
Functionality for working with and sparsifying Models in the TensorFlow 1.x framework
"""

# flake8: noqa

import os as _os

from .base import (
    check_tensorflow_install,
    check_tf2onnx_install,
    require_tensorflow,
    require_tf2onnx,
    tensorflow,
    tensorflow_err,
    tf2onnx,
    tf2onnx_err,
    tf_compat,
)


if not _os.getenv("SPARSEML_IGNORE_TFV1", False):
    # TODO: remove once files within package load without installs
    check_tensorflow_install()


from .framework import detect_framework, framework_info, is_supported
from .sparsification import sparsification_info
