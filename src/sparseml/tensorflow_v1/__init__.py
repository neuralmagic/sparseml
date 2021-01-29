# Copyright 2021-present Neuralmagic, Inc.
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
Code for working with the tensorflow_v1 framework for creating /
editing models for performance in the Neural Magic System
"""

try:
    import tensorflow

    version = [int(v) for v in tensorflow.__version__.split(".")]
    if version[0] != 1 or version[1] < 8:
        raise Exception
except:
    raise RuntimeError(
        "Unable to import tensorflow. tensorflow>=1.8,<2.0 is required"
        " to use sparseml.tensorflow_v1."
    )
