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
Code for working with the keras framework for creating /
editing models for performance in the Neural Magic System
"""

# flake8: noqa

try:
    import tensorflow

    version = [int(v) for v in tensorflow.__version__.split(".")]
    if version[0] != 2 or version[1] < 2:
        raise Exception
except:
    raise RuntimeError(
        "Unable to import tensorflow. tensorflow>=2.2 is required"
        " to use sparseml.keras."
    )
