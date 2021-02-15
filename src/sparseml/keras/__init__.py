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

    if tensorflow.__version__ < "2.1.0":
        raise RuntimeError("TensorFlow >= 2.1.0 is required, found {}".format(version))
except:
    raise RuntimeError(
        "Unable to import tensorflow. TensorFlow>=2.1.0 is required"
        " to use sparseml.keras."
    )


try:
    import keras

    v = keras.__version__
    if v < "2.4.3":
        raise RuntimeError(
            "Native keras is found and will be used, but required >= 2.4.3, found {}".format(
                v
            )
        )
except:
    pass
