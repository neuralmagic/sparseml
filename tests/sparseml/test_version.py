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

from sparseml.version import (
    __version__,
    version,
    version_bug,
    version_major,
    version_major_minor,
    version_minor,
)


def test_version():
    assert __version__
    assert version
    assert version_major == version.split(".")[0]
    assert version_minor == version.split(".")[1]
    assert version_bug == version.split(".")[2]
    assert version_major_minor == f"{version_major}.{version_minor}"
