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

from subprocess import PIPE, CalledProcessError, run

import pytest

from sparseml import recipe_template


def test_cli_entrypoint_invocation():
    try:
        output = run(["sparseml.recipe_template"], stdout=PIPE, stderr=PIPE)
        assert (
            "NotImplementedError" in output.stdout.decode()
            or "NotImplementedError" in output.stderr.decode()
        )
    except CalledProcessError as e:
        assert "NotImplementedError" in e.stderr.decode()


def test_function_entrypoint():
    with pytest.raises(NotImplementedError):
        recipe_template()
