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

import pytest

from click.testing import CliRunner
from sparseml.pytorch import recipe_template
from sparseml.pytorch.recipe_template.cli import main


def test_function_entrypoint():
    recipe_template()


@pytest.mark.parametrize(
    "command",
    [
        ["--pruning", "true", "--quantization", "true"],
        ["--pruning", "true", "--quantization", "true", "--distillation"],
        ["--quantization", "true", "--target", "vnni", "--lr", "constant"],
        [
            "--quantization",
            "true",
            "--target",
            "vnni",
            "--lr",
            "constant",
            "--distillation",
        ],
    ],
)
def test_docstring_cli_examples(command, tmp_path):
    runner = CliRunner()
    command.extend(["--file_name", str(tmp_path / "temp.md")])
    result = runner.invoke(main, command)
    assert result.exit_code == 0
