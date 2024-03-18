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

import shutil
import unittest
from pathlib import Path
from typing import Optional

import pytest

from huggingface_hub import snapshot_download
from parameterized import parameterized, parameterized_class
from sparseml import export
from tests.custom_test import CustomIntegrationTest
from tests.data import CustomTestConfig
from tests.testing_utils import parse_params


CONFIGS_DIRECTORY = "tests/sparseml/export/transformers/generation_configs"


@pytest.mark.integration
@parameterized_class(parse_params(CONFIGS_DIRECTORY))
class TestGenerationExportIntegration(unittest.TestCase):
    stub = None
    task = None

    def setUp(self):
        self.tmp_path = Path("tmp")
        self.tmp_path.mkdir(exist_ok=True)

        model_path = self.tmp_path / "model"
        self.target_path = self.tmp_path / "target"
        self.source_path = snapshot_download(self.stub, local_dir=model_path)

    def test_export_with_external_data(self):
        export(
            source_path=self.source_path,
            target_path=self.target_path,
            task=self.task,
            save_with_external_data=True,
        )
        assert (self.target_path / "deployment" / "model.onnx").exists()
        assert (self.target_path / "deployment" / "model.data").exists()

    def tearDown(self):
        shutil.rmtree(self.tmp_path)


class TestGenerationExportIntegrationCustom(CustomIntegrationTest):
    custom_scripts_directory = (
        "tests/sparseml/export/transformers/generation_configs/custom_script"
    )
    custom_class_directory = (
        "tests/sparseml/export/transformers/generation_configs/custom_class"
    )

    # TODO: make enum
    @parameterized.expand([parse_params(custom_scripts_directory, type="custom")])
    def test_custom_scripts(self, config: Optional[CustomTestConfig] = None):
        super().test_custom_scripts(config)

    @parameterized.expand([parse_params(custom_class_directory, type="custom")])
    def test_custom_class(self, config: Optional[CustomTestConfig] = None):
        super().test_custom_class(config)
