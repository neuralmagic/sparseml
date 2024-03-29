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

from typing import Optional

import pytest

from parameterized import parameterized
from tests.custom_test import CustomIntegrationTest
from tests.data import CustomTestConfig
from tests.testing_utils import parse_params


@pytest.mark.custom
class TestExampleIntegrationCustom(CustomIntegrationTest):
    """
    Integration test class which uses the base CustomIntegrationTest class.
    """

    custom_scripts_directory = "tests/examples/generation_configs/custom_script"
    custom_class_directory = "tests/examples/generation_configs/custom_class"

    @parameterized.expand(parse_params(custom_scripts_directory, type="custom"))
    def test_custom_scripts(self, config: Optional[CustomTestConfig] = None):
        super().test_custom_scripts(config)

    @parameterized.expand(parse_params(custom_class_directory, type="custom"))
    def test_custom_class(self, config: Optional[CustomTestConfig] = None):
        super().test_custom_class(config)
