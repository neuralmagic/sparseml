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

import runpy
import unittest
from typing import Optional

import pytest

from tests.data import CustomTestConfig


class CustomTestCase(unittest.TestCase):
    ...


# TODO: consider breaking this up into two classes, similar to non-custom
# integration tests. Could then make use of parameterize_class instead
@pytest.mark.custom
class CustomIntegrationTest(unittest.TestCase):
    custom_scripts_directory: str = None
    custom_class_directory: str = None

    def test_custom_scripts(self, config: Optional[CustomTestConfig] = None):
        if config is None:
            self.skipTest("No custom scripts found. Testing test")
        script_path = f"{self.custom_scripts_directory}/{config.script_path}"
        runpy.run_path(script_path)

    def test_custom_class(self, config: Optional[CustomTestConfig] = None):
        if config is None:
            self.skipTest("No custom class found. Testing test")
        loader = unittest.TestLoader()
        tests = loader.discover(self.custom_class_directory, pattern=config.script_path)
        testRunner = unittest.runner.TextTestRunner()
        output = testRunner.run(tests)
        for out in output.errors:
            raise Exception(output[-1])

        for out in output.failures:
            assert False
        assert True
