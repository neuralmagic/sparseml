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

import glob
import inspect
import os
from functools import wraps

import pytest
import yaml


def get_configs_with_cadence(cadence: str, dir_path: str = "."):
    all_files_found = glob.glob(
        "/home/konstantin/Source/sparseml/tests/integrations/yolov5/test*.test"
    )
    matching_files = []
    print(all_files_found)
    for file in all_files_found:
        config = yaml.safe_load(file)
        if config.get("cadence") == cadence:
            matching_files.append(config)
        # read one line a time


def skip_inactive_stage(test):
    @wraps(test)
    def wrapped_test(self, *args, **kwargs):
        command_type = inspect.currentframe().f_code.co_name.split("_")[1]
        if command_type not in self.command_stubs:
            raise ValueError(
                "Invalid test function definition. Test names must take the form "
                f"test_{{CommandType}}_... Found instead {command_type} for "
                "{Command_type}"
            )
        if command_type not in self.command_types:
            pytest.skip(f"No {command_type} stage active. Skipping test")
        test(self, *args, **kwargs)

    return wrapped_test
