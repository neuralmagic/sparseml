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
import os
from functools import wraps

import pytest


def get_configs_with_cadence(cadence: str, dir_path: str = "."):
    """
    Find all config files in the given directory with a matching cadence.
    :param cadence: string signifying how often to run this test. Possible values are:
        commit, daily, weekly
    :param dir_path: path to the directory in which to search for the config files
    :return List of file paths to matching configs
    """
    all_files_found = glob.glob(os.path.join(dir_path, "test*.yaml"))
    matching_files = []
    for file in all_files_found:
        with open(file) as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("cadence:"):
                    if line.split(":")[1].strip().strip('"').lower() == cadence:
                        matching_files.append(file)
                        break
    return matching_files


def skip_inactive_stage(test):
    """
    Check whether the this test's command type is active in this run. If not,
    skip test.

    :param test: test function which follows the name convention test_{command_type}_...
    """

    @wraps(test)
    def wrapped_test(self, *args, **kwargs):
        command_type = test.__name__.split("_")[1]
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
