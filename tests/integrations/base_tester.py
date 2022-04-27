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

import os
import shutil
import subprocess

import pytest

from .helpers import config  # noqa


class BaseIntegrationTester:
    @pytest.fixture(scope="class")
    def setup(self, config, command_type):  # noqa
        raw_config, command_types = config
        if command_type not in command_types:
            pytest.skip(f"Not a {command_type} command")
        self._command_config = raw_config["command_types"][command_type]
        self._start_file_count = sum(len(files) for _, _, files in os.walk(r"."))
        self._command = self.create_script_command(self._command_config)
        self._target = self.get_acceptable_target_range(self._command_config)
        self.run_command(self._command_config)
        yield
        self.cleanup_files()
        self.check_file_creation()

    @classmethod
    def create_script_command(cls, config, base_command=""):  # noqa
        raw_value_arguments = config.get("command_args")
        if "flags" in raw_value_arguments:
            raw_flags = config.pop("flags")
        value_arguments = " ".join(
            [
                "--" + str(key) + " " + str(item)
                for key, item in raw_value_arguments.items()
            ]
        )
        flags = " ".join("--" + flag for flag in raw_flags.items())
        return " ".join([base_command, value_arguments, flags])

    @classmethod
    def get_acceptable_target_range(cls, config):  # noqa
        target = config.get("target")
        return [-target["std"], target["std"]] + target["mean"] if target else None

    @classmethod
    def run_command(cls, command: str, kwargs: dict = {}):
        subprocess.call(command, **kwargs)

    def update_config(self, update_dict: dict):
        self._command_config.update(update_dict)
        self._command = self.create_script_command()

    def cleanup_files(self, dir):
        if os.path.isdir(dir):
            shutil.rmtree(dir)

    def check_file_creation(self, dir):
        self._end_file_count = sum(len(files) for _, _, files in os.walk(r"."))
        assert self._start_file_count >= self._end_file_count, (
            f"{self._end_file_count - self._start_file_count} files created during "
            "pytest run"
        )


class BaseTrainTester(BaseIntegrationTester):
    def test_checkpoint_save(self, setup):
        pass

    def test_target_metric_compliance(self):
        pass


class BaseExportTester(BaseIntegrationTester):
    def test_onnx_nodes(self, setup):
        pass


class BaseDeployTester(BaseIntegrationTester):
    def test_inference_target(self, setup):
        pass
