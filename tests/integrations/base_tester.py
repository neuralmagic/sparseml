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
Integration Testing Dev Guide
To implement an integration-specific set of integrations tests 4 components are needed:
A. {Integration_name}_args.py file containing pydantic classes for the args of each of 
the commands (i.e. train, export, deploy).
B. {Integration_name}_tester.py file containing your class implementation with the
contents outlined below (#1-3)
C. The tests themselves, implemented in the same class as B
D. .yaml files specifying which scenarios to run (one per file)

Components needed for an integration-specific test class. More in depth information
can be found in the code (marked by the item #).
1. A mapping of commands (i.e. train, export, and deploy) to their CLI command stubs
2. A mapping of commands to the pydantic models defining their args
3. Overrides of functions as needed
"""

import os
import shutil
import subprocess
from multiprocessing.sharedctypes import Value
from typing import Dict, List, Tuple, Union

import pytest
import yaml
from pydantic import BaseModel

from tests.integrations.base_args import (
    DummyDeployArgs,
    DummyExportArgs,
    DummyTrainArgs,
)
from tests.integrations.helpers import get_configs_with_cadence


pytest.mark.parametrize(
    "config_path",
    get_configs_with_cadence(os.environ["NM_TEST_CADENCE"], indirect=True),
)


class BaseIntegrationTester:
    # 1 The command stubs should not include tasks. To modify the command stubs to add
    # tasks (or make any other changes), ovveride the get_base_commands() method.
    command_stubs = {
        "train": "sparseml.foo_integration.train",
        "export": "sparseml.foo_integration.export",
        "deploy": "sparseml.foo_integration.deploy",
    }
    # 2 Based on the CLI arguments that each command can accept, a pydantic model should
    # be created to enforce argument validation and provide easy reference to possible
    # arg values for devs
    command_args_classes = {
        "train": DummyTrainArgs,
        "export": DummyExportArgs,
        "deploy": DummyDeployArgs,
    }

    @pytest.fixture(scope="class")
    def setup(self, config_path):
        # A path for a config file which mathes our integration and cadence is provided
        # The raw config can contain multiple command stages (train, export, deploy)
        raw_config = yaml.safe_load(config_path)
        # Remove cadence for easier processing. It's saved, but not utilized later
        self.cadence = raw_config.pop("cadence")

        # Will be used in tests to skip test if type doesn't apply
        self.command_types = [_type for _type in raw_config]

        # 3 The command stub is fetched. See method description
        self.command_stubs_final = self.get_base_commands(raw_config)

        # 3 extract test args.
        self.test_args = self.get_test_args(raw_config)

        # Command-specific args are loaded into their respective pydantic classes.
        # pre_args are CLI commands that can precede the python call
        # e.g. "torch distributed launch" or env variable setting. If any are present,
        # they are read in as well.
        self.configs = {
            _type: {
                "pre_args": config.pop("pre_args", None),
                "args": self.command_args_classes[_type](config.get("command_args")),
            }
            for _type, config in raw_config
        }
        # Generate full CLI commands for each stage of run
        self.commands = self.compose_command_scripts(self.configs)
        # 3 Override to save any information which may be needed for testing post-run
        self.capture_pre_run_state()
        # All commands are run in order
        self.run_commands(self.commands)
        yield  # all tests are run here
        # 3 WIP post-test teardown
        self.cleanup_files()
        self.check_file_creation()
        # pytest.skip(f"Not a {class_command_type} command")

    @classmethod
    def get_base_commands(cls, configs: Dict[str, Union[str, BaseModel]]):
        """
        Returns the command stubs to use. e.g. "sparseml.yolov5.train".
        If the stub for the integration needs to be determined based on the configs
        (e.g. sparseml.transformers.{task}.train), ovveride this function

        :param configs: unprocessed configs dict
        :return: dict of type {"command_type": "command_stub"}
        """
        return cls.command_stubs

    @classmethod
    def get_test_args(cls, configs):
        """
        Return test args. Test args are open ended in function and are meant to
        interact with the tests.Example target args are "target F1 score" and
        "flag to not test onnx model structure"

        WIP, but there may not be much here than can be generalizable. i.e. this
        function my have to be implemented on the integration level
        """
        return None

    @classmethod
    def compose_command_scripts(cls, configs):
        """
        For each command, create the full CLI command by combining the pre args,
        command stub, and run args.
        """
        commands = []
        for _type, config in configs:
            if _type not in cls.command_stubs:
                raise ValueError(f"{_type} is not a valid command type")
        commands.append(
            cls.create_command_script(
                config["pre_args"], cls.command_stubs_final[_type], config["args"]
            )
        )
        return commands

    @classmethod
    def create_command_script(cls, pre_args: str, command_stub: str, config: BaseModel):
        """
        Handles logic for converting pydantic classes into valid argument strings.
        This should set arg standards for all integrations and should generally not
        be overridden. If the need to override comes up, consider updating this method
        instead.
        """
        args_dict = config.dict()
        args_string_list = []
        for key, value in args_dict:
            # Handles bool type args (e.g. --do-train)
            if isinstance(value, bool):
                if value:
                    args_string_list.append("--" + key)
            # Handles args that are both bool and value based
            elif isinstance(value, Tuple):
                if not isinstance(value[0], bool):
                    raise ValueError(
                        "Tuple types are used to specify bool args with "
                        "a defined const value. {value} does not qualify"
                    )
                if value:
                    args_string_list.append("--" + key + " " + value[1])
            # Handles args that have multiple values after the keyword.
            # e.g. --freeze-layers 0 10 15
            elif isinstance(value, List):
                args_string_list.append("--" + key + " ".join(value))
            # Handles the most straightforward case of keyword followed by value
            # e.g. --epochs 30
            else:
                args_string_list.append("--" + key + " " + value)
        args_string_combined = " ".join(args_string_list)
        return " ".join([pre_args, command_stub, args_string_combined])

    def capture_pre_run_state(self):
        """
        Store pre-run information which will be relevant for post-run testing
        """
        self._start_file_count = sum(len(files) for _, _, files in os.walk(r"."))

    def run_commands(self, kwargs_list: List[dict] = [{}]):
        """
        Execute CLI commands in order
        """
        for command, kwargs in zip(self.commands, kwargs_list):
            self.share_stage_information()
            subprocess.call(command, **kwargs)

    def share_stage_information(self):
        """
        Optional function for saving information between stages. WIP - will likely be
        moved or changed
        """
        pass

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

    def cleanup_files(self, dir):
        """
        Dummy cleanup function. Will be fleshed out later
        """
        if os.path.isdir(dir):
            shutil.rmtree(dir)

    def check_file_creation(self, dir):
        """
        Dummy function for testing for file creation. Will be fleshed out later
        """
        self._end_file_count = sum(len(files) for _, _, files in os.walk(r"."))
        assert self._start_file_count >= self._end_file_count, (
            f"{self._end_file_count - self._start_file_count} files created during "
            "pytest run"
        )
