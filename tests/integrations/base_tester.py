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
To implement an integration-specific set of integrations tests 4 components are needed:

A. {Integration_name}_args.py file containing pydantic classes for the args of each of 
    the commands (i.e. train, export, deploy).

B. {Integration_name}_tester.py file containing integration testing class inherited 
    from BaseIntegrationTester

C. Tests to run after commands are run. Tests should be implemented in the tester class
    described in B and should be decorated by @skip_inactive_stage found in helpers.py

D. .yaml file for each scenario to run
"""

import os
import shutil
import subprocess
from multiprocessing.sharedctypes import Value
from typing import Dict, List, Tuple, Union

import pytest
import yaml
from pydantic import BaseModel

from tests.integrations.helpers import get_configs_with_cadence


class BaseIntegrationTester:
    """
    Base class for testing integrations through train, export, and deploy scripts.

    Each of train, export, and deploy are "command types" and can constitute a stage
    in a multi-stage run.

    Each integration will implement a subclass of this class, with the following
    fields and functions filled out:

    :field command_stubs: Mapping from command type to the respective CLI command.
        Note that the stubs can be modified via the get_root_commands() function
    :field command_args_classes: Mapping from command type to the pydnatic class
        which holds the CLI args for that command
    :function teardown: Perform the appropriate post-testing teardown, including
        file cleanup
    :function get_root_commands: [Optional] If the CLI root commands are dynamic
        (e.g. transformers commands are task-dependent), override this function to
        return the correct stubs
    :function capture_pre_run_state: [Optional] Used to save any information about
        the pre-run state which may be needed for testing post-run
    :function check_teardown: [Optional] Add checks for a successful environment
        cleanup
    :function save_stage_information [Optional] Save state information in between
        stage runs
    """

    command_stubs = {
        "train": "sparseml.foo_integration.train",
        "export": "sparseml.foo_integration.export",
        "deploy": "sparseml.foo_integration.deploy",
    }
    command_args_classes = {
        "train": BaseModel,
        "export": BaseModel,
        "deploy": BaseModel,
    }

    @pytest.fixture(
        scope="class",
        # Iterate over configs with the matching cadence (default commit)
        params=get_configs_with_cadence(os.environ.get("NM_TEST_CADENCE", "commit")),
    )
    def setup(self, request):

        # A path to the current config file
        config_path = request.param
        with open(config_path) as f:
            raw_config = yaml.safe_load(f)

        # Remove cadence for easier processing. It's saved, but not utilized later
        self.cadence = raw_config.pop("cadence")

        # Command types present in this config
        self.command_types = [_type for _type in raw_config]

        # Final command stub
        self.command_stubs_final = self.get_root_commands(raw_config)

        # Map each present command type to the pydantic class and pre-args (see below)
        self.configs = {
            _type: {
                # pre_args are CLI commands that can precede the python call
                # e.g. "torch.distributed.launch" or "CUDA_VISIBLE_DEVICES": 2
                "pre_args": config.pop("pre_args", ""),
                # Instantiate the correspoding pydantic class with the command args
                "args": self.command_args_classes[_type](**config.get("command_args")),
            }
            for _type, config in raw_config.items()
        }

        # test args are used to guide testing of the stage. e.g. target metrics or
        # named quantities/qualities to test for
        self.test_args = {
            _type: config["test_args"] for _type, config in raw_config.items()
        }

        # Combine pre-args, command stubs, and args into complete CLI commands
        self.commands = self.compose_command_scripts(self.configs)

        # Capture any pre-run information that may be needed for post-run testing
        self.capture_pre_run_state()

        # All commands are run sequentially
        self.run_commands(self.commands)

        yield  # all tests are run here

        # Clean up environment after testing is complete
        self.teardown()

        # Check for successful teardown
        self.check_teardown()

        self.check_file_creation()

    @classmethod
    def get_root_commands(cls, configs: Dict[str, Union[str, BaseModel]]):
        """
        Returns the command stubs to use. e.g. "sparseml.yolov5.train".
        If the stub for the integration needs to be determined based on the configs
        (e.g. sparseml.transformers.{task}.train), override this function

        :param configs: unprocessed configs dict
        :return: dict mapping command type to the respective CLI root command
        """
        return cls.command_stubs

    @classmethod
    def compose_command_scripts(cls, configs: Dict[str, BaseModel]):
        """
        For each command, create the full CLI command by combining the pre-args,
        command stub, and run args.

        :param configs: dict mapping command type to a pydantic class holding
            the command args
        :return: dict mapping command type to the full CLI command string
        """
        commands = {}
        for _type, config in configs.items():
            if _type not in cls.command_stubs:
                raise ValueError(f"{_type} is not a valid command type")
            commands[_type] = cls.create_command_script(
                config["pre_args"], cls.command_stubs_final[_type], config["args"]
            )
        return commands

    @classmethod
    def create_command_script(cls, pre_args: str, command_stub: str, config: BaseModel):
        """
        Handles logic for converting pydantic classes into valid argument strings.
        This should set arg standards for all integrations and should generally not
        be overridden. If the need to override comes up, consider updating this method
        instead.

        :param pre_args: string of arguments to prepend to the command stub
        :param command_stub: the root command e.g. sparseml.yolov5.train
        :param config: a pydantic class holding the args for this command
        :return: string of the full CLI command
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

    def run_commands(self, kwargs_dict: Dict[str, Dict] = {}):
        """
        Execute CLI commands in order

        :param kwargs_dict: dict mapping command type to subprocess.call() kwargs
            to be used with the command, if any
        """
        for _type in self.command_types:
            # Optionally, save intermediate state variables between stages
            self.save_stage_information(_type)
            subprocess.call(self.command_types[_type], **kwargs_dict[_type])

    def save_stage_information(self):
        """
        Optional function for saving state information between stages.
        """
        pass

    def teardown(self):
        """
        Cleanup environment after test completion
        NOTE: Generalized directory deletion logic will go here
        """
        pass

    def teardown_check(self):
        """
        Check for successful environment cleanup. Logic to be fleshed out
        """
        self.check_file_creation()

    def check_file_creation(self, dir):
        """
        Check whether files have been created during the run. Logic to be fleshed out
        TODO: Move to universal fixtures file?
        """
        self._end_file_count = sum(len(files) for _, _, files in os.walk(r"."))
        assert self._start_file_count >= self._end_file_count, (
            f"{self._end_file_count - self._start_file_count} files created during "
            "pytest run"
        )
