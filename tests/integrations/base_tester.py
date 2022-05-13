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

- {Integration_name}_args.py file containing pydantic classes for the args of each of
the commands (i.e. train, export, deploy).

- {Integration_name}_tester.py file containing integration testing class inherited
from BaseIntegrationTester

- Tests to run after commands are run. Tests should be implemented in the tester class
described in B and should be decorated by @skip_inactive_stage found in helpers.py

- .yaml file for each scenario to run
"""

import os
import subprocess
from functools import wraps
from typing import Dict, Union

import pytest
import yaml
from pydantic import BaseModel

from tests.integrations.helpers import Config


class BaseIntegrationManager:
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

    def __init__(self, config_path):
        with open(config_path) as f:
            raw_config = yaml.safe_load(f)

        # Remove cadence for easier processing. It's saved, but not utilized later
        self.cadence = raw_config.pop("cadence")

        # Command types present in this config
        self.command_types = [_type for _type in raw_config]

        # Final command stub
        self.command_stubs_final = self.get_root_commands(raw_config)

        # Compose commands into arg managers
        self.configs = {
            _type: Config(
                self.command_args_classes[_type],
                config,
                self.command_stubs_final[_type],
            )
            for _type, config in raw_config.items()
        }

        # Capture any pre-run information that may be needed for post-run testing
        self.capture_pre_run_state()

        # Combine pre-args, command stubs, and args into complete CLI commands
        self.commands = {
            _type: config.create_command_script()
            for _type, config in self.configs.items()
        }

        # All commands are run sequentially
        self.run_commands()

    def get_root_commands(self, configs: Dict[str, Union[str, BaseModel]]):
        """
        Returns the command stubs to use. e.g. "sparseml.yolov5.train".
        If the stub for the integration needs to be determined based on the configs
        (e.g. sparseml.transformers.{task}.train), override this function

        :param configs: unprocessed configs dict
        :return: dict mapping command type to the respective CLI root command
        """
        return self.command_stubs

    def capture_pre_run_state(self):
        """
        Store pre-run information which will be relevant for post-run testing
        """
        self._start_file_count = sum(len(files) for _, _, files in os.walk(r"."))

    def run_commands(self, kwargs_dict: Union[Dict[str, Dict], None] = None):
        """
        Execute CLI commands in order

        :param kwargs_dict: dict mapping command type to subprocess.call() kwargs
            to be used with the command, if any
        """
        # Debug only code
        self.commands["train"] = [
            "/home/konstantin/Source/sparseml/dev-venv/bin/python3.8",
            "/home/konstantin/Source/sparseml/src/sparseml/pytorch/object_detection/train.py",
        ] + self.commands["train"][1:]
        self.commands["export"] = [
            "/home/konstantin/Source/sparseml/dev-venv/bin/python3.8",
            "/home/konstantin/Source/sparseml/src/sparseml/pytorch/object_detection/export.py",
        ] + self.commands["export"][1:]

        if not kwargs_dict:
            kwargs_dict = {key: {} for key in self.command_types}
        for _type in self.command_types:
            # Optionally, save intermediate state variables between stages
            self.save_stage_information(_type)
            try:
                subprocess.check_output(self.commands[_type], **kwargs_dict[_type])
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    "command '{}' return with error (code {}): {}".format(
                        e.cmd, e.returncode, e.output
                    )
                )

    def save_stage_information(self, command_type):
        """
        Optional function for saving state information between stages.
        """
        pass

    def teardown(self):
        """
        Cleanup environment after test completion
        """
        raise NotImplementedError()

    def teardown_check(self):
        """
        Check for successful environment cleanup.
        """
        pass

    def check_file_creation(self, dir):
        """
        Check whether files have been created during the run.
        TODO: Move to universal fixtures file?
        """
        self._end_file_count = sum(len(files) for _, _, files in os.walk(r"."))
        assert self._start_file_count >= self._end_file_count, (
            f"{self._end_file_count - self._start_file_count} files created during "
            "pytest run"
        )


def skip_inactive_stage(test):
    """
    Check whether the this test's command type is active in this run. If not,
    skip test.

    :param test: test function which follows the name convention test_{command_type}_...
    """

    @wraps(test)
    def wrapped_test(self, *args, **kwargs):
        manager = [arg for arg in args if isinstance(arg, BaseIntegrationManager)] or [
            val
            for kwarg, val in kwargs.items()
            if isinstance(val, BaseIntegrationManager)
        ]
        if len(manager) != 1:
            raise KeyError(
                f"Expected function {test.__name__} to ingest a manager object of type "
                f"BaseIntegrationManager. Found {len(manager)} matching args."
            )
        manager = manager[0]
        command_type = test.__name__.split("_")[1]
        if command_type not in manager.command_stubs:
            raise ValueError(
                "Invalid test function definition. Test names must take the form "
                f"test_{{CommandType}}_... Found instead {command_type} for "
                "{Command_type}"
            )
        if command_type not in manager.command_types:
            pytest.skip(f"No {command_type} stage active. Skipping test")
        test(self, *args, **kwargs)

    return wrapped_test


class BaseIntegrationTester:
    @pytest.fixture(scope="class")
    def integration_manager(request):
        raise NotImplementedError()

    @skip_inactive_stage
    def test_train_complete(self, integration_manager):
        raise NotImplementedError()

    @skip_inactive_stage
    def test_train_metrics(self, integration_manager):
        raise NotImplementedError()

    @skip_inactive_stage
    def test_export_onnx_graph(self, integration_manager):
        raise NotImplementedError()

    @skip_inactive_stage
    def test_export_target_model(self, integration_manager):
        raise NotImplementedError()
