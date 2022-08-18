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

- {Integration_name}_tester.py file containing integration manager and test classes
inherited from BaseIntegrationManager and BaseIntegrationTester, respectively

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

from flaky import flaky
from tests.integrations.config import Config


__all__ = [
    "BaseIntegrationManager",
    "BaseIntegrationTester",
]


class BaseIntegrationManager:
    """
    Base class for testing integrations through train, export, and deploy scripts.

    Each of train, export, and deploy are "command types" and can constitute a stage
    in a multi-stage run.

    Each integration will implement a subclass of this class, with the following
    fields and functions filled out:

    :field command_stubs: Mapping from command type to the respective CLI command.
        Note that the stubs can be modified via the get_root_commands() function
    :field config_classes: Mapping from command type to the pydnatic class
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
    config_classes = {
        "train": BaseModel,
        "export": BaseModel,
        "deploy": BaseModel,
    }

    def __init__(self, config_path):
        with open(config_path) as f:
            raw_config = yaml.safe_load(f)

        # Remove cadence for easier processing. It's saved, but not utilized later
        self.cadence = raw_config.pop("cadence")

        # If abridged, the run parameters are modified to significantly shorten the run
        self.abridged = raw_config.pop("abridged", False)

        # Command types present in this config
        self.command_types = [_type for _type in raw_config]

        # Final command stub
        self.command_stubs_final = self.get_root_commands(raw_config)

        # Compose commands into arg managers
        self.configs = {
            _type: Config(
                self.config_classes[_type],
                config,
                self.command_stubs_final[_type],
            )
            for _type, config in raw_config.items()
        }

        # Capture any pre-run information that may be needed for post-run testing
        self.capture_pre_run_state()

        # Shorten run to a standardized abridged format
        if self.abridged:
            self.add_abridged_configs()

        # Combine pre-args, command stubs, and args into complete CLI commands
        # If command stub is of type None, skip generating command
        self.commands = {
            _type: config.create_command_script()
            for _type, config in self.configs.items()
            if self.command_stubs[_type]
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

    def add_abridged_configs(self):
        """
        Update configs to shorten run. e.g. set small steps_per_epoch for training run
        """
        raise NotImplementedError()

    def run_commands(self, kwargs_dict: Union[Dict[str, Dict], None] = None):
        """
        Execute CLI commands in order

        :param kwargs_dict: dict mapping command type to subprocess.call() kwargs
            to be used with the command, if any
        """

        if not kwargs_dict:
            kwargs_dict = {key: {} for key in self.command_types}
        for _type in self.command_types:
            # Optionally, save intermediate state variables between stages
            self.save_stage_information(_type)
            if self.command_stubs[_type]:  # check if command is executable
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
        """
        self._end_file_count = sum(len(files) for _, _, files in os.walk(r"."))
        assert self._start_file_count >= self._end_file_count, (
            f"{self._end_file_count - self._start_file_count} files created during "
            "pytest run"
        )

    def _check_deploy_requirements(self, deepsparse_error):
        """
        If a deploy stage is present and deepsparse is not installed, throw an error.
        """
        if "deploy" in self.command_types and deepsparse_error:
            raise ImportError(
                "DeepSparse is required for integration tests with a deploy stage."
                f"DeepSparse import error: {deepsparse_error}"
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


@flaky(max_runs=2, min_passes=1)
class BaseIntegrationTester:
    """
    Class from which integration test-holding classes should inherit. Tests defined here
    need to be implemented for each integration on the integration level.

    All tests are expected to follow the name convention `test_{stage}_{name}` where
    stage is `train`, `export`, or `deploy` and name is a unique name to describe the
    test. This naming convention is used and enforced within the decorator
    `@skip_inactive_stage`

    Adding the fixture below to each test class will parameterize the test over
    the set of test configs that match the cadence setting.

    @pytest.fixture(
        params=get_configs_with_cadence(
            os.environ.get("SPARSEML_TEST_CADENCE", "pre-commit"),
            os.path.dirname(__file__),
        ),
        scope="class",
    )
    """

    def integration_manager(request):
        """
        Fixture with a lifecycle of:
        - Create integration manager instance (child of BaseIntegrationManager)
        - Yield manager to test
        - Teardown

        Child implementations of fixture need to include the same fixture decorator
        """
        raise NotImplementedError()

    @skip_inactive_stage
    def test_train_complete(self, integration_manager):
        """
        Tests:
            - Run created a model file
            - Model file is loadable
            - The model epoch corresponds to the expected value
        """
        raise NotImplementedError()

    @skip_inactive_stage
    def test_train_metrics(self, integration_manager):
        """
        Tests:
            - Train metrics are within the expected range
        """
        raise NotImplementedError()

    @skip_inactive_stage
    def test_export_onnx_graph(self, integration_manager):
        """
        Test:
            - Export generated an onnx model which passes the onnx checker
        """
        raise NotImplementedError()

    @skip_inactive_stage
    def test_export_target_model(self, integration_manager):
        """
        If no target model provided in config file, skip test
        Tests:
            - Target model and generated model have equivalent graphs
            - Target model and generated model produce similar outputs when run through
            onnxruntime. Tolerance set via pytest.approx(abs=1e-5)
        """
        raise NotImplementedError()

    @skip_inactive_stage
    def test_deploy_model_compile(self, integration_manager):
        """
        Tests:
            - Exported onnx model can be loaded into a DeepSparse Pipeline
            - Generated Pipeline can process input
        """
        raise NotImplementedError()
