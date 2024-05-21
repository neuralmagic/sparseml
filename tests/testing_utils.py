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

import dataclasses
import enum
import logging
import os
import unittest
from subprocess import PIPE, STDOUT, run
from typing import List, Optional, Union

import yaml

from tests.data import CustomTestConfig, TestConfig


# TODO: probably makes sense to move this type of function to a more central place,
# which can be used by __init__.py as well
def is_torch_available():
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def is_gpu_available():
    """
    Check for GPU and warn if not found
    """
    try:
        import torch  # noqa: F401

        return torch.cuda.device_count() > 0
    except ImportError:
        return False


def requires_torch(test_case):
    return unittest.skipUnless(is_torch_available(), "test requires PyTorch")(test_case)


def requires_gpu(test_case):
    return unittest.skipUnless(is_gpu_available(), "test requires GPU")(test_case)


def _load_yaml(configs_directory, file):
    if file.endswith(".yaml") or file.endswith(".yml"):
        config_path = os.path.join(configs_directory, file)
        # reads the yaml file
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    return None


def _validate_test_config(config: dict):
    for f in dataclasses.fields(TestConfig):
        if f.name not in config:
            return False
        config_value = config.get(f.name)
        if issubclass(f.type, enum.Enum):
            try:
                f.type(config_value)
            except ValueError:
                raise False
    return True


# Set cadence in the config. The environment must set if nightly, weekly or commit
# tests are running
def parse_params(
    configs_directory: str, type: Optional[str] = None
) -> List[Union[dict, CustomTestConfig]]:
    # parses the config file provided
    assert os.path.isdir(
        configs_directory
    ), f"Config_directory {configs_directory} is not a directory"

    config_dicts = []
    for file in os.listdir(configs_directory):
        config = _load_yaml(configs_directory, file)
        if not config:
            continue

        cadence = os.environ.get("CADENCE", "commit")
        expected_cadence = config.get("cadence")

        if not isinstance(expected_cadence, list):
            expected_cadence = [expected_cadence]
        if cadence in expected_cadence:
            if type == "custom":
                config = CustomTestConfig(**config)
            else:
                if not _validate_test_config(config):
                    raise ValueError(
                        "The config provided does not comply with the expected "
                        "structure. See tests.data.TestConfig for the expected "
                        "fields."
                    )
            config_dicts.append(config)
        else:
            logging.info(
                f"Skipping testing model: {file} for cadence: {config['cadence']}"
            )
    return config_dicts


def run_cli_command(cmd: List[str]):
    """
    Run a cli command and return the response. The cli command is launched through a new
    subprocess.

    :param cmd: cli command provided as a list of arguments where each argument
        should be a string
    """
    return run(cmd, stdout=PIPE, stderr=STDOUT, check=False, encoding="utf-8")
