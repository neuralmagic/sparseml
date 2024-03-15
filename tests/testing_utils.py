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

import logging
import os
import unittest
from typing import Any, Dict, List

import yaml


# TODO: probably makes sense to move this type of function to a more central place,
# which can be used by __init__.py as well
def is_torch_available():
    try:
        import torch

        return True
    # Update
    except ImportError:
        return False


def requires_torch(test_case):
    return unittest.skipUnless(is_torch_available(), "test requires PyTorch")(test_case)


def requires_gpu(test_case):
    return unittest.skipUnless(False, "test requires GPU")(test_case)


def parse_params(configs_directory: str) -> List[Dict[str, Any]]:
    # parses the config file provided
    assert os.path.isdir(
        configs_directory
    ), f"Config_directory {configs_directory} is not a directory"

    config_dicts = []
    for file in os.listdir(configs_directory):
        if file.endswith(".yaml") or file.endswith(".yml"):
            config_path = os.path.join(configs_directory, file)
            # reads the yaml file
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            config_dicts.append(config)
    return config_dicts


def parse_custom(configs_directory: str):
    pass
