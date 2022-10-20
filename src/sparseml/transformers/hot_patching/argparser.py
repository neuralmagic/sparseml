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
Argparser related hot patching. Hot patches applied in this file:

- Wrap arg parsing to detect SparseZoo model stubs and handle them appropriately
"""

import os
from pathlib import Path
from typing import Any, NewType

import transformers
from transformers import HfArgumentParser
from transformers.utils.logging import get_logger

from sparsezoo import Model


DataClass = NewType("DataClass", Any)

logger = get_logger(__name__)
HANDLED_FUNCTIONS = {}


def arg_parser_wrapper(parse_args_into_dataclasses_func):
    def _download_dataclass_zoo_stub_files(data_class: DataClass):
        for name, val in data_class.__dict__.items():
            if (
                not isinstance(val, str)
                or "recipe" in name
                or not val.startswith("zoo:")
            ):
                continue

            logger.info(f"Downloading framework files for SparseZoo stub: {val}")

            zoo_model = Model(val)
            framework_file_paths = [
                file.path for file in zoo_model.training.default.files
            ]
            assert (
                framework_file_paths
            ), "Unable to download any framework files for SparseZoo stub {val}"
            framework_file_names = [
                os.path.basename(path) for path in framework_file_paths
            ]
            if "pytorch_model.bin" not in framework_file_names or (
                "config.json" not in framework_file_names
            ):
                raise RuntimeError(
                    "Unable to find 'pytorch_model.bin' and 'config.json' in framework "
                    f"files downloaded from {val}. Found {framework_file_names}. Check "
                    "if the given stub is for a transformers repo model"
                )
            framework_dir_path = Path(framework_file_paths[0]).parent.absolute()

            logger.info(
                f"Overwriting argument {name} to downloaded {framework_dir_path}"
            )

            data_class.__dict__[name] = str(framework_dir_path)

        return data_class

    def wrapper(self, *args, **kwargs):
        out = parse_args_into_dataclasses_func(self, *args, **kwargs)
        return tuple(map(_download_dataclass_zoo_stub_files, out))

    return wrapper


transformers.HfArgumentParser.parse_args_into_dataclasses = arg_parser_wrapper(
    HfArgumentParser.parse_args_into_dataclasses
)
