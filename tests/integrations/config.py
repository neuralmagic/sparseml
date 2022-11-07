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
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel


# path to sparseml root directory
ROOT = Path(Path(__file__).resolve().parents[2])

__all__ = ["Config"]


class Config:
    """
    Class containing all the configurations for a single stage of an integration

    :param args_class: pydantic class for the CLI args for this stage
    :param config: the raw config dictionary read in with yaml
    :param command_stub: the base CLI command used to run the stage. A value of None
        indicates that this stage doesn't have a subprocess script to be run
    """

    def __init__(
        self, args_class: BaseModel, config: Dict, command_stub: Optional[str]
    ):
        self.raw_config = config
        # pre_args are CLI commands that can precede the python call
        # e.g. "torch.distributed.launch" or "CUDA_VISIBLE_DEVICES": 2
        self.pre_args = config.pop("pre_args", "")
        self._args_class = args_class
        self.run_args = (
            args_class(**config.get("command_args"))
            if config.get("command_args")
            else args_class()
        )
        self.command_stub = command_stub
        # test args are used to guide testing of the stage. e.g. target metrics or
        # named quantities/qualities to test for
        self.test_args = config.pop("test_args", {})
        # whether to replace '_' with '-' for run keywords
        self.dashed_keywords = False

        # Construct explicit path from relative path, using sparseml base directory
        # as root
        recipe = getattr(self.run_args, "recipe", None)
        if recipe and not recipe.startswith("zoo:"):
            self.run_args.recipe = str(
                Path(os.path.join(ROOT, self.run_args.recipe)).resolve()
            )

        self._validate_config()

    def create_command_script(self):
        """
        Handles logic for converting pydantic classes into valid argument strings.
        This should set arg standards for all integrations and should generally not
        be overridden. If the need to override comes up, consider updating this method
        instead.

        :return: string of the full CLI command
        """
        args_dict = self.run_args.dict()
        args_string_list = []
        for key, value in args_dict.items():
            key = "--" + key
            key = key.replace("_", "-") if self.dashed_keywords else key
            # Handles bool type args (e.g. --do-train)
            if isinstance(value, bool):
                if value:
                    args_string_list.append(key)
            elif isinstance(value, List):
                if len(value) < 2:
                    raise ValueError(
                        "List arguments must have more one entry. "
                        f"Received {key}:{value}"
                    )
                # Handles args that are both bool and value based (see evolve in yolov5)
                if isinstance(value[0], bool):
                    if value[0]:
                        args_string_list.extend([key, str(value[1])])
                # Handles args that have multiple values after the keyword.
                # e.g. --freeze-layers 0 10 15
                else:
                    args_string_list.append(key)
                    args_string_list.extend(map(str, value))
            # Handles the most straightforward case of keyword followed by value
            # e.g. --epochs 30
            else:
                if value is None:
                    continue
                args_string_list.extend([key, str(value)])
        pre_args = self.pre_args.split(" ") if self.pre_args else []
        return pre_args + [self.command_stub] + args_string_list

    def _validate_config(self):
        # Check that all provided run args correspond to expected run args. Mainly
        # meant to catch typos
        if self.raw_config.get("command_args") is None:
            return
        unknown_run_args = [
            arg
            for arg in self.raw_config["command_args"].keys()
            if arg not in self.run_args.__fields_set__
        ]
        if len(unknown_run_args) > 0:
            raise ValueError(
                f"Found unexpected run args {unknown_run_args} for "
                f"{type(self.run_args).__name__}"
            )
