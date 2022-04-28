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

from pathlib import Path
from typing import List, Literal, Tuple, Union

from pydantic import BaseModel


"""
These are dummy classes for demonstration purposes and should not be used or
inherited from
"""


class DummyTrainArgs(BaseModel):
    model_path: Union[str, Path]
    data: Union[str, Path]
    do_train: bool = True
    export_path: Union[str, Path] = "integration_output"

    def __init__(self):
        raise NotImplementedError(
            "This is an abstract class for demonstration "
            "purposes and should not be instantiatied"
        )


class DummyExportArgs(BaseModel):
    model_path: Union[str, Path]
    dynamic: bool = True
    export_path: Union[str, Path] = "integration_output/onnx"

    def __init__(self):
        raise NotImplementedError(
            "This is an abstract class for demonstration "
            "purposes and should not be instantiatied"
        )


class DummyDeployArgs(BaseModel):
    model_path: Union[str, Path]
    data: Union[str, Path]

    def __init__(self):
        raise NotImplementedError(
            "This is an abstract class for demonstration "
            "purposes and should not be instantiatied"
        )
