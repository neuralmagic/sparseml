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

from typing import Any, Callable, Optional

from pydantic import BaseModel, Field, validator

from sparseml.pytorch.image_classification.utils.helpers import (
    create_model as create_model_ic,
)
from sparsezoo.utils.registry import RegistryMixin


class IntegrationHelperFunctions(RegistryMixin, BaseModel):
    """
    Registry that maps integration names to helper functions
    for creation/export/manipulation of models for a specific
    integration.
    """

    create_model: Optional[Callable] = Field(
        description="A function that creates a (sparse) "
        "PyTorch model from a source path."
    )
    create_dummy_input: Optional[Callable] = Field(
        description="A function that creates a dummy input "
        "given a (sparse) PyTorch model."
    )
    export_model: Optional[Callable] = Field(
        description="A function that exports a (sparse) PyTorch "
        "model to an ONNX format appropriate for a "
        "deployment target."
    )
    apply_optimizations: Optional[Callable] = Field(
        description="A function that takes a set of "
        "optimizations and applies them to an ONNX model."
    )
    export_sample_inputs_outputs: Optional[Callable] = Field(
        description="A function that exports input/output samples given "
        "a (sparse) PyTorch model."
    )
    create_deployment_folder: Optional[Callable] = Field(
        description="A function that creates a "
        "deployment folder for the exporter ONNX model"
        "with the appropriate structure."
    )

    # use validator to ensure that "create_model" outputs only the first output
    @validator()
    def create_model_only_one_output(cls, v):
        if v is not None:
            v = cls.wrap_to_return_first_output(v)
        return v

    @staticmethod
    def wrap_to_return_first_output(func):
        return lambda *args, **kwargs: func(*args, **kwargs)[0]


@IntegrationHelperFunctions.register()
class ImageClassification(IntegrationHelperFunctions):
    create_model: Any = Field(default=create_model_ic)
