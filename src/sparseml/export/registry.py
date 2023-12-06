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

from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Union

from pydantic import BaseModel, Field, validator

from sparseml.pytorch.image_classification.utils.helpers import (
    create_model as create_model_ic,
)
from sparseml.pytorch.image_classification.utils.helpers import (
    is_image_classification_model,
)
from sparsezoo.utils.registry import RegistryMixin


__all__ = ["IntegrationHelperFunctions", "infer_integration"]


class Integrations(Enum):
    """
    Holds the names of the available integrations.
    """

    image_classification = "image-classification"


# TODO: Fold it into the functionalities of the RegistryMixin
# when the `resolve` method is generically implemented
def infer_integration(source_path: Union[Path, str]) -> str:
    """
    Infer the integration to use for exporting the model from the source_path.

    :param source_path: The path to the PyTorch model to export.
    :return: The name of the integration to use for exporting the model.
    """
    if is_image_classification_model(source_path):
        return Integrations.image_classification.value
    else:
        raise ValueError(
            f"Could not infer integration from source_path: {source_path}."
            f"Please specify an argument `integration` from one of"
            f"the available integrations: {list(Integrations)}."
        )


class IntegrationHelperFunctions(RegistryMixin, BaseModel):
    """
    Registry that maps names to helper functions
    for creation/export/manipulation of models for a specific
    integration.
    """

    create_model: Optional[Callable] = Field(
        description="A function that creates a (sparse) "
        "PyTorch model from a source path and additional "
        "arguments"
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
    @validator("create_model", pre=True)
    def create_model_only_one_output(cls, v: Optional[Callable]) -> Optional[Callable]:
        """
        Ensure that the create_model function only outputs
        the first output - the model itself.
        """
        if v is not None:
            v = cls.wrap_to_return_first_output(v)
        return v

    @staticmethod
    def wrap_to_return_first_output(func: Callable) -> Callable:
        return lambda *args, **kwargs: func(*args, **kwargs)[0]


@IntegrationHelperFunctions.register(name=Integrations.image_classification.value)
class ImageClassification(IntegrationHelperFunctions):
    create_model: Any = Field(default=create_model_ic)
