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
from typing import Any, Callable, Dict, Optional, Tuple, Union

from pydantic import BaseModel, Field

from sparseml.export.export_torch_model import export_model
from sparsezoo.utils.registry import RegistryMixin


__all__ = ["IntegrationHelperFunctions", "infer_integration"]


class Integrations(Enum):
    """
    Holds the names of the available integrations.
    """

    image_classification = "image-classification"


class IntegrationHelperFunctions(RegistryMixin, BaseModel):
    """
    Registry that maps names to helper functions
    for creation/export/manipulation of models for a specific
    integration.
    """

    create_model: Optional[
        Callable[
            Tuple[Union[str, Path], Optional[Dict[str, Any]]],
            Tuple["torch.nn.Module", Dict[str, Any]],  # noqa F821
        ]
    ] = Field(
        description="A function that takes: "
        "- a source path to a PyTorch model "
        "- (optionally) a dictionary of additional arguments"
        "and returns: "
        "- a (sparse) PyTorch model "
        "- (optionally) a dictionary of additional arguments"
    )
    create_dummy_input: Optional[
        Callable[..., "torch.Tensor"]  # noqa F821
    ] = Field(  # noqa: F82
        description="A function that takes: "
        "- a dictionary of arguments"
        "and returns: "
        "- a dummy input for the model (a torch.Tensor) "
    )
    export: Callable[..., str] = Field(
        description="A function that takes: "
        " - a (sparse) PyTorch model "
        " - sample input data "
        " - the path to save the exported model to "
        " - the name to save the exported ONNX model as "
        " - the deployment target to export to "
        " - the opset to use for the export "
        " - (optionally) a dictionary of additional arguments"
        "and returns nothing",
        default=export_model,
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


def infer_integration(source_path: Union[Path, str]) -> str:
    """
    Infer the integration to use for exporting the model from the source_path.

    :param source_path: The path to the PyTorch model to export.
    :return: The name of the integration to use for exporting the model.
    """
    from sparseml.pytorch.image_classification.utils.helpers import (
        is_image_classification_model,
    )

    if is_image_classification_model(source_path):
        # import to register the image_classification integration helper functions
        import sparseml.pytorch.image_classification.integration_helper_functions  # noqa F401

        return Integrations.image_classification.value
    else:
        raise ValueError(
            f"Could not infer integration from source_path: {source_path}."
            f"Please specify an argument `integration` from one of"
            f"the available integrations: {list(Integrations)}."
        )
