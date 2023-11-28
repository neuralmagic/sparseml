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
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

from pydantic import BaseModel, Field

from sparseml.pytorch.opset import TORCH_DEFAULT_ONNX_OPSET
from sparsezoo.utils.registry import RegistryMixin


_LOGGER = logging.getLogger(__name__)
AVAILABLE_DEPLOYMENT_TARGETS = ["deepsparse", "onnxruntime"]


class SparseMLIntegrationRegistry(RegistryMixin):
    """
    Registry that maps integration names to helper functions
    for creation/export/manipulation of models for a specific
    integration.
    """

    pass


class IntegrationHelperFunctions(BaseModel):
    """
    A model that contains a set of helper
    functions for a specific integration.
    """

    create_model: Callable = Field(
        description="A function that creates a (sparse) "
        "PyTorch model from a source path."
    )
    create_dummy_input: Callable = Field(
        description="A function that creates a dummy " "input for a model."
    )
    export_model: Callable = Field(
        description="A function that exports a model " "to a deployment target."
    )
    apply_optimizations: Callable = Field(
        description="A function that applies " "optimizations to a model."
    )
    export_sample_inputs_outputs: Callable = Field(
        description="A function that exports sample " "inputs and outputs for a model."
    )
    create_deployment_folder: Callable = Field(
        description="A function that creates a "
        "deployment folder for a model"
        "with the appropriate structure."
    )


def export(
    source_path: Union[Path, str],
    target_path: Union[Path, str],
    deployment_target: str = "deepsparse",
    integration: Optional[str] = None,
    sample_data: Optional[Any] = None,
    opset: int = TORCH_DEFAULT_ONNX_OPSET,
    batch_size: Optional[int] = None,
    single_graph_file: bool = True,
    graph_optimizations: Union[str, List[str], None] = "all",
    validate_correctness: bool = False,
    export_sample_inputs_outputs: bool = False,
    deployment_directory_name: str = "deployment",
    device: str = "auto",
):
    """
    Export a PyTorch model to a deployment target specified by the `deployment_target`.

    :param source_path: The path to the PyTorch model to export.
    :param target_path: The path to save the exported model to.
    :param deployment_target: The deployment target to export
        the model to. Defaults to 'deepsparse'.
    :param integration: The name of the integration to use for
        exporting the model.Defaults to None, which will infer
        the integration from the source_path.
    :param sample_data: Optional sample data to use for exporting
        the model. If not provided, a dummy input will be created
        for the model. Defaults to None.
    :param opset: The ONNX opset to use for exporting the model.
        Defaults to the latest supported opset.
    :param batch_size: The batch size to use for exporting the model.
        Defaults to None.
    :param single_graph_file: Whether to save the model as a single
        file (that contains both the model graph and model weights).
        Defaults to True.
    :param graph_optimizations: The graph optimizations to apply
        to the exported model. Defaults to 'all'.
    :param validate_correctness: Whether to validate the correctness
        of the exported model. Defaults to False.
    :param export_sample_inputs_outputs: Whether to export sample
        inputs and outputs for the exported model.Defaults to False.
    :param deployment_directory_name: The name of the deployment
        directory to create for the exported model. Thus, the exported
        model will be saved to `target_path/deployment_directory_name`.
        Defaults to 'deployment'.
    :param device: The device to use for exporting the model.
        Defaults to 'auto'.
    """

    if deployment_target not in AVAILABLE_DEPLOYMENT_TARGETS:
        raise ValueError(
            "Argument: deployment_target must be "
            f"one of {AVAILABLE_DEPLOYMENT_TARGETS}. "
            f"Got {deployment_target} instead."
        )

    integration = integration or infer_integration(source_path)
    helper_functions: IntegrationHelperFunctions = (
        SparseMLIntegrationRegistry.load_from_registry(integration)
    )

    model = helper_functions.create_model(source_path, device)
    sample_data = (
        helper_functions.create_dummy_input(model, batch_size)
        if sample_data is None
        else sample_data
    )
    onnx_file_path = helper_functions.export_model(
        model, sample_data, target_path, deployment_target, opset, single_graph_file
    )

    helper_functions.apply_optimizations(onnx_file_path, graph_optimizations)

    if export_sample_inputs_outputs:
        helper_functions.export_sample_inputs_outputs(model, target_path)

    deployment_path = helper_functions.create_deployment_folder(
        source_path, target_path, deployment_directory_name
    )

    if validate_correctness:
        if not export_sample_inputs_outputs:
            raise ValueError(
                "To validate correctness sample inputs/outputs are needed."
                "To enable the validation, set `export_sample_inputs_outputs`"
                "to True"
            )
        validate_correctness(deployment_path)

    _LOGGER.info(
        f"Successfully exported model from:\n{target_path}"
        f"\nto\n{deployment_path}\nfor integration: {integration}"
    )


def infer_integration(source_path: Union[Path, str]) -> str:
    """
    Infer the integration to use for exporting the model from the source_path.
    For example:
        - for transformers model the integration
            can be inferred from `config.json`
        - for computer vision, the integration
            can be inferred from the model architecture (`arch_key`)
    :param source_path: The path to the PyTorch model to export.
    :return: The name of the integration to use for exporting the model.
    """
    raise NotImplementedError


def validate_correctness(deployment_path: Union[Path, str]):
    """
    Validate the correctness of the exported model.
    The validation is performed by running the exported model
    on the sample inputs and comparing the outputs to the expected outputs.

    :param deployment_path: The path to the exported model.
    """
    raise NotImplementedError
