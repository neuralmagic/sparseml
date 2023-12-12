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
from typing import Any, List, Optional, Union

from sparseml.export.helpers import (
    AVAILABLE_DEPLOYMENT_TARGETS,
    ONNX_MODEL_NAME,
    apply_optimizations,
    create_deployment_folder,
)
from sparseml.export.validate_correctness import validate_correctness
from sparseml.export_data import export_data_samples
from sparseml.integration_helper_functions import (
    IntegrationHelperFunctions,
    infer_integration,
)
from sparseml.pytorch.opset import TORCH_DEFAULT_ONNX_OPSET


_LOGGER = logging.getLogger(__name__)


def export(
    source_path: Union[Path, str],
    target_path: Union[Path, str],
    model_onnx_name: str = ONNX_MODEL_NAME,
    deployment_target: str = "deepsparse",
    integration: Optional[str] = None,
    sample_data: Optional[Any] = None,
    opset: int = TORCH_DEFAULT_ONNX_OPSET,
    batch_size: Optional[int] = None,
    single_graph_file: bool = True,
    graph_optimizations: Union[str, List[str], None] = "all",
    validate_model_correctness: bool = False,
    num_export_samples: int = 0,
    deployment_directory_name: str = "deployment",
    device: str = "auto",
):
    """
    Export a PyTorch model to a deployment target specified by the `deployment_target`.

    The functionality follows a set of steps:
    1. Create a PyTorch model from the file located in source_path.
    2. Create model dummy input.
    3. Export the model to the format specified by the `deployment_target`.
    4. (Optional) Apply optimizations to the exported model.
    5. Export sample inputs and outputs for the exported model (optional).
    6. Create a deployment folder for the exported model with the appropriate structure.
    7. Validate the correctness of the exported model (optional).

    :param source_path: The path to the PyTorch model to export.
    :param target_path: The path to save the exported model to.
    :param model_onnx_name: The name of the exported model.
        Defaults to ONNX_MODEL_NAME.
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
    :param validate_model_correctness: Whether to validate the correctness
        of the exported model. Defaults to False.
    :param num_export_samples: The number of samples to export for
        the exported model. Defaults to 0.
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
        IntegrationHelperFunctions.load_from_registry(integration)
    )

    # for now, this code is not runnable, serves as a blueprint
    model, auxiliary_items = helper_functions.create_model(
        source_path, **kwargs  # noqa: F821
    )
    sample_data = (
        helper_functions.create_dummy_input(**auxiliary_items)
        if sample_data is None
        else sample_data
    )
    onnx_file_path = helper_functions.export_model(
        model, sample_data, target_path, deployment_target, opset
    )

    apply_optimizations(
        onnx_file_path=onnx_file_path,
        graph_optimizations=graph_optimizations,
        available_graph_optimizations=helper_functions.graph_optimizations,
        single_graph_file=single_graph_file,
    )

    if num_export_samples:
        data_loader = auxiliary_items.get("validation_loader")
        if data_loader is None:
            raise ValueError(
                "To export sample inputs/outputs a data loader is needed."
                "To enable the export, provide a `validation_loader` "
                "as a part of `auxiliary_items` output of the `create_model` function."
            )
        (
            input_samples,
            output_samples,
            label_samples,
        ) = helper_functions.create_data_samples(
            num_samples=num_export_samples, data_loader=data_loader, model=model
        )
        export_data_samples(
            input_samples=input_samples,
            output_samples=output_samples,
            label_samples=label_samples,
            target_path=target_path,
            as_tar=True,
        )

    deployment_path = create_deployment_folder(
        source_path=source_path,
        target_path=target_path,
        deployment_directory_name=deployment_directory_name,
        deployment_directory_files=helper_functions.deployment_directory_structure,
        onnx_model_name=model_onnx_name,
    )

    if validate_model_correctness:
        if not num_export_samples:
            raise ValueError(
                "To validate correctness sample inputs/outputs are needed."
                "To enable the validation, set `num_export_samples`"
                "to True"
            )
        validate_correctness(deployment_path, model_onnx_name)

    _LOGGER.info(
        f"Successfully exported model from:\n{target_path}"
        f"\nto\n{deployment_path}\nfor integration: {integration}"
    )
