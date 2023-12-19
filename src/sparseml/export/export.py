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
from pathlib import Path
from typing import Any, List, Optional, Union

from sparseml.export.export_data import export_data_samples
from sparseml.export.helpers import (
    AVAILABLE_DEPLOYMENT_TARGETS,
    ONNX_MODEL_NAME,
    apply_optimizations,
    create_deployment_folder,
    create_export_kwargs,
)
from sparseml.export.validators import validate_correctness as validate_correctness_
from sparseml.export.validators import validate_structure as validate_structure_
from sparseml.pytorch.opset import TORCH_DEFAULT_ONNX_OPSET
from sparseml.pytorch.utils.helpers import default_device, use_single_gpu
from src.sparseml.integration_helper_functions import (
    IntegrationHelperFunctions,
    resolve_integration,
)


_LOGGER = logging.getLogger(__name__)


def export(
    source_path: Union[Path, str],
    target_path: Union[Path, str],
    onnx_model_name: str = ONNX_MODEL_NAME,
    deployment_target: str = "deepsparse",
    opset: int = TORCH_DEFAULT_ONNX_OPSET,
    single_graph_file: bool = True,
    num_export_samples: int = 0,
    batch_size: int = 1,
    deployment_directory_name: str = "deployment",
    device: str = "auto",
    graph_optimizations: Union[str, List[str], None] = "all",
    validate_correctness: bool = False,
    validate_structure: bool = True,
    integration: Optional[str] = None,
    sample_data: Optional[Any] = None,
    task: Optional[str] = None,
    **kwargs,
):
    """
    Export a PyTorch model located in source_path, to target_path.
    The deployment files will be located at target_path/deployment_directory_name

    The exporting logic consists of the following steps:
    1. Create the model and validation dataloader (if needed) using the
        integration-specific `create_model` function.
    2. Export the model to ONNX using the integration-specific `export` function.
    3. Apply the graph optimizations to the exported model.
    4. Create the deployment folder at target_path/deployment_directory_name
        using the integration-specific `create_deployment_folder` function.
    5. Optionally, export samples using the integration-specific
        `create_data_samples` function.
    6. Optionally, validate the correctness of the exported model using
        the integration-specific `validate_correctness` function.
    7. Optionally, validate the structure of the exported model using
        the integration-specific `validate_structure` function.

    :param source_path: The path to the PyTorch model to export.
    :param target_path: The path to save the exported model to.
    :param onnx_model_name: The name of the exported model.
        Defaults to ONNX_MODEL_NAME.
    :param deployment_target: The deployment target to export
        the model to. Defaults to 'deepsparse'.
    :param opset: The ONNX opset to use for exporting the model.
        Defaults to the latest supported opset.
    :param single_graph_file: Whether to save the model as a single
        file. Defaults to True.
    :param num_export_samples: The number of samples to create for
        the exported model. Defaults to 0.
    :param batch_size: The batch size to use for exporting the data.
        Defaults to None.
    :param deployment_directory_name: The name of the deployment
        directory to create for the exported model. Thus, the exported
        model will be saved to `target_path/deployment_directory_name`.
        Defaults to 'deployment'.
    :param device: The device to use for exporting the model.
        Defaults to 'auto'.
    :param graph_optimizations: The graph optimizations to apply
        to the exported model. Defaults to 'all'.
    :param validate_correctness: Whether to validate the correctness
        of the exported model. Defaults to False.
    :param validate_structure: Whether to validate the structure
        of the exporter model (contents of the target_path).
    :param integration: The name of the integration to use for
        exporting the model.Defaults to None, which will infer
        the integration from the source_path.
    :param sample_data: Optional sample data to use for exporting
        the model. If not provided, a dummy input will be created
        for the model. Defaults to None.
    :param task: Optional task to use for exporting the model.
        Defaults to None.
    """

    # create the target path if it doesn't exist
    if not Path(target_path).exists():
        Path(target_path).mkdir(parents=True, exist_ok=True)

    # choose the appropriate device
    device = default_device() if device == "auto" else device
    device = use_single_gpu(device) if "cuda" in device else device

    # assert the valid deployment target
    if deployment_target not in AVAILABLE_DEPLOYMENT_TARGETS:
        raise ValueError(
            "Argument: deployment_target must be "
            f"one of {AVAILABLE_DEPLOYMENT_TARGETS}. "
            f"Got {deployment_target} instead."
        )

    integration = resolve_integration(source_path, integration)

    _LOGGER.info(f"Starting export for {integration} model...")

    helper_functions: IntegrationHelperFunctions = (
        IntegrationHelperFunctions.load_from_registry(integration)
    )

    _LOGGER.info("Creating model for the export...")

    # loaded_model_kwargs may include any objects
    # that were created along with the model and are needed
    # for the export
    model, loaded_model_kwargs = helper_functions.create_model(
        source_path, device=device, task=task, batch_size=batch_size, **kwargs
    )

    if loaded_model_kwargs:
        _LOGGER.info(
            "Created additional items that will "
            f"be used for the export: {list(loaded_model_kwargs.keys())}"
        )

    sample_data = (
        helper_functions.create_dummy_input(**loaded_model_kwargs, **kwargs)
        if sample_data is None
        else sample_data
    )

    _LOGGER.info(f"Exporting {onnx_model_name} to {target_path}...")

    export_kwargs = create_export_kwargs(loaded_model_kwargs)

    onnx_file_path = helper_functions.export(
        model=model,
        sample_data=sample_data,
        target_path=target_path,
        onnx_model_name=onnx_model_name,
        deployment_target=deployment_target,
        opset=opset,
        **export_kwargs,
    )
    _LOGGER.info(f"Successfully exported {onnx_model_name} to {onnx_file_path}...")

    if num_export_samples:
        _LOGGER.info(f"Exporting {num_export_samples} samples...")
        (
            input_samples,
            output_samples,
            label_samples,
        ) = helper_functions.create_data_samples(
            num_samples=num_export_samples,
            model=model,
            **loaded_model_kwargs,
        )
        export_data_samples(
            input_samples=input_samples,
            output_samples=output_samples,
            label_samples=label_samples,
            target_path=target_path,
            as_tar=False,
        )

    _LOGGER.info(
        f"Creating deployment folder {deployment_directory_name} "
        f"at directory: {target_path}..."
    )

    deployment_path = create_deployment_folder(
        source_path=source_path,
        target_path=target_path,
        deployment_directory_name=deployment_directory_name,
        deployment_directory_files_mandatory=helper_functions.deployment_directory_files_mandatory,  # noqa: E501
        deployment_directory_files_optional=helper_functions.deployment_directory_files_optional,  # noqa: E501
        onnx_model_name=onnx_model_name,
    )

    _LOGGER.info(
        f"Applying optimizations: {graph_optimizations} to the exported model..."
    )
    apply_optimizations(
        onnx_file_path=os.path.join(deployment_path, onnx_model_name),
        target_optimizations=graph_optimizations,
        available_optimizations=helper_functions.graph_optimizations,
        single_graph_file=single_graph_file,
    )

    if validate_structure:
        _LOGGER.info("Validating model structure...")
        validate_structure_(
            target_path=target_path,
            deployment_directory_name=deployment_directory_name,
            onnx_model_name=onnx_model_name,
            deployment_directory_files_mandatory=helper_functions.deployment_directory_files_mandatory,  # noqa: E501
            deployment_directory_files_optional=helper_functions.deployment_directory_files_optional,  # noqa: E501
        )

    if validate_correctness:
        _LOGGER.info("Validating model correctness...")
        if not num_export_samples:
            raise ValueError(
                "To validate correctness sample inputs/outputs are needed."
                "To enable the validation, set `num_export_samples`"
                "to True"
            )
        validate_correctness_(target_path, deployment_path, onnx_model_name)

    _LOGGER.info(
        f"Successfully exported model from:\n{target_path}"
        f"\nto\n{deployment_path}\nfor integration: {integration}"
    )
