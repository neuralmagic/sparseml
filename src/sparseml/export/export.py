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
Usage: sparseml.export [OPTIONS] SOURCE_PATH

Options:
  --target_path TEXT              Path to write the exported model to,
                                  defaults to a source_path
  --onnx_model_name TEXT          Name of onnx model to write, defaults to
                                  model.onnx
  --deployment_target TEXT        Engine or engine family exported model will
                                  run on, default 'deepsparse'
  --opset INTEGER                 Onnx opset to export to. Defaults to the latest
                                  supported opset.
  --single_graph_file BOOLEAN     Default True - if True, onnx graph will be
                                  written to a single file
  --num_export_samples INTEGER    Number of sample inputs/outputs to save.
                                  Default 0
  --recipe TEXT                   Optional sparsification recipe to apply at
                                  runtime
  --deployment_directory_name TEXT
                                  Name of the folder inside the target_path
                                  to save the exported model to. Default -
                                  `deployment'
  --device TEXT                   Device to run export trace with. Default -
                                  'cpu'
  --graph_optimizations TEXT      csv list of graph optimizations to apply.
                                  Default all, can set to none
  --validate_correctness BOOLEAN  Default False - if True, graph will be
                                  validated for output correctness
  --validate_structure BOOLEAN    Default True - if True, graph structure will
                                  be statically validated
  --integration TEXT              Integration the model was trained under. ie
                                  transformers, image-classification. Will be
                                  inferred by default
  --sample_data TEXT              Path to sample data to export with. default
                                  None
  --task TEXT                     Task within the integration this model
                                  was trained on. Default - None
  --help                          Show this message and exit.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy

import click
from sparseml.export.export_data import export_data_samples
from sparseml.export.helpers import (
    AVAILABLE_DEPLOYMENT_TARGETS,
    ONNX_MODEL_NAME,
    create_deployment_folder,
    create_export_kwargs,
    format_source_path,
)
from sparseml.export.validators import validate_correctness as validate_correctness_
from sparseml.export.validators import validate_structure as validate_structure_
from sparseml.integration_helper_functions import (
    IntegrationHelperFunctions,
    resolve_integration,
)
from sparseml.pytorch.opset import TORCH_DEFAULT_ONNX_OPSET
from sparseml.pytorch.utils.helpers import default_device
from sparsezoo.utils.numpy import load_numpy


_LOGGER = logging.getLogger(__name__)


def export(
    source_path: Union[Path, str],
    target_path: Union[Path, str, None] = None,
    onnx_model_name: str = ONNX_MODEL_NAME,
    deployment_target: str = "deepsparse",
    opset: int = TORCH_DEFAULT_ONNX_OPSET,
    single_graph_file: bool = True,
    num_export_samples: int = 0,
    recipe: Optional[Union[Path, str]] = None,
    deployment_directory_name: str = "deployment",
    device: str = "cpu",
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
    :param target_path: The path to save the exported model to. If not provided
        will default to source_path
    :param onnx_model_name: The name of the exported model.
        Defaults to ONNX_MODEL_NAME.
    :param deployment_target: The deployment target to export
        the model to. Defaults to 'deepsparse'.
    :param opset: The ONNX opset to use for exporting the model.
        Defaults to the latest supported opset.
    :param recipe: The path to the recipe to use for exporting the model.
        Defaults to None. If a recipe is found in the source_path, it will
        be automatically used for export.
    :param single_graph_file: Whether to save the model as a single
        file. Defaults to True.
    :param num_export_samples: The number of samples to create for
        the exported model. Defaults to 0.
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
        exporting the model. Defaults to None, which will infer
        the integration from the source_path.
    :param sample_data: Optional sample data to use for exporting
        the model. If not provided, a dummy input will be created
        for the model. Defaults to None.
    :param task: Optional task to use for exporting the model.
        Defaults to None.
    """
    # TODO: Remove with the following once sparsezoo: #404 lands
    """
    from sparsezoo.utils.registry import standardize_lookup_name
    task = standardize_lookup_name(task)
    """
    source_path = format_source_path(source_path)
    if task is not None:
        task = task.replace("_", "-").replace(" ", "-")

    # TODO: Remove once sparsezoo: #404 lands
    if integration is not None:
        integration = integration.replace("_", "-").replace(" ", "-")

    if target_path is None:
        target_path = source_path
    # create the target path if it doesn't exist
    if not Path(target_path).exists():
        Path(target_path).mkdir(parents=True, exist_ok=True)

    # choose the appropriate device
    device = default_device() if device == "auto" else device

    # assert the valid deployment target
    if deployment_target not in AVAILABLE_DEPLOYMENT_TARGETS:
        raise ValueError(
            "Argument: deployment_target must be "
            f"one of {AVAILABLE_DEPLOYMENT_TARGETS}. "
            f"Got {deployment_target} instead."
        )

    integration = resolve_integration(source_path, integration)

    deployment_folder_dir = os.path.join(target_path, deployment_directory_name)

    if os.path.isdir(deployment_folder_dir):
        _LOGGER.warning(
            f"Deployment directory at: {deployment_folder_dir} already exists."
            "Overwriting the existing deployment directory... "
        )
        shutil.rmtree(deployment_folder_dir)

    _LOGGER.info(f"Starting export for {integration} model...")

    helper_functions: IntegrationHelperFunctions = (
        IntegrationHelperFunctions.load_from_registry(integration, task=task)
    )

    _LOGGER.info("Creating model for the export...")

    # loaded_model_kwargs may include any objects
    # that were created along with the model and are needed
    # for the export
    model, loaded_model_kwargs = helper_functions.create_model(
        source_path,
        device=device,
        task=task,
        recipe=recipe,
        **kwargs,
    )
    model.eval()

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

    deployment_folder_dir = create_deployment_folder(
        source_path=source_path,
        target_path=target_path,
        deployment_directory_name=deployment_directory_name,
        deployment_directory_files_mandatory=helper_functions.deployment_directory_files_mandatory,  # noqa: E501
        deployment_directory_files_optional=helper_functions.deployment_directory_files_optional,  # noqa: E501
        onnx_model_name=onnx_model_name,
    )

    if validate_correctness:
        _LOGGER.info("Validating model correctness...")
        if not num_export_samples:
            raise ValueError(
                "To validate correctness sample inputs/outputs are needed."
                "To enable the validation, set `num_export_samples`"
                "to True"
            )
        validate_correctness_(target_path, deployment_folder_dir, onnx_model_name)

    _LOGGER.info(
        f"Applying optimizations: {graph_optimizations} to the exported model..."
    )

    if helper_functions.apply_optimizations is not None:
        helper_functions.apply_optimizations(
            exported_file_path=os.path.join(deployment_folder_dir, onnx_model_name),
            optimizations=graph_optimizations,
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

    _LOGGER.info(
        f"Successfully exported model from:\n{target_path}"
        f"\nto\n{deployment_folder_dir}\nfor integration: {integration}"
    )


@click.command()
@click.argument("source_path", type=str)
@click.option(
    "--target_path",
    type=str,
    default=None,
    help=(
        "Path to write the exported model to, defaults to a `deployment` "
        "directory in the source_path"
    ),
)
@click.option(
    "--onnx_model_name",
    type=str,
    default=ONNX_MODEL_NAME,
    help=f"Name of onnx model to write, defaults to {ONNX_MODEL_NAME}",
)
@click.option(
    "--deployment_target",
    type=str,
    default="deepsparse",
    help="Engine or engine family exported model will run on, default 'deepsparse'",
)
@click.option(
    "--opset",
    type=int,
    default=TORCH_DEFAULT_ONNX_OPSET,
    help=f"Onnx opset to export to, default: {TORCH_DEFAULT_ONNX_OPSET}",
)
@click.option(
    "--single_graph_file",
    type=bool,
    default=True,
    help="Default True - if True, onnx graph will be written to a single file",
)
@click.option(
    "--num_export_samples",
    type=int,
    default=0,
    help="Number of sample inputs/outputs to save. Default 0",
)
@click.option(
    "--recipe",
    type=str,
    default=None,
    help="Optional sparsification recipe to apply at runtime",
)
@click.option(
    "--deployment_directory_name",
    type=str,
    default="deployment",
    help="Name to save exported files under. Default - 'deployment'",
)
@click.option(
    "--device",
    type=str,
    default="cpu",
    help="Device to run export trace with. Default - 'cpu'",
)
@click.option(
    "--graph_optimizations",
    type=str,
    default="all",
    help="csv list of graph optimizations to apply. "
    "Available options: 'all', 'none' or referring to optimization by name. ",
)
@click.option(
    "--validate_correctness",
    type=bool,
    default=False,
    help="Default False - if True, graph will be validated for output correctness",
)
@click.option(
    "--validate_structure",
    type=bool,
    default=True,
    help="Default True - if True, graph structure will be statically validated",
)
@click.option(
    "--integration",
    type=click.Choice(["image-classification, transformers"]),
    default=None,
    help="Integration the model was trained under. By default, inferred from the model",
)
@click.option(
    "--sample_data",
    type=str,
    default=None,
    help="Path to sample data to export with. Default - None",
)
@click.option(
    "--task",
    type=str,
    default=None,
    help="Task within the integration this model was trained on. Default - None",
)
def main(
    source_path: str,
    target_path: str,
    onnx_model_name: str = ONNX_MODEL_NAME,
    deployment_target: str = "deepsparse",
    opset: int = TORCH_DEFAULT_ONNX_OPSET,
    single_graph_file: bool = True,
    num_export_samples: int = 0,
    recipe: str = None,
    deployment_directory_name: str = "deployment",
    device: str = "cpu",
    graph_optimizations: str = "all",
    validate_correctness: bool = False,
    validate_structure: bool = True,
    integration: str = None,
    sample_data: str = None,
    task: str = None,
):
    export(
        source_path=source_path,
        target_path=target_path,
        onnx_model_name=onnx_model_name,
        deployment_target=deployment_target,
        opset=opset,
        single_graph_file=single_graph_file,
        num_export_samples=num_export_samples,
        recipe=recipe,
        deployment_directory_name=deployment_directory_name,
        device=device,
        graph_optimizations=_parse_graph_optimizations(graph_optimizations),
        validate_correctness=validate_correctness,
        validate_structure=validate_structure,
        integration=integration,
        sample_data=_parse_sample_data(sample_data),
        task=task,
    )


def _parse_graph_optimizations(graph_optimizations):
    if "," in graph_optimizations:
        return graph_optimizations.split(",")
    elif graph_optimizations.lower() in ["none", "null", "", "false", "0"]:
        return None
    return graph_optimizations


def _parse_sample_data(
    sample_data: Union[None, Path, str]
) -> Union[None, numpy.ndarray]:
    if sample_data is None:
        return None
    elif sample_data.endswith((".npz", ".npy")):
        return load_numpy(sample_data)
    else:
        raise NotImplementedError(
            "Only numpy files (.npy) are supported for sample_data"
        )
