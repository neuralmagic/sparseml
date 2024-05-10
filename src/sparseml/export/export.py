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
  --save_with_external_data BOOLEAN
                                  Default False - if True, large constant tensors,
                                  such as initializers, will be serialised
                                  in a separate file. Note: if the model is
                                  sufficiently large, it will be saved with
                                  external data regardless of this flag
  --external_data_chunk_size_mb INTEGER
                                  Size of external data chunks to use for
                                  exporting the model. Defaults to None, which
                                  will use the default chunk size. If set, will
                                  force the export with external data
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
from sparseml.core.session import reset_session
from sparseml.export.helpers import (
    AVAILABLE_DEPLOYMENT_TARGETS,
    ONNX_MODEL_NAME,
    create_deployment_folder,
    create_export_kwargs,
    process_source_path,
    save_model_with_external_data,
)
from sparseml.utils.helpers import parse_kwarg_tuples
from sparsezoo.utils.numpy import load_numpy


_LOGGER = logging.getLogger(__name__)


def export(
    source_path: Union[Path, str] = None,
    target_path: Union[Path, str, None] = None,
    model: Optional["torch.nn.Module"] = None,  # noqa F401
    tokenizer: Optional["PreTrainedTokenizer"] = None,  # noqa F401
    onnx_model_name: str = ONNX_MODEL_NAME,
    deployment_target: str = "deepsparse",
    opset: Optional[int] = None,
    save_with_external_data: bool = False,
    external_data_chunk_size_mb: Optional[int] = None,
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
    Export a PyTorch model that is either:
     - located in source_path (and will be loaded)
     - passed directly to the function
    to target_path.
    The deployment files will be located at target_path/deployment_directory_name

    The exporting logic consists of the following steps:
    1. Create the model (if required) and the data loader using the
       integration-specific `create_model` and `create_data_loader` functions.
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

    :param source_path: The path to the PyTorch model to export. Will be
        omitted if model is provided
    :param target_path: The path to save the exported model to. If not provided
        will default to source_path
    :param model: The PyTorch model to export. If provided, the source_path
        should be set to None to avoid potential confusion and entaglement
        of sources
    :param tokenizer: An optional tokenizer to export if passing in a source through
    the model argument. This argument takes no effect if a source_path is provided
    :param onnx_model_name: The name of the exported model.
        Defaults to ONNX_MODEL_NAME.
    :param deployment_target: The deployment target to export
        the model to. Defaults to 'deepsparse'.
    :param opset: The ONNX opset to use for exporting the model.
        Defaults to the latest supported opset.
    :param recipe: The path to the recipe to use for exporting the model.
        Defaults to None. If a recipe is found in the source_path, it will
        be automatically used for export.
    :param save_with_external_data: if True, large constant tensors,
        such as initializers, will be serialised in a separate file.
        Defaults to False. Note: if the model is sufficiently large,
        it will be saved with external data regardless of this flag.
    :param external_data_chunk_size_mb: The size of the external data
        chunks to use for exporting the model. Defaults to None, which
        will use the default chunk size. If set, will force the
        export with external data.
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
    from sparseml.export.export_data import export_data_samples
    from sparseml.export.validators import validate_correctness as validate_correctness_
    from sparseml.export.validators import validate_structure as validate_structure_
    from sparseml.integration_helper_functions import (
        IntegrationHelperFunctions,
        remove_past_key_value_support_from_config,
        resolve_integration,
    )
    from sparseml.pytorch.opset import TORCH_DEFAULT_ONNX_OPSET
    from sparseml.pytorch.utils.helpers import default_device

    opset = opset or TORCH_DEFAULT_ONNX_OPSET

    # start a new SparseSession for potential recipe application
    reset_session()

    if source_path is not None and model is not None:
        raise ValueError(
            "Not allowed to specify multiple model "
            "sources for export: source_path and model. "
            "Specify either source_path or model, not both"
        )

    if source_path is not None:
        source_path = process_source_path(source_path)
        if target_path is None:
            target_path = source_path
        if tokenizer is not None:
            _LOGGER.warning(
                "Passed a tokenizer is not supported when exporting from ",
                "a source path. The tokenizer will be ignored. ",
            )

    if model is not None and hasattr(model, "config"):
        model.config = remove_past_key_value_support_from_config(model.config)

    integration = resolve_integration(
        source_path=source_path, source_model=model, integration=integration
    )
    _LOGGER.info(f"Starting export for {integration} model...")

    if target_path is None:
        raise ValueError("targe_path is None. Provide the target_path argument.")

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

    deployment_folder_dir = os.path.join(target_path, deployment_directory_name)

    if os.path.isdir(deployment_folder_dir):
        _LOGGER.warning(
            f"Deployment directory at: {deployment_folder_dir} already exists."
            "Overwriting the existing deployment directory... "
        )
        shutil.rmtree(deployment_folder_dir)

    helper_functions: IntegrationHelperFunctions = (
        IntegrationHelperFunctions.load_from_registry(integration, task=task)
    )
    loaded_model_kwargs = {}
    if model is None:
        _LOGGER.info("Creating model for the export...")
        model, loaded_model_kwargs = helper_functions.create_model(
            source_path,
            device=device,
            task=task,
            recipe=recipe,
            **kwargs,
        )
    model.eval()

    # merge arg dictionaries
    for arg_name, arg_val in kwargs.items():
        if arg_name not in loaded_model_kwargs:
            loaded_model_kwargs[arg_name] = arg_val

    # once model is loaded we can clear the SparseSession, it was only needed for
    # adding structural changes (ie quantization) to the model
    reset_session()

    _LOGGER.info("Creating data loader for the export...")
    if tokenizer is not None:
        loaded_model_kwargs["tokenizer"] = tokenizer
    data_loader, loaded_data_loader_kwargs = helper_functions.create_data_loader(
        model=model,
        task=task,
        device=device,
        **loaded_model_kwargs,
    )
    # join kwargs that are created during the initialization of the model
    # and data_loader
    export_kwargs = {**loaded_model_kwargs, **loaded_data_loader_kwargs}

    if export_kwargs:
        _LOGGER.info(
            "Created additional items that will "
            f"be used for the export: {list(export_kwargs.keys())}"
        )

    sample_data = (
        helper_functions.create_dummy_input(data_loader=data_loader, **kwargs)
        if sample_data is None
        else sample_data
    )

    _LOGGER.info(f"Exporting {onnx_model_name} to {target_path}...")

    export_kwargs = create_export_kwargs(export_kwargs)

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
            num_samples=num_export_samples, model=model, data_loader=data_loader
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
        source_config=getattr(model, "config", None),
        source_tokenizer=tokenizer,
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
                "To enable the validation, set `num_export_samples` "
                "to positive integer"
            )
        validate_correctness_(target_path, deployment_folder_dir, onnx_model_name)

    _LOGGER.info(
        f"Applying optimizations: {graph_optimizations} to the exported model..."
    )

    if helper_functions.apply_optimizations is not None:
        helper_functions.apply_optimizations(
            exported_file_path=os.path.join(deployment_folder_dir, onnx_model_name),
            optimizations=graph_optimizations,
        )

    if save_with_external_data is True or external_data_chunk_size_mb:
        save_model_with_external_data(
            os.path.join(deployment_folder_dir, onnx_model_name),
            external_data_chunk_size_mb,
        )

    if validate_structure and source_path:
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


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
    )
)
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
    default=None,
    help="Onnx opset to export to, defaults to default torch opset",
)
@click.option(
    "--save_with_external_data",
    type=bool,
    default=False,
    help="Default False - if True, large constant tensors, such as initializers, "
    "will be serialised in a separate file. Note: if the model is sufficiently "
    "large, it will be saved with external data regardless of this flag",
)
@click.option(
    "--external_data_chunk_size_mb",
    type=int,
    default=False,
    help="Default False - if explicitely set to a number, "
    "it will force the model to be exported with external "
    "data, with the given chunk size in MB",
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
    type=click.Choice(["image-classification", "transformers"]),
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
@click.argument("kwargs", nargs=-1, type=click.UNPROCESSED)
def main(
    source_path: str,
    target_path: str,
    onnx_model_name: str = ONNX_MODEL_NAME,
    deployment_target: str = "deepsparse",
    opset: Optional[int] = None,
    save_with_external_data: bool = False,
    external_data_chunk_size_mb: Optional[int] = None,
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
    kwargs: Optional[tuple] = None,
):
    export(
        source_path=source_path,
        target_path=target_path,
        onnx_model_name=onnx_model_name,
        deployment_target=deployment_target,
        opset=opset,
        save_with_external_data=save_with_external_data,
        external_data_chunk_size_mb=external_data_chunk_size_mb,
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
        **parse_kwarg_tuples(kwargs) if kwargs is not None else {},
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


if __name__ == "__main__":
    main()
