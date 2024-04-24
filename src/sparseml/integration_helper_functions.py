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
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from sparseml.export.export_data import create_data_samples as create_data_samples_
from sparseml.export.export_torch_model import export_model
from sparsezoo.utils.registry import RegistryMixin


__all__ = ["IntegrationHelperFunctions", "resolve_integration"]


class Integrations(Enum):
    """
    Holds the names of the available integrations.
    """

    image_classification = "image-classification"
    transformers = "transformers"


def resolve_integration(
    source_path: Union[Path, str, None] = None,
    source_model: Optional["PreTrainedModel"] = None,  # noqa F401
    integration: Optional[str] = None,
) -> str:
    """
    Resolve the integration to use.

    If integration is not provided, attempt to infer it from the source_path or model.
    Once the integration is resolved, perform the hot import to register
    the integration helper functions.

    :param source_path: The path to the PyTorch model to export.
    :param source_model: An instantiated model to export
    :param integration: Optional name of the integration to use. If not provided,
        will attempt to infer it from the source_path.
    :return: The name of the integration to use for exporting the model.
    """

    integration = integration or _infer_integration_from_source_path(source_path)

    # attempt to infer transformers based on model attribute
    if source_model is not None and hasattr(source_model, "config_class"):
        integration = Integrations.transformers.value

    if integration == Integrations.image_classification.value:
        import sparseml.pytorch.image_classification.integration_helper_functions  # noqa F401

        return Integrations.image_classification.value

    elif integration == Integrations.transformers.value:
        import sparseml.transformers.integration_helper_functions  # noqa F401

        return Integrations.transformers.value
    else:
        raise ValueError(
            f"Could not infer integration from source_path:\n{source_path}\n"
            "Please specify an argument `integration` from one of "
            "the available integrations: "
            f"{[integration.value for integration in Integrations]}."
        )


def remove_past_key_value_support_from_config(config):
    """
    Modify config of the causal language model so that it turns off the
    past key value support. This means that the model initialized from
    this config will not take past key values as input and will not output
    past key values.
    """
    # not take past_key_values as input
    config.is_decoder = True
    # whether to use past key values an input
    config.use_past = False
    # whether to output past key values
    config.use_cache = False
    return config


def _infer_integration_from_source_path(
    source_path: Union[Path, str, None] = None
) -> Optional[str]:
    """
    Infer the integration to use from the source_path.

    :param source_path: The path to the PyTorch model to export.
    :return: The name of the integration to use for exporting the model.
    """
    if source_path is None:
        return None

    try:
        from sparseml.pytorch.image_classification.utils.helpers import (
            is_image_classification_model,
        )
    except ImportError:
        # unable to import integration, always return False
        is_image_classification_model = _null_is_model

    try:
        from sparseml.transformers.utils.helpers import is_transformer_model
    except ImportError:
        # unable to import integration, always return False
        is_transformer_model = _null_is_model

    if is_image_classification_model(source_path):
        return Integrations.image_classification.value
    elif is_transformer_model(source_path):
        return Integrations.transformers.value
    else:
        return None


def _null_is_model(*args, **kwargs):
    # convenience function to always return False for an integration
    # to be used if that integration is not importable
    return False


class IntegrationHelperFunctions(RegistryMixin, BaseModel):
    """
    Registry that maps names to helper functions
    for creation/export/manipulation of models for a specific
    integration.
    """

    create_model: Callable[
        [Union[str, Path]],
        Tuple[
            "torch.nn.Module",  # noqa F821
            Optional[Dict[str, Any]],
        ],
    ] = Field(
        description="A function that takes: "
        "- a source path to a PyTorch model "
        "- (optionally) additional arguments"
        "and returns: "
        "- a (sparse) PyTorch model "
        "- (optionally) loaded_model_kwargs "
        "(any relevant objects created along with the model)"
    )
    create_data_loader: Callable[
        [],
        Tuple[
            Union["torch.utils.data.DataLoader", Generator],  # noqa F821
            Optional[Dict[str, Any]],
        ],
    ] = Field(
        description="A function that takes: "
        "arbitrary arguments and returns: "
        "- a dataloader "
        "- (optionally) loaded data_loader kwargs "
        "(any relevant objects created along with the data_loader)"
    )
    create_dummy_input: Callable[[Any], "torch.Tensor"] = Field(  # noqa F821
        description="A function that takes: "
        "- appropriate arguments "
        "and returns: "
        "- a dummy input for the model (a torch.Tensor) "
    )
    export: Callable[[Any], str] = Field(
        description="A function that takes: "
        " - a (sparse) PyTorch model "
        " - sample input data "
        " - the path to save the exported model to "
        " - the name to save the exported ONNX model as "
        " - the deployment target to export to "
        " - the opset to use for the export "
        " - (optionally) a dictionary of additional arguments"
        "and returns the path to the exported model",
        default=export_model,
    )
    apply_optimizations: Optional[Callable[[Any], None]] = Field(
        None,
        description="A function that takes:"
        " - path to the exported model"
        " - names of the optimizations to apply"
        " and applies the optimizations to the model",
    )

    create_data_samples: Callable[
        [
            Tuple[
                Optional["torch.nn.Module"], int, Optional[Dict[str, Any]]  # noqa: F821
            ]
        ],
        Tuple[
            List["torch.Tensor"],  # noqa F821
            Optional[List["torch.Tensor"]],  # noqa F821
            Optional[List["torch.Tensor"]],  # noqa F821
        ],
    ] = Field(
        default=create_data_samples_,
        description="A function that takes: "
        " - (optionally) a (sparse) PyTorch model "
        " - the number of samples to generate "
        " - (optionally) loaded_model_kwargs "
        "(any relevant objects created along with the model) "
        "and returns: "
        " - the inputs, (optionally) labels and (optionally) outputs as torch tensors ",
    )

    deployment_directory_files_mandatory: List[str] = Field(
        description="A list that describes the "
        "mandatory expected files of the deployment directory",
        default=["model.onnx"],
    )

    deployment_directory_files_optional: Optional[List[str]] = Field(
        None,
        description="A list that describes the "
        "optional expected files of the deployment directory",
    )
