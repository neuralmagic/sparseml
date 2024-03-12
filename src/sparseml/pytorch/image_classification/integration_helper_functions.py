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
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from pydantic import Field

from sparseml.export.export_data import create_data_samples as create_data_samples_
from sparseml.integration_helper_functions import (
    IntegrationHelperFunctions,
    Integrations,
)
from sparseml.pytorch.image_classification.utils.helpers import (
    create_model as create_image_classification_model,
)
from sparseml.pytorch.image_classification.utils.helpers import (
    get_dataset_and_dataloader,
)


def create_model(
    source_path: Union[Path, str], **kwargs
) -> Tuple[torch.nn.Module, Dict]:
    """
    A contract to create a model and optional dictionary of
    loaded_model_kwargs (any relevant objects created along with the model)

    :param source_path: Path to the model files

    :return: A tuple of the
        - torch model
        - (optionally) loaded_model_kwargs
          (any relevant objects created along with the model)
    """

    checkpoint_path = (
        os.path.join(source_path, "model.pth")
        if not os.path.isfile(source_path)
        else source_path
    )

    return (
        create_image_classification_model(checkpoint_path=checkpoint_path, **kwargs)[0],
        {},
    )


def create_data_loader(
    model: "torch.nn.Module",
    batch_size: Optional[int] = 1,
    device: Optional[str] = None,
    **kwargs,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    A contract to create a model and optional dictionary of
    loaded_data_loader_kwargs (any relevant objects created along with the data_loader)

    :param batch_size: The batch size to use for the dataloader creation
    :param device: The device to use for the model and dataloader instantiation

    :return: A tuple of the
        - a data_loader
        - (optionally) loaded_data_loader_kwargs
          (any relevant objects created along with the model)
    """
    dataset_path = kwargs.get("dataset_path", None)
    dataset_name = kwargs.get("dataset_name", None)
    image_size = kwargs.get("image_size", None)

    if dataset_path is not None:
        dataset, dataloader = get_dataset_and_dataloader(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            batch_size=batch_size,
            image_size=image_size,
            training=False,
            loader_num_workers=1,
            loader_pin_memory=False,
            device=device,
        )
    else:
        dataloader = None

    return dataloader, dict(image_size=image_size)


def create_dummy_input(
    data_loader: Optional[torch.utils.data.DataLoader] = None,
    image_size: Optional[int] = None,
    **kwargs,
) -> torch.Tensor:
    """
    A contract to create a dummy input for a model

    :param data_loader: The validation dataloader to get a batch from.
        If None, a fake batch will be created
    :param image_size: The image size to use for the dummy input
    :return: The dummy input as a torch tensor
    """

    if not data_loader:
        # create fake data for export
        if image_size is None:
            raise ValueError(
                "In the absence of validation_dataloader, the "
                "image_size must be provided to create a dummy input"
            )
        data_loader = [[torch.randn(1, 3, image_size, image_size)]]

    return next(iter(data_loader))[0]


def create_data_samples(
    num_samples: int,
    data_loader: Optional[torch.utils.data.DataLoader] = None,
    model: Optional["torch.nn.Module"] = None,
    **kwargs,
):
    if data_loader is None:
        raise ValueError(
            "Attempting to create data samples without a validation dataloader."
        )

    return create_data_samples_(
        data_loader=data_loader, model=model, num_samples=num_samples
    )


@IntegrationHelperFunctions.register(name=Integrations.image_classification.value)
class ImageClassification(IntegrationHelperFunctions):
    create_model: Callable[[], Tuple[torch.nn.Module, Dict[str, Any]]] = Field(
        default=create_model
    )
    create_data_loader: Callable[
        [], Tuple[Union[torch.utils.data.DataLoader], Dict[str, Any]]
    ] = Field(default=create_data_loader)
    create_dummy_input: Callable[[], torch.Tensor] = Field(default=create_dummy_input)
    create_data_samples: Callable = Field(create_data_samples)
