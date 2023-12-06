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

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from pydantic import Field

from src.sparseml.integration_helper_functions import (
    IntegrationHelperFunctions,
    Integrations,
)
from src.sparseml.pytorch.image_classification.utils.helpers import (
    create_model as create_image_classification_model,
)


def create_model(
    source_path: Union[Path, str], **kwargs
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    A contract to create a model from a source path

    :param source_path: The path to the model
    :param kwargs: Additional kwargs to pass to the model creation function
    :return: A tuple of the
        - torch model
        - additional dictionary of items created during model creation
    """
    model, *_, validation_loader = create_image_classification_model(
        checkpoint_path=source_path, **kwargs
    )
    return model, dict(validation_loader=validation_loader)


def create_dummy_input(
    validation_loader: Optional[torch.utils.data.DataLoader] = None,
    image_size: Optional[int] = 224,
) -> torch.Tensor:
    """
    A contract to create a dummy input for a model

    :param validation_loader: The validation loader to get a batch from.
        If None, a fake batch will be created
    :param image_size: The image size to use for the dummy input. Defaults to 224
    :return: The dummy input as a torch tensor
    """

    if not validation_loader:
        # create fake data for export
        val_loader = [[torch.randn(1, 3, image_size, image_size)]]
    return next(iter(val_loader))[0]


@IntegrationHelperFunctions.register(name=Integrations.image_classification.value)
class ImageClassification(IntegrationHelperFunctions):
    create_model: Callable[..., Tuple[torch.nn.Module, Dict[str, Any]]] = Field(
        default=create_model
    )
    create_dummy_input: Callable[..., torch.Tensor] = Field(default=create_dummy_input)
