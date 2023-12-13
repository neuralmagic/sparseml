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
from typing import Any, Union

import torch
from pydantic import Field

from src.sparseml.integration_helper_functions import (
    IntegrationHelperFunctions,
    Integrations,
)
from src.sparseml.transformers.utils.create_model import (
    create_model as create_transformers_model,
)


def create_model(source_path: Union[Path, str], **kwargs) -> torch.nn.Module:
    """
    A contract to create a model from a source path

    :param source_path: The path to the model
    :return: The torch model
    """
    model, *_, validation_loader = create_transformers_model(
        model_path=source_path, **kwargs
    )
    model.eval()
    return model, dict(validation_loader=validation_loader)


@IntegrationHelperFunctions.register(name=Integrations.transformers.value)
class Transformers(IntegrationHelperFunctions):
    create_model: Any = Field(default=create_model)
