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

from sparseml.transformers.utils.load_task_dataset import load_task_dataset
from src.sparseml.integration_helper_functions import (
    IntegrationHelperFunctions,
    Integrations,
)
from src.sparseml.transformers.utils.initializers import (
    _parse_data_args,
    initialize_config,
    initialize_model,
    initialize_tokenizer,
    initialize_trainer,
    resolve_sequence_length,
)


def create_model(source_path: Union[Path, str], **kwargs) -> torch.nn.Module:
    """
    Create a model and the data loader from the source path

    :param source_path: The path to the source code for the model
    :param kwargs: Additional arguments to pass to the model creation
    :return: The model and the validation dataset
    """
    config_args = kwargs.get("config_args", {})
    sequence_length = kwargs.get("sequence_length", None)
    task = kwargs.get("task", None)
    data_args = kwargs.get("data_args", {})

    config = initialize_config(source_path, trust_remote_code=True, **config_args)
    sequence_length = sequence_length or resolve_sequence_length(config)
    tokenizer = initialize_tokenizer(source_path, sequence_length, task)
    model = initialize_model(source_path, **kwargs)

    data_args = _parse_data_args(data_args)
    if data_args:
        dataset = load_task_dataset(
            task=task,
            tokenizer=tokenizer,
            data_args=data_args,
            model=model,
            config=config,
        )
        validation_dataset = dataset.get("validation")
    else:
        validation_dataset = None

    model.train()
    initialize_trainer(model, source_path, validation_dataset)
    model.eval()
    return model, validation_dataset


@IntegrationHelperFunctions.register(name=Integrations.transformers.value)
class Transformers(IntegrationHelperFunctions):
    create_model: Any = Field(default=create_model)
