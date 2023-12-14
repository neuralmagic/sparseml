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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from pydantic import Field
from transformers import AutoTokenizer

from sparseml.transformers.sparsification.trainer import Trainer
from sparseml.transformers.utils.helpers import (
    MANDATORY_DEPLOYMENT_FILES,
    NLG_TOKENIZER_FILES,
    OPTIONAL_DEPLOYMENT_FILES,
)
from sparseml.transformers.utils.load_task_dataset import load_task_dataset
from sparseml.transformers.utils.optimizations import apply_kv_cache_injection
from src.sparseml.export.export_data import create_data_samples as create_data_samples_
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


_LOGGER = logging.getLogger(__name__)


# TODO: Think about how to handle batch_size and device here
def create_model(
    source_path: Union[Path, str],
    batch_size: Optional[int],
    device: Optional[str],
    **kwargs,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    A contract to create a model and optional dictionary of
    auxiliary items related to the model

    :param source_path: The path to the model
    :param batch_size: The batch size to use for the dataloader creation
    :param device: The device to use for the model and dataloader instantiation

    :return: A tuple of the
        - torch model
        - (optionally) a dictionary of auxiliary items
    """
    config_args = kwargs.get("config_args", {})
    sequence_length = kwargs.get("sequence_length", None)
    task = kwargs.get("task", None)
    data_args = kwargs.get("data_args", {})
    trust_remote_code = kwargs.get("trust_remote_code", False)

    if task is None:
        raise ValueError("To create a transformer model, a task must be specified")
    if not trust_remote_code:
        _LOGGER.warning(
            "trust_remote_code is set to False. It is possible, "
            "that the model will not be loaded correctly."
        )

    config = initialize_config(
        source_path, trust_remote_code=trust_remote_code, **config_args
    )
    sequence_length = sequence_length or resolve_sequence_length(config)
    tokenizer = initialize_tokenizer(source_path, sequence_length, task)
    model = initialize_model(
        model_path=source_path,
        task=task,
        config=config,
        trust_remote_code=trust_remote_code,
    )

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
    trainer = initialize_trainer(model, source_path, validation_dataset)
    model.eval()

    return model, dict(trainer=trainer, tokenizer=tokenizer)


def create_dummy_input(
    trainer: Optional[Trainer] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    **kwargs,
) -> torch.Tensor:
    try:
        data_loader = trainer.get_eval_dataloader()
    except:
        if not tokenizer:
            raise ValueError(
                "Tokenizer is needed to generate fake sample inputs when the trainer is "
                "not initialized with an eval dataset"
            )
        data_loader = trainer._get_fake_dataloader(num_samples=1, tokenizer=tokenizer)
    return next(iter(data_loader))


def create_data_samples(
    num_samples: int,
    trainer: Trainer,
    model: Optional["torch.nn.Module"] = None,
    **kwargs,
):
    try:
        data_loader = trainer.get_eval_dataloader()
    except:
        raise ValueError()

    return create_data_samples_(
        data_loader=data_loader, model=model, num_samples=num_samples
    )


generative_transformers_graph_optimizations = {
    "kv_cache_injection": apply_kv_cache_injection
}


@IntegrationHelperFunctions.register(name=Integrations.transformers.value)
class Transformers(IntegrationHelperFunctions):
    create_model: Callable[..., Tuple[torch.nn.Module, Dict[str, Any]]] = Field(
        default=create_model
    )
    create_dummy_input: Callable[..., torch.Tensor] = Field(default=create_dummy_input)
    create_data_samples: Callable = Field(create_data_samples)
    deployment_directory_files_mandatory: List[str] = Field(
        default=MANDATORY_DEPLOYMENT_FILES
    )
    deployment_directory_files_optional: List[str] = Field(
        default=OPTIONAL_DEPLOYMENT_FILES
    )


@IntegrationHelperFunctions.register(name="text-generation")
class Transformers(IntegrationHelperFunctions):
    create_model: Callable[..., Tuple[torch.nn.Module, Dict[str, Any]]] = Field(
        default=create_model
    )
    create_dummy_input: Callable[..., torch.Tensor] = Field(default=create_dummy_input)
    create_data_samples: Callable = Field(create_data_samples)
    graph_optimizations: Dict[str, Callable] = Field(
        default=generative_transformers_graph_optimizations
    )
    deployment_directory_files_mandatory: List[str] = Field(
        default=MANDATORY_DEPLOYMENT_FILES.union(NLG_TOKENIZER_FILES)
    )
    deployment_directory_files_optional: List[str] = Field(
        default=OPTIONAL_DEPLOYMENT_FILES
    )
