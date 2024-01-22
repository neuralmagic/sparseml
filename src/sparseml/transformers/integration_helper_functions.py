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

from sparseml.export.export_data import create_data_samples as create_data_samples_
from sparseml.export.helpers import apply_optimizations as apply_optimizations_onnx
from sparseml.integration_helper_functions import (
    IntegrationHelperFunctions,
    Integrations,
)
from sparseml.transformers.sparsification.trainer import Trainer
from sparseml.transformers.utils.helpers import (
    ALL_TASK_NAMES,
    MANDATORY_DEPLOYMENT_FILES,
    NLG_MANDATORY_DEPLOYMENT_FILES,
    NLG_OPTIONAL_DEPLOYMENT_FILES,
    OPTIONAL_DEPLOYMENT_FILES,
    TaskNames,
    remove_past_key_value_support_from_config,
    resolve_sequence_length,
)
from sparseml.transformers.utils.initializers import (
    _parse_data_args,
    initialize_config,
    initialize_sparse_model,
    initialize_tokenizer,
    initialize_trainer,
)
from sparseml.transformers.utils.load_task_dataset import load_task_dataset
from sparseml.transformers.utils.optimizations import apply_kv_cache_injection


_LOGGER = logging.getLogger(__name__)


def create_model(
    source_path: Union[Path, str],
    dataset_with_labels: bool = False,
    device: Optional[str] = None,
    task: Optional[str] = None,
    recipe: Optional[str] = None,
    export: bool = True,
    **kwargs,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    A contract to create a model and optional dictionary of
    loaded_model_kwargs (any relevant objects created along with the model)

    :param source_path: The path to the model
    :param dataset_with_labels: Whether to allow the dataset to
        have "labels" inputs or not. Text-generation datasets may
        contain labels (needed for training only)
    :param device: The device to use for the model and dataloader instantiation
    :param task: The task to use for the model and dataloader instantiation
    :param recipe: The recipe to use for the model and dataloader instantiation.
        If None, attempt to use the default recipe
    :param export: Whether the created model is for export or not.

    :return: A tuple of the
        - torch model
        - (optionally) loaded_model_kwargs
          (any relevant objects created along with the model)
    """
    config_args = kwargs.get("config_args", {})
    sequence_length = kwargs.get("sequence_length", None)
    data_args = kwargs.get("data_args", {})
    trust_remote_code = kwargs.get("trust_remote_code", False)

    if task is None:
        raise ValueError(
            "To create a transformer model, a task must be specified. "
            f"Choose one from {ALL_TASK_NAMES}"
        )

    if not trust_remote_code:
        _LOGGER.warning(
            "trust_remote_code is set to False. It is possible, "
            "that the model will not be loaded correctly."
        )

    config = initialize_config(source_path, trust_remote_code, **config_args)
    sequence_length = sequence_length or resolve_sequence_length(config)
    tokenizer = initialize_tokenizer(source_path, sequence_length, task)
    if export:
        if task in TaskNames.text_generation.value:
            config = remove_past_key_value_support_from_config(config)

    model = initialize_sparse_model(
        model_path=source_path,
        task=task,
        config=config,
        trust_remote_code=trust_remote_code,
        recipe=recipe,
        sequence_length=sequence_length,
        device=device,
    )

    validation_dataset = None
    data_args = _parse_data_args(data_args)

    if data_args:
        validation_dataset = load_task_dataset(
            task=task,
            tokenizer=tokenizer,
            data_args=data_args,
            model=model,
            config=config,
            split="validation",
        )
        if task in TaskNames.text_generation.value and not dataset_with_labels:
            validation_dataset = validation_dataset.remove_columns("labels")

    trainer = initialize_trainer(model, source_path, validation_dataset)

    return model, dict(
        trainer=trainer,
        tokenizer=tokenizer,
        input_names=list(next(trainer._get_fake_dataloader(1, tokenizer)).keys()),
    )


def create_dummy_input(
    trainer: Optional[Trainer] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    **kwargs,
) -> torch.Tensor:
    if trainer.eval_dataset is not None:
        data_loader = trainer.get_eval_dataloader()
    else:
        if not tokenizer:
            raise ValueError(
                "Tokenizer is needed to generate "
                "fake sample inputs when the trainer is "
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
    if kwargs.get("batch_size"):
        _LOGGER.info(
            "For exporting samples for transformers integration,"
            "batch size is ignored (equal to 1)"
        )
    if trainer.eval_dataset is None:
        raise ValueError(
            "Attempting to create data samples without an eval dataloader. "
            "Initialize a trainer with an eval dataset"
        )

    return create_data_samples_(
        data_loader=trainer.get_eval_dataloader(), model=model, num_samples=num_samples
    )


def apply_optimizations_generative_transformer(
    exported_file_path: Union[str, Path],
    optimizations: Union[str, List[str]],
    single_graph_file: bool = True,
):

    if exported_file_path.endswith(".onnx"):
        available_optimizations = dict(kv_cache_injection=apply_kv_cache_injection)
        apply_optimizations_onnx(
            onnx_file_path=exported_file_path,
            target_optimizations=optimizations,
            available_optimizations=available_optimizations,
            single_graph_file=single_graph_file,
        )
    else:
        raise NotImplementedError(
            "Applying optimizations is only supported for ONNX files"
        )


@IntegrationHelperFunctions.register(name=Integrations.transformers.value)
class Transformers(IntegrationHelperFunctions):
    def __init__(self, *args, **kwargs):
        super().__init__()
        task = kwargs.get("task")
        if task is None:
            _LOGGER.warning("The task for transformers is not specified.")
        elif task in TaskNames.text_generation.value:
            # if the task is text generation, alter the default attributes
            # to reflect the idiosyncrasies for text generation
            self.apply_optimizations = apply_optimizations_generative_transformer
            self.deployment_directory_files_mandatory = list(
                MANDATORY_DEPLOYMENT_FILES.union(NLG_MANDATORY_DEPLOYMENT_FILES)
            )
            self.deployment_directory_files_optional = list(
                OPTIONAL_DEPLOYMENT_FILES.union(NLG_OPTIONAL_DEPLOYMENT_FILES)
            )
        else:
            _LOGGER.info(
                "Fetching default helper functions for transformers integration"
            )

    create_model: Callable[..., Tuple[torch.nn.Module, Dict[str, Any]]] = Field(
        default=create_model
    )
    create_dummy_input: Callable[..., torch.Tensor] = Field(default=create_dummy_input)
    create_data_samples: Callable = Field(create_data_samples)
    deployment_directory_files_mandatory: List[str] = Field(
        default=list(MANDATORY_DEPLOYMENT_FILES)
    )
    deployment_directory_files_optional: List[str] = Field(
        default=list(OPTIONAL_DEPLOYMENT_FILES)
    )
