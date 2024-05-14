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
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import torch
from pydantic import Field

from sparseml.export.export_data import create_data_samples as create_data_samples_
from sparseml.export.helpers import apply_optimizations as apply_optimizations_onnx
from sparseml.integration_helper_functions import (
    IntegrationHelperFunctions,
    Integrations,
    remove_past_key_value_support_from_config,
)
from sparseml.transformers.finetune.data.data_helpers import format_calibration_data
from sparseml.transformers.utils.helpers import (
    ALL_TASK_NAMES,
    MANDATORY_DEPLOYMENT_FILES,
    NLG_MANDATORY_DEPLOYMENT_FILES,
    NLG_OPTIONAL_DEPLOYMENT_FILES,
    OPTIONAL_DEPLOYMENT_FILES,
    TaskNames,
    create_fake_dataloader,
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
    :param device: The device to use for the model
    :param task: The task to use for the model
    :param recipe: The recipe to use for the model
    :param export: Whether the created model is for export or not.

    :return: A tuple of:
        - torch model
        - dict of loaded_model_kwargs
    """
    config_args = kwargs.get("config_args", {})
    sequence_length = kwargs.get("sequence_length", None)
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
    return model, dict(
        tokenizer=tokenizer, sequence_length=sequence_length, config=config
    )


def create_data_loader(
    model: torch.nn.Module,
    task: str,
    data_args: Optional[Dict[str, Any]] = None,
    config: Optional["AutoConfig"] = None,  # noqa F821
    source_path: Optional[str] = None,
    sequence_length: int = 384,
    tokenizer: Optional["AutoTokenizer"] = None,  # noqa F821
    dataset_with_labels: bool = False,
    **kwargs,
):
    """
    A contract to create a dataloader and optional dictionary of
    loaded_dataloader_kwargs (any relevant objects created along with the dataloader)

    :param model: A model for which the data_loader is created
    :param task: The task to use for the model
    :param data_args: Arguments for instantiation of the dataset
    :param source_path: Path to the model files
    :param sequence_length: The sequence length to use for the model
    :param tokenizer: The tokenizer to use for the model
    :param dataset_with_labels: Whether to allow the dataset to
        have "labels" inputs or not. Text-generation datasets may
        contain labels (needed for training only)

    :return: A tuple of:
        - torch model
        - dict of loaded_model_kwargs
    """
    split = kwargs.get("split", None)

    config = config or model.config
    source_path = source_path or model.name_or_path
    if tokenizer is None:
        if sequence_length is None:
            raise ValueError(
                "Sequence length for the transformer model export missing. "
                "Provide it manually using sequence_length argument"
            )
        tokenizer = initialize_tokenizer(config.name_or_path, sequence_length, task)
    data_args = _parse_data_args(data_args or {})

    if data_args:
        dataset = load_task_dataset(
            task=task,
            tokenizer=tokenizer,
            data_args=data_args,
            model=model,
            config=config,
            split=split,
        )

        if task in TaskNames.text_generation.value:
            # text-generation datasets have a separate
            # logic for creating a dataloader
            if not dataset_with_labels:
                dataset = dataset.remove_columns("labels")
            data_loader = format_calibration_data(tokenized_dataset=dataset)
            input_names = list(next(iter(data_loader)).keys())

        else:
            trainer = initialize_trainer(model, source_path, dataset)
            data_loader = trainer.get_eval_dataloader()
            input_names = list(next(trainer._get_fake_dataloader(1, tokenizer)).keys())

    else:
        # if no data_args are provided, create a fake dataloader
        data_loader, input_names = create_fake_dataloader(
            model, tokenizer, num_samples=1
        )

    return data_loader, dict(input_names=input_names)


def create_dummy_input(
    data_loader: torch.utils.data.DataLoader,
    **kwargs,
) -> torch.Tensor:

    return next(iter(data_loader))


def create_data_samples(
    num_samples: int,
    data_loader: torch.utils.data.DataLoader,
    model: Optional["torch.nn.Module"] = None,
    **kwargs,
):
    if kwargs.get("batch_size"):
        _LOGGER.info(
            "For exporting samples for transformers integration,"
            "batch size is ignored (equal to 1)"
        )

    return create_data_samples_(
        data_loader=data_loader, model=model, num_samples=num_samples
    )


def apply_optimizations_generative_transformer(
    exported_file_path: Union[str, Path],
    optimizations: Union[str, List[str]],
):

    if exported_file_path.endswith(".onnx"):
        available_optimizations = dict(kv_cache_injection=apply_kv_cache_injection)
        apply_optimizations_onnx(
            onnx_file_path=exported_file_path,
            available_optimizations=available_optimizations,
            target_optimizations=optimizations,
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

    create_model: Callable[[], Tuple[torch.nn.Module, Dict[str, Any]]] = Field(
        default=create_model
    )
    create_data_loader: Callable[
        [], Tuple[Union[Generator, torch.utils.data.DataLoader], Dict[str, Any]]
    ] = Field(default=create_data_loader)
    create_dummy_input: Callable[[], torch.Tensor] = Field(default=create_dummy_input)
    create_data_samples: Callable = Field(create_data_samples)
    deployment_directory_files_mandatory: List[str] = Field(
        default=list(MANDATORY_DEPLOYMENT_FILES)
    )
    deployment_directory_files_optional: List[str] = Field(
        default=list(OPTIONAL_DEPLOYMENT_FILES)
    )
