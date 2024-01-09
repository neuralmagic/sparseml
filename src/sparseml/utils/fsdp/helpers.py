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

try:
    from torch.distributed.fsdp import (
        FullStateDictConfig,
        FullyShardedDataParallel,
        StateDictType,
    )
except ImportError:
    FullyShardedDataParallel = None

from torch.nn import Module

from sparseml.pytorch.model_load.helpers import save_model_and_recipe
from sparseml.utils.pytorch import set_layer


__all__ = [
    "is_fsdp_model",
    "unwrap_and_export_model",
    "save_pretrained_fsdp",
]


def is_fsdp_model(model: Module) -> bool:
    """
    Check if a model instance is wrapped by FSDP

    :param model: pytorch model to check
    :return: True if module is wrapped, False otherwise
    """
    if not FullyShardedDataParallel:
        return False

    return isinstance(model, FullyShardedDataParallel)


def unwrap_and_export_model(model, accelerator, output_dir, tokenizer):
    """
    Recursively unwraps an FSDP model, then saves the unwrapped model and the
    currently active recipe to disk

    :param model: model to unwrap
    :param accelerator: Accelerator instance used to perform unwrapping
    :param output_dir: where to save output model
    :param tokenizer: tokenizer used by the model
    """
    full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FullyShardedDataParallel.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        full_state_dict_config,
    ):
        unwrapped_model = accelerator.unwrap_model(model)
        for name, module in unwrapped_model.named_modules():
            if isinstance(module, FullyShardedDataParallel):
                set_layer(name, accelerator.unwrap_model(module), unwrapped_model)

        save_model_and_recipe(
            model=unwrapped_model,
            save_path=output_dir,
            tokenizer=tokenizer,
        )


def save_pretrained_fsdp(model, accelerator, output_dir):
    full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    """
    Gathers the full FSDP state dict of the model onto rank0 GPU, then uses it to save
    the pretrained FSDP model to disk

    :param model: model to save
    :param accelerator: Accelerator instance used to perform unwrapping
    :param output_dir: where to save output model
    """
    with FullyShardedDataParallel.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, full_state_dict_config
    ):
        state_dict = accelerator.get_state_dict(model, unwrap=False)

    accelerator.unwrap_model(model).save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=state_dict,
    )
