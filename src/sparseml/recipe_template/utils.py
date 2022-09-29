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

from typing import Any, Dict, List, Optional, Union

from torch.nn import Module

from sparseml.pytorch.sparsification import (
    ACDCPruningModifier,
    EpochRangeModifier, GMPruningModifier,
    LearningRateFunctionModifier, MagnitudePruningModifier,
    Modifier, QuantizationModifier,
)
from sparseml.pytorch.utils import get_prunable_layers, get_quantizable_layers
from sparseml.sparsification import ModifierYAMLBuilder

__all__ = [
    "ModifierBuildInfo",
    "get_pruning_info",
    "get_quantization_info",
    "get_training_info",
]


class ModifierBuildInfo:
    """
    A class with state and helper methods for building a recipe from
    modifier(s)

    :param modifier: class of the Modifier in question
    :param modifier_name: str A string name to be prefixed with all attribute
        variables defined for this modifier, defaults to "mod"
    :param fields: An Optional dictionary mapping modifier attribute names to their
        corresponding values
    """

    def __init__(
        self,
        modifier: Modifier,
        modifier_name: str = "mod",
        fields: Optional[Dict[str, Any]] = None,
    ):
        self.modifier = modifier
        self.modifier_name = modifier_name
        self.modifier_builder = ModifierYAMLBuilder(modifier)
        self.fields = fields or {}
        self.__modifier_recipe_variables = {}
        self.update(updated_fields=self.fields)

    def update(self, updated_fields: Optional[Dict[str, Any]] = None):
        """
        Utility method to update the fields for current modifier

        :param updated_fields: An Optional dictionary mapping modifier attribute names
            to their corresponding values
        """
        for key, value in updated_fields.items():
            variable_name = f"{self.modifier_name}_{key}"
            setattr(self.modifier_builder, key, f"eval({variable_name})")
            self.__modifier_recipe_variables[variable_name] = updated_fields[key]

    @property
    def modifier_recipe_variables(self) -> Dict[str, Any]:
        """
        :returns: A dict mapping b/w recipe variable names --> values
            for the current object
        """
        return self.__modifier_recipe_variables


# PRUNING MODIFIERS INFO

_PRUNING_MODIFIER_INFO_REGISTRY = {
    "false": None,
    "true": ModifierBuildInfo(
        modifier=MagnitudePruningModifier,
        modifier_name="pruning",
        fields={
            "init_sparsity": 0.05,
            "final_sparsity": 0.8,
            "start_epoch": 0.0,
            "end_epoch": 10.0,
            "update_frequency": 1.0,
            "params": "__ALL_PRUNABLE__",
            "leave_enabled": True,
            "inter_func": "cubic",
            "mask_type": "unstructured",
        },
    ),
    "acdc": ModifierBuildInfo(
        modifier=ACDCPruningModifier,
        modifier_name="pruning",
        fields={
            "compression_sparsity": 0.9,
            "start_epoch": 0,
            "end_epoch": 100,
            "update_frequency": 5,
            "params": "__ALL_PRUNABLE__",
            "global_sparsity": True,
        },
    ),
    "gmp": ModifierBuildInfo(
        modifier=GMPruningModifier,
        modifier_name="pruning",
        fields={
            "init_sparsity": 0.05,
            "final_sparsity": 0.8,
            "start_epoch": 0.0,
            "end_epoch": 10.0,
            "update_frequency": 1.0,
            "params": ["re:.*weight"],
            "leave_enabled": True,
            "inter_func": "cubic",
            "mask_type": "unstructured",
        },
    ),
}

# QUANTIZATION MODIFIERS INFO

_QUANTIZATION_MODIFIER_INFO_REGISTRY = {
    "false": None,
    "true": ModifierBuildInfo(
        modifier=QuantizationModifier,
        modifier_name="quantization",
        fields={
            "start_epoch": 0.0,
            "submodules": "null",
            "model_fuse_fn_name": "fuse_module",
            "disable_quantization_observer_epoch": 2.0,
            "freeze_bn_stats_epoch": 3.0,
            "reduce_range": False,
            "activation_bits": False,
            "tensorrt": False,
        },
    ),
}


# getters

def get_quantization_info(
    quantization: Union[str, bool] = False,
    target: str = "vnni",
    model: Optional[Module] = None,
) -> List[ModifierBuildInfo]:
    """
    :param quantization: Union[str, bool] flag, if "true" or `True` then quantization
        info will be added, else not.
    :param target: A str representing the target deployment hardware. defaults to
        `vnni`, can also be set to `tensorrt`
    :param model: An Optional instantiated torch module, that needs to be quantized
    :returns: a List[ModifierBuildInfo] containing quantization info if any
    """
    if isinstance(quantization, bool):
        quantization = str(quantization)

    quantization_info = _QUANTIZATION_MODIFIER_INFO_REGISTRY.get(quantization.lower())
    if quantization_info is None:
        return []

    quantization_info.update({"tensorrt": target == "tensorrt"})

    if model:
        quantizable_layers = [name for name, _ in get_quantizable_layers(module=model)]
        quantization_info.update({"submodules": quantizable_layers})

    return [quantization_info]


def get_pruning_info(
    pruning: Union[str, bool] = False,
    mask_type: str = "unstructured",
    global_sparsity: bool = False,
    model: Optional[Module] = None,
) -> List[ModifierBuildInfo]:
    """
    :param pruning: Union[str, bool] flag, if "true" or `True` then pruning
        info will be added, can also be set to `acdc`, or `gmp`.
    :param mask_type: A str representing the mask type for pruning modifiers,
        defaults to "unstructured", can be set to 4-block for `vnni` specific
        quantization
    :param global_sparsity: if set then induce sparsity globally rather than layer wise
    :param model: An Optional instantiated torch module, that needs to be quantized
    :returns: a List[ModifierBuildInfo] containing pruning info if any
    """
    if isinstance(pruning, bool):
        pruning = str(pruning)

    modifier_info = _PRUNING_MODIFIER_INFO_REGISTRY.get(pruning.lower())
    if modifier_info is None:
        return []

    if isinstance(
        modifier_info.modifier, (MagnitudePruningModifier, GMPruningModifier)
    ):
        modifier_info.update(
            updated_fields={
                "mask_type": mask_type,
                "global_sparsity": global_sparsity,
            }
        )
    elif isinstance(modifier_info.modifier, ACDCPruningModifier):
        modifier_info.update(
            updated_fields={
                "global_sparsity": global_sparsity,
            }
        )

    if model:
        prunable_layers = [name for name, module in get_prunable_layers(module=model)]
        modifier_info.update({"params": prunable_layers})

    return [modifier_info]


def get_training_info(lr_func: str = "linear") -> List[ModifierBuildInfo]:
    """
    :param lr_func: str representing the learning rate function to use
    :return: a List[ModifierBuildInfo] containing information on training modifiers
        to be added
    """
    epoch_modifier_info = ModifierBuildInfo(
        modifier=EpochRangeModifier,
        modifier_name="epoch_range_mod",
        fields={
            "start_epoch": 0.0,
            "end_epoch": 10,
        })
    lr_modifier_info = ModifierBuildInfo(
        modifier=LearningRateFunctionModifier,
        modifier_name="lr_function_mod",
        fields={
            "start_epoch": 0.0,
            "end_epoch": 10,
            "lr_func": lr_func,
            "init_lr": "1e-3",
            "final_lr": "1e-8",
        })

    return [epoch_modifier_info, lr_modifier_info]
