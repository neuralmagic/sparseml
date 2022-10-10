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


from typing import Dict, List, Optional, Union

from torch.nn import Module


__all__ = [
    "recipe_template",
]

from sparseml.recipe_template.utils import (
    ModifierBuildInfo,
    get_pruning_info,
    get_quantization_info,
    get_training_info,
)


def recipe_template(
    pruning: Union[str, bool] = False,
    quantization: Union[str, bool] = False,
    lr_func: str = "linear",
    mask_type: str = "unstructured",
    global_sparsity: bool = False,
    target: str = "vnni",
    model: Optional[Module] = None,
    convert_to_md: bool = False,
):
    """
    Return a valid recipe given specified args and kwargs

    :param pruning: Union[str, bool] flag, if "true" or `True` then pruning
        info will be added, can also be set to `acdc`, or `gmp`.
    :param quantization: Union[str, bool] flag, if "true" or `True` then quantization
        info will be added, else not.
    :param lr_func: str representing the learning rate function to use
    :param mask_type: A str representing the mask type for pruning modifiers,
        defaults to "unstructured", can be set to 4-block for `vnni` specific
        quantization
    :param global_sparsity: if set then induce sparsity globally rather than layer wise
    :param target: A str representing the target deployment hardware. defaults to
        `vnni`, can also be set to `tensorrt`
    :param model: An Optional instantiated torch module, that needs to be pruned
        and/or quantized
    :param convert_to_md: A boolean flag, if True the string is returned with
            yaml front matter that can be embedded in a .md file
    :return: A str formatted in yaml or md representing a valid recipe
    """
    # This is where we setup all the modifiers
    modifier_groups: Dict[str, List[ModifierBuildInfo]] = {
        "training_modifiers": get_training_info(lr_func=lr_func),
        "pruning_modifiers": get_pruning_info(
            pruning=pruning,
            mask_type=mask_type,
            global_sparsity=global_sparsity,
            model=model,
        ),
        "quantization_modifiers": get_quantization_info(
            quantization=quantization,
            target=target,
            model=model,
        ),
    }

    yaml_recipe = ModifierBuildInfo.build_recipe_from_modifier_info(
        modifier_info_groups=modifier_groups,
        convert_to_md=convert_to_md,
    )
    return yaml_recipe
