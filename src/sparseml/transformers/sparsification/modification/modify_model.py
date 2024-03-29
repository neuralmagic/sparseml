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
import os

import torch

from sparseml.transformers.sparsification.modification.registry import (
    ModificationRegistry,
)


_LOGGER = logging.getLogger(__name__)


def modify_model(model: torch.nn.Module, disable: int = False) -> torch.nn.Module:
    """
    Modify the original transformers model so that it is
    compatible with the SparseML library.
    The model will be modified, if there exist a modification
    function for the model in the registry of modifications.
    Otherwise, the original model will be returned.

    :param model: The original HuggingFace transformers model
    :return: The potentially modified model
    """
    model_name = model.__class__.__name__
    NM_DISABLE_TRANSFORMERS_MODIFICATION = os.environ.get(
        "NM_DISABLE_TRANSFORMERS_MODIFICATION", "False"
    ).lower() in ["true", "1"]
    try:
        modification_func = ModificationRegistry.get_value_from_registry(model_name)
    except KeyError:
        _LOGGER.debug(
            f"No modification function found for the model {model_name}. "
            "Returning the original model. Available modification functions"
            f"are available for models: {ModificationRegistry.registered_names()}"
        )
        return model

    if NM_DISABLE_TRANSFORMERS_MODIFICATION:
        _LOGGER.debug(
            "Application of the modification function to model "
            "disabled through the environment variable."
        )
        return model

    if disable:
        _LOGGER.debug(
            "Application of the modification function for to model "
            "disabled through the `disable` argument."
        )
        return model

    _LOGGER.info(
        f"Modifying the model {model_name} to be compatible with SparseML library"
    )
    return modification_func(model)
