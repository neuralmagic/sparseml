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

from torch.nn import Module

import sparseml.core.session as session_manager
from sparseml.core.framework import Framework
from sparseml.pytorch.model_load import reload_model_state


__all__ = ["apply_recipe_structure_to_model"]

_LOGGER = logging.getLogger(__name__)


def apply_recipe_structure_to_model(model: Module, recipe_path: str, model_path: str):
    """
    Takes a loaded Pytorch model and applies any structural changes such as quantization
    to the model, then reloads the model.

    :param model: PyTorch model to apply structure to
    :param recipe_path: path to recipe to apply to the model
    :param model_path: path to model, used for reloading the state dict
    """
    orig_state_dict = model.state_dict()

    # apply structural changes to the model
    if not session_manager.active_session():
        session_manager.create_session()
    session_manager.pre_initialize_structure(
        model=model, recipe=recipe_path, framework=Framework.pytorch
    )

    session = session_manager.active_session()
    num_stages = len(session.lifecycle.recipe_container.compiled_recipe.stages)
    msg = (
        "an unstaged recipe"
        if num_stages == 1
        else f"a staged recipe with {num_stages} stages"
    )
    _LOGGER.info(f"Applied {msg} to the model at {model_path}")

    # reload the state dict for the model now that architecture matches expected
    if reload_model_state(model, model_path, orig_state_dict):
        _LOGGER.info(
            "Reloaded model state after SparseML recipe structure modifications "
            f"from {model_path}"
        )
