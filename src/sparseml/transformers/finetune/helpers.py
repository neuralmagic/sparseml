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
from typing import Any, Dict

import torch

import sparseml.core.session as session_manager
from sparseml.core.framework import Framework
from sparseml.pytorch.sparsification.quantization.helpers import (
    initialize_channel_wise_scale_zp,
)
from sparseml.transformers.utils import SparseAutoModel


_LOGGER = logging.getLogger(__name__)


def apply_recipe_structure_to_model(model, recipe_path, model_path):
    orig_state_dict = model.state_dict()

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
    if _reload_model_state(model, model_path, orig_state_dict):
        _LOGGER.info(
            "Reloaded model state after SparseML recipe structure modifications "
            f"from {model_path}"
        )



# TODO: clean up references to this function
def _reload_model_state(model, load_path: str, orig_state_dict: Dict[str, Any]):
    """
    Reload the weights after model arch changes due to recipe application
    Return True if weights are successfully reloaded; False otherwise
    """
    invalid_load_path = not load_path or not os.path.isdir(load_path)
    files = os.listdir(load_path) if not invalid_load_path else []
    weight_files = [
        os.path.join(load_path, os.path.basename(f))
        for f in files
        if f.startswith("pytorch_model") and f.endswith("bin")
    ]
    if not weight_files:
        _LOGGER.warning(
            "Model state was not reloaded for SparseML: "
            f"could not find model weights for {load_path}"
        )
        return False

    # PerChannel quantization observers initialize variables
    # to dummy shapes that do not match the ones saved in
    # state_dict.
    # Need to reshape these variables in order to load state_dict
    # properly.
    initialize_channel_wise_scale_zp(model)

    current_state_dict = model.state_dict()

    if set(orig_state_dict.keys()) == set(current_state_dict):
        # no change in keys, ignore reload
        return False

    # change in keys due to architecture changes, reload statedict
    loaded_state_dict = {}
    for f in weight_files:
        dd = torch.load(f, map_location="cpu")
        loaded_state_dict.update(dd)

    _, missing, unexpected, mismatched, _, _ = model._load_pretrained_model(
        model=model,
        state_dict=loaded_state_dict,
        loaded_keys=list(loaded_state_dict.keys()),
        resolved_archive_file=None,
        pretrained_model_name_or_path=load_path,
        _fast_init=False,
    )

    if missing:
        _LOGGER.warning(
            "Missing keys found when reloading model state for SparseML recipe:"
            f"{missing}"
        )

    if unexpected:
        _LOGGER.warning(
            f"Unexpected keys found when reloading model state for SparseML recipe:"
            f"{unexpected}"
        )

    if mismatched:
        _LOGGER.warning(
            f"Mismatched keys found when reloading model state for SparseML recipe:"
            f"{mismatched}"
        )

    total_loaded = len(current_state_dict) - (len(missing) if len(missing) else 0)
    _LOGGER.info(
        f"Reloaded {total_loaded} model params for SparseML Recipe from {load_path}"
    )
    SparseAutoModel.log_model_load(
        model,
        load_path,
        model_type="student",
        delayed_load=False,
    )
    return True
