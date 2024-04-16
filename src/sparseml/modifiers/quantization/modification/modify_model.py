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

from sparseml.modifiers.quantization.modification.registry import ModificationRegistry


_LOGGER = logging.getLogger(__name__)


def modify_model(model: "torch.nn.Module") -> "torch.nn.Module":  # noqa: F821
    """
    Modify the original model so that it is
    compatible with the quantization format required by the
    SparseML library.

    The model will be modified, if there exist a modification
    function for the model in the registry of modifications.
    Otherwise, the original model will be returned.

    :param model: The original model to be modified
    :return: The potentially modified model to support
        SparseML quantization
    """
    model_name = model.__class__.__name__

    try:
        modification_func = ModificationRegistry.get_value_from_registry(model_name)
    except KeyError:
        _LOGGER.debug(
            f"No modification function found for the model {model_name}. "
            "Returning the original model. Available modification functions"
            f"are available for models: {ModificationRegistry.registered_names()}"
        )
        return model

    _LOGGER.info(
        f"Modifying the model {model_name} to be compatible with SparseML quantization"
    )
    return modification_func(model)
