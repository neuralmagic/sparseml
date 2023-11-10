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

import collections
import inspect
import logging
from typing import Any, Dict, Union

import numpy
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PaddingStrategy


__all__ = ["create_dummy_inputs"]

_LOGGER = logging.getLogger(__name__)


def create_dummy_inputs(
    model: Any, tokenizer: AutoTokenizer, type: str = "pt"
) -> Dict[str, Union["torch.Tensor", numpy.ndarray]]:  # noqa: F821

    if type not in ["pt", "np"]:
        raise ValueError(f"Type of inputs must be one of ['pt', 'np'], got {type}")

    if not hasattr(model, "forward"):
        raise ValueError(
            f"Model: {model} is expected to have a forward function, but it does not"
        )

    inputs: Dict[str, Union["torch.Tensor", numpy.ndarray]] = tokenizer(
        "", return_tensors="pt", padding=PaddingStrategy.MAX_LENGTH.value
    ).data

    # Rearrange inputs' keys to match those defined by model forward function, which
    # defines how the order of inputs is determined in the exported model
    forward_args_spec = inspect.getfullargspec(model.__class__.forward)

    # Drop inputs that were added by the tokenizer and are not expected by the model
    dropped = [
        input_key
        for input_key in inputs.keys()
        if input_key not in forward_args_spec.args
    ]
    if dropped:
        _LOGGER.warning(
            "The following inputs were not present in the model forward function "
            f"and therefore dropped from ONNX export: {dropped}"
        )

    # Rearrange inputs so that they all have shape (batch_size=1, tokenizer.max_length)
    inputs = collections.OrderedDict(
        [
            (
                func_input_arg_name,
                inputs[func_input_arg_name][0].reshape(1, tokenizer.model_max_length),
            )
            for func_input_arg_name in forward_args_spec.args
            if func_input_arg_name in inputs
        ]
    )
    # Map every input to a str: "{dtype}: {shape}" representation
    inputs_shapes: Dict[str, str] = {
        key: (
            f"{val.dtype if hasattr(val, 'dtype') else 'unknown'}: "
            f"{list(val.shape) if hasattr(val, 'shape') else 'unknown'}"
        )
        for key, val in inputs.items()
    }

    _LOGGER.info(f"Created sample inputs for the ONNX export process: {inputs_shapes}")

    return inputs
