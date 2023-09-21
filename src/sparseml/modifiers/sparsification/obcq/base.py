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


from typing import Optional

from sparseml.core import Modifier


__all__ = ["SparseGPTModifier"]


class SparseGPTModifier(Modifier):
    """
    Modifier for applying the one-shot OBCQ algorithm to a model. This modifier should
    not be run directly and instead is instantiated from one of the child classes:
    SparseOPTModifier, SparseMPTModifier or SparseLlamaModifier.

    Life-cycle:
        - initialze
            - compress
        - finalize

    :param sparsity: Sparsity to compress model to
    :param block_size: Used to determine number of columns to compress in one pass
    :param quantize: Whether or not model is quantized (affects layer names)
    :param dampening_frac: Amount of dampening to apply to H, as a fraction of the
        diagonal norm
    :param sequential_update: Whether or not to update weights sequentially by layer,
        True saves on GPU memory
    """

    sparsity: float
    block_size: int
    quantize: bool
    dampening_frac: Optional[float] = 0.01
    sequential_update: Optional[bool] = True

    def on_initialize_structure(self, state: "State", **kwargs):
        pass  # nothing needed for this modifier
