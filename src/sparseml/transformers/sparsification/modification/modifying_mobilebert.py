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

"""
Modification to the original MobileBert model required in the
context of SparseML quantization
"""

from torch import nn
from transformers.models.mobilebert.modeling_mobilebert import MobileBertEmbeddings

from sparseml.modifiers.quantization.modification.modification_objects import QATLinear
from sparseml.modifiers.quantization.modification.registry import ModificationRegistry
from sparseml.pytorch.utils.helpers import swap_modules
from sparseml.transformers.sparsification.modification.base import (
    check_transformers_version,
)


@ModificationRegistry.register(name="MobileBertModel")
def modify(model: nn.Module) -> nn.Module:
    """
    Modify the MobileBert model to be compatible with SparseML
    quantization

    Replaces the MobileBertEmbeddings modules with
    MobileBertEmbeddingsWithQuantizableMatmuls modules

    :param model: the original MobileBert model
    :return: the modified MobileBert model
    """
    check_transformers_version()
    for name, submodule in model.named_modules():
        if isinstance(submodule, MobileBertEmbeddings) and not isinstance(
            submodule, MobileBertEmbeddingsWithQuantizableLinear
        ):
            swap_modules(
                model, name, MobileBertEmbeddingsWithQuantizableLinear(submodule)
            )
    return model


class MobileBertEmbeddingsWithQuantizableLinear(MobileBertEmbeddings):
    """
    Wrapper around the original MobileBertEmbeddings module to replace the
    linear projection with a quantizable linear projection

    :param mobilebert_emb: the original MobileBertEmbeddings module
    """

    def __init__(self, mobilebert_emb: MobileBertEmbeddings):
        self.__class__ = type(
            self.__class__.__name__,
            (self.__class__, mobilebert_emb.__class__),
            {},
        )
        self.__dict__ = mobilebert_emb.__dict__

        embed_dim_multiplier = 3 if self.trigram_input else 1
        embedded_input_size = self.embedding_size * embed_dim_multiplier

        self.embedding_transformation = QATLinear(embedded_input_size, self.hidden_size)
