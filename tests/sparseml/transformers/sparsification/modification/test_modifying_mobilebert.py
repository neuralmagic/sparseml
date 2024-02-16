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

from copy import deepcopy

from torch import nn

from sparseml.transformers.sparsification.modification import modify_model
from sparseml.transformers.sparsification.modification.modification_objects import (
    QATLinear,
)


def test_modifying_distilbert(mobilebert_model):
    from sparseml.transformers.sparsification.modification.modifying_mobilebert import (  # noqa F401
        modify,
    )

    mobilebert_ = deepcopy(mobilebert_model)
    mobilebert = modify_model(mobilebert_model)

    assert isinstance(mobilebert_.embeddings.embedding_transformation, nn.Linear)
    assert isinstance(mobilebert.embeddings.embedding_transformation, QATLinear)
