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

import os

import pytest

from sparseml.transformers.sparsification.obcq.sparse_opt_modifier import (
    SparseOPTModifier,
)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
def test_sparse_opt_yaml():
    sparsity = 0.7
    block_size = 64
    quantize = True
    dampening_frac = 0.001
    sequential_update = True
    yaml_str = f"""
    !SparseOPTModifier
        sparsity: {sparsity}
        block_size: {block_size}
        quantize: {quantize}
        dampening_frac: {dampening_frac}
        sequential_update: {sequential_update}
    """
    yaml_modifier = SparseOPTModifier.load_obj(yaml_str)
    serialized_modifier = SparseOPTModifier.load_obj(str(yaml_modifier))
    obj_modifier = SparseOPTModifier(
        sparsity=sparsity,
        block_size=block_size,
        quantize=quantize,
        dampening_frac=dampening_frac,
        sequential_update=sequential_update,
    )

    assert isinstance(yaml_modifier, SparseOPTModifier)
    assert isinstance(serialized_modifier, SparseOPTModifier)
    assert (
        yaml_modifier.sparsity == serialized_modifier.sparsity == obj_modifier.sparsity
    )
    assert (
        yaml_modifier.block_size
        == serialized_modifier.block_size
        == obj_modifier.block_size
    )
    assert (
        yaml_modifier.dampening_frac
        == serialized_modifier.dampening_frac
        == obj_modifier.dampening_frac
    )
