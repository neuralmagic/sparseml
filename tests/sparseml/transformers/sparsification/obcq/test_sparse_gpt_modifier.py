import os

import pytest

from sparseml.transformers.sparsification.obcq.sparse_opt_modifier import SparseOPTModifier

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
        sequential_update=sequential_update
    )

    assert isinstance(yaml_modifier, SparseOPTModifier)
    assert isinstance(serialized_modifier, SparseOPTModifier)
    assert (
        yaml_modifier.sparsity
        == serialized_modifier.sparsity
        == obj_modifier.sparsity
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