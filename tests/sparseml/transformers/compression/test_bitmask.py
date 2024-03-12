import torch
import math
from sparseml.transformers.compression import BitmaskConfig, BitmaskCompressor
from sparseml.transformers.compression.compressors.sparse_bitmask import NumpyBitmaskTensor
import pytest

@pytest.mark.parametrize(
    "shape,sparsity",
    [
        [(512,1024), 0.5],
        [(830,545), 0.8],
    ]
)
def test_bitmask_sizes(shape, sparsity):
    test_tensor = torch.rand(shape, dtype=torch.float32)
    mask = (test_tensor.abs() < (1 - sparsity)).int()
    test_tensor *= mask
    dense_state_dict = {"dummy.weight": test_tensor}

    sparsity_config = BitmaskConfig()
    compressor = BitmaskCompressor(config=sparsity_config)
    sparse_state_dict = compressor.compress(dense_state_dict)

    # each dense tensor has 4 parameters for compression
    assert len(dense_state_dict) * 4 == len(sparse_state_dict)

    # bitmask should be 1 bit per dense element, rounded up to nearest int8
    sparse_shape = sparse_state_dict["dummy.weight.shape"]
    assert torch.all(torch.eq(sparse_shape, torch.tensor(shape)))
    bitmask_shape = sparse_state_dict["dummy.weight.bitmask"].shape
    assert(bitmask_shape[0] == sparse_shape[0])
    assert(bitmask_shape[1] == int(math.ceil(sparse_shape[1] / 8.0)))

    # one value for each non-zero weight
    values_shape = sparse_state_dict["dummy.weight.compressed"].shape
    assert values_shape[0] == torch.sum(mask)
    row_offsets_shape = sparse_state_dict["dummy.weight.row_offsets"].shape
    assert row_offsets_shape[0] == test_tensor.shape[0] 
    

@pytest.mark.parametrize(
    "shape,sparsity",
    [
        [(256,512), 0.5],
        [(128,280), 0.8],
    ]
)
def test_match(shape, sparsity):
    test_tensor1 = torch.rand(shape, dtype=torch.float32)
    mask = (test_tensor1.abs() < (1 - sparsity)).int()
    test_tensor1 *= mask

    test_tensor2 = torch.rand(shape, dtype=torch.float32)
    mask = (test_tensor2.abs() < (1 - sparsity)).int()
    test_tensor2 *= mask

    dense_state_dict = {
        "dummy.weight": test_tensor1,
        "dummy2.weight": test_tensor2
    }
    
    for key in dense_state_dict.keys():
        dense_tensor = dense_state_dict[key]
        sparse_tensor = NumpyBitmaskTensor(dense_tensor)
        assert torch.equal(dense_tensor, sparse_tensor.to_dense())