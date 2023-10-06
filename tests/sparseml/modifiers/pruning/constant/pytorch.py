import numpy
import pytest
from sparseml.core.framework import Framework
from sparseml.pytorch.utils import tensor_sparsity


def _induce_sparsity(model: "Module", sparsity=0.5) -> "Module":
    """
    Introduces sparsity to the given model by zeroing out weights
    with a probability of sparsity
    
    :param model: the model to introduce sparsity to
    :param sparsity: the probability of zeroing out a weight
    :return: the model with sparsity introduced
    """
    import torch
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name:
                param.data = param.mul_(torch.rand_like(param) > sparsity).float()
    return model

def _test_models():
    from tests.sparseml.pytorch.helpers import LinearNet
    from tests.sparseml.pytorch.helpers import ConvNet
    return [
        LinearNet(), 
        # _induce_sparsity(LinearNet()), 
        # ConvNet(), 
        # _induce_sparsity(ConvNet()),
        ]

def _test_optims():
    import torch
    return [
        torch.optim.Adam, 
        # torch.optim.SGD
        ]

@pytest.mark.parametrize("model", _test_models())
@pytest.mark.parametrize("optimizer", _test_optims())
def test_modifier_e2e(model, optimizer):
    expected_sparsities = {name: tensor_sparsity(param.data) for name, param in model.named_parameters() if "weight" in name}
    
    # init modifier with model
    from sparseml.modifiers.pruning.constant.pytorch import ConstantPruningModifierPyTorch
    from sparseml.core import State
    
    state = State(framework=Framework.pytorch)
    optim = optimizer(model.parameters(), lr=0.001)
    state.update(model=model, optimizer=optim)
    
    
    modifier = ConstantPruningModifierPyTorch(model=model, start=0, end=1, targets=["re:*.weight"])
    modifier.initialize(state)
    
    sparsity_to_induce = 0.8
    model = _induce_sparsity(model, sparsity_to_induce)
    
    
    # assert sparsity has been messed up
    induced_sparsities = {name: tensor_sparsity(param.data) for name, param in model.named_parameters() if "weight" in name}
    assert induced_sparsities != expected_sparsities, "Sparsity mess up failed"
    
    # apply modifier
    
    modifier.finalize(state)
    
    # assert sparsity has been restored
    actual_sparsities = {name: tensor_sparsity(param.data) for name, param in model.named_parameters() if "weight" in name}
    assert actual_sparsities == expected_sparsities, "Sparsity was not constant"
    
    
    
    