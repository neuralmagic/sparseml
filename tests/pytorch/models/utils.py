import sys
from torch.nn import Module


__all__ = ["compare_model"]


def compare_model(model_one: Module, model_two: Module, same: bool):
    total_dif = 0.0

    for param_one, param_two in zip(model_one.parameters(), model_two.parameters()):
        assert param_one.data.shape == param_two.data.shape
        total_dif += (param_one.data - param_two.data).abs().sum()

    if same:
        assert total_dif < sys.float_info.epsilon
    else:
        assert total_dif > sys.float_info.epsilon
