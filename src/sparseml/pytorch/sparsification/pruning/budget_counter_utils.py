import torch
import numpy as np
import torch.nn as nn

from torch import Tensor
from torch.nn import Module


__all__ = [
    "get_nonzero_fraction",
    "get_param_counter",
    "get_flop_counter",
]


def get_nonzero_fraction(x: Tensor):
    return (torch.count_nonzero(x) / x.numel()).item()


# computed only for selected layers
@torch.no_grad()
def get_param_counter(layer_names: list[str], layers: list[Module]) -> dict:
    param_counts = {}
    for layer_name, layer in zip(layer_names, layers):
        if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            param_counts[layer_name] = layer.weight.numel()
        # do nothing for other layers
        else:
            pass
    return param_counts


# computed only for selected layers
@torch.no_grad()
def get_flop_counter(model: nn.Module, layer_names: list[str], layers: list[Module], sample_input: torch.Tensor) -> dict:
    flop_counts = {}
    hooks = {}

    def flop_counting_hook(layer_name):
        def _hook(layer, inp, out):
            # assuming input has the shape (B, *, C)
            if isinstance(layer, nn.Linear):
                flop_counts[layer_name] = np.prod(inp[0].shape[1:-1]) * layer.weight.numel()
            elif isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                flop_counts[layer_name] = np.prod(inp[0].shape[2:]) * layer.weight.numel() / \
                    np.prod(layer.stride)
        return _hook

    # init hooks
    for layer_name, layer in zip(layer_names, layers):
        if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            hooks[layer_name] = layer.register_forward_hook(flop_counting_hook(layer_name))
        # do nothing for other layers
        else:
            pass

    # make forward pass
    _ = model(sample_input)

    for _, hook in hooks.items():
        hook.remove()

    return flop_counts
