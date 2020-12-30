"""
Helper functions for performing quantization aware training with PyTorch
"""

from copy import deepcopy
from onnx import (
    ModelProto,
    NodeProto,
    numpy_helper,
)
import torch
from torch.nn import (
    Module,
    Conv2d,
    BatchNorm2d,
    ReLU,
)
from typing import NamedTuple

try:
    from torch import quantization as torch_quantization
except:
    torch_quantization = None

from sparseml.onnx.utils import (
    get_node_output_nodes,
    get_nodes_by_output_id,
    get_node_params,
    get_init_by_name,
    update_model_param,
    remove_node_and_params_from_graph,
    swap_node_output,
)
from sparseml.pytorch.nn import ReLU as ReLU_nm

__all__ = [
    "add_quant_dequant",
    "get_qat_qconfig",
    "fuse_module_conv_bn_relus",
]


def add_quant_dequant(module):
    """
    Wraps all Conv and Linear submodule with a qconfig with a QuantWrapper
    :param module: the module to modify
    """
    module_type = str(type(module)).split(".")[-1].lower()
    is_quantizable_module = "conv" in module_type or "linear" in module_type

    if is_quantizable_module and hasattr(module, "qconfig") and module.qconfig:
        return torch_quantization.QuantWrapper(module)

    for name, child in module.named_children():
        setattr(module, name, add_quant_dequant(child))

    return module


def get_qat_qconfig() -> torch_quantization.QConfig:
    """
    :return: A QAT fake quantization config for symmetric weight quantization and
        asymmetric activation quantization.  The difference between this and
        torch.quantization.default_qat_qconfig is that the activation observer
        will not have reduce_range enabled.
    """
    activation_observer = torch_quantization.FakeQuantize.with_args(
        observer=torch_quantization.MovingAverageMinMaxObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
    )
    weight_observer = torch_quantization.default_weight_fake_quant
    return torch_quantization.QConfig(
        activation=activation_observer, weight=weight_observer,
    )


def fuse_module_conv_bn_relus(module: Module, inplace: bool = True) -> Module:
    """
    Performs fusion of Conv2d, BatchNorm2d, and ReLU layers found in the
    given module. To be fused, these layers must appear sequentially in
    module.named_modules() and be in the same submodule.
    Fuses either Conv2d -> BatchNorm2d or Conv2d -> BatchNorm2d -> ReLU blocks

    If this function does not fuse the model in the desired way, implement an
    in place fusing function for the model.

    :param module: the module to fuse
    :param inplace: set True to perform fusions in-place. default is True
    :return: the fused module
    """
    if torch_quantization is None:
        raise RuntimeError(
            "Unable to import package torch.quantization. "
            "Try upgrading your PyTorch version."
        )
    if not inplace:
        module = deepcopy(module)
    conv_blocks = []
    current_block = []
    current_block_submodule_name = ""
    for name, layer in module.named_modules():
        submodule_name = ".".join(name.split(".")[:-1])
        if (
            len(current_block) == 1  # [Conv2d]
            and isinstance(layer, BatchNorm2d)
            and submodule_name == current_block_submodule_name
        ) or (
            len(current_block) == 2  # [Conv2d, BatchNorm2d]
            and isinstance(layer, ReLU)
            and submodule_name == current_block_submodule_name
        ):
            if isinstance(layer, ReLU_nm):
                _replace_nm_relu(module, name, layer)
            current_block.append(name)
        else:
            if current_block:
                if len(current_block) > 1:  # cannot fuse single module
                    conv_blocks.append(current_block)
                current_block = []
                current_block_submodule_name = ""
            if isinstance(layer, Conv2d):
                current_block.append(name)
                current_block_submodule_name = submodule_name
    if conv_blocks:
        torch_quantization.fuse_modules(module, conv_blocks, inplace=True)
    return module


def _replace_nm_relu(root_module, relu_path, nm_relu):
    current_module = root_module
    relu_path = relu_path.split(".")
    for sub_module in relu_path[:-1]:
        current_module = getattr(current_module, sub_module)
    new_relu = ReLU(inplace=nm_relu.inplace)
    setattr(current_module, relu_path[-1], new_relu)
