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
Helper functions for performing quantization aware training with PyTorch
"""

from copy import deepcopy
from typing import Union

import torch
from torch.nn import BatchNorm2d, Conv2d, Module, ReLU


try:
    import torch.nn.intrinsic as nni
    from torch import quantization as torch_quantization
except Exception:
    nni = None
    torch_quantization = None

from sparseml.pytorch.nn import ReLU as ReLU_nm


__all__ = [
    "add_quant_dequant",
    "get_qat_qconfig",
    "fuse_module_conv_bn_relus",
]


_QUANTIZABLE_MODULE_TYPES = (
    {
        # Conv based layers
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        torch.nn.Conv3d,
        nni.ConvBn1d,
        nni.ConvBn2d,
        nni.ConvBn3d,
        nni.ConvReLU1d,
        nni.ConvReLU2d,
        nni.ConvReLU3d,
        nni.ConvBnReLU1d,
        nni.ConvBnReLU2d,
        nni.ConvBnReLU3d,
        # Linear Layers
        torch.nn.Linear,
        nni.LinearReLU,
    }
    if nni  # nni will always import if torch.quantization is available
    else None
)


def add_quant_dequant(module):
    """
    Wraps all Conv and Linear submodule with a qconfig with a QuantWrapper
    :param module: the module to modify
    """
    if (
        type(module) in _QUANTIZABLE_MODULE_TYPES
        and hasattr(module, "qconfig")
        and module.qconfig
    ):
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
        activation=activation_observer,
        weight=weight_observer,
    )


def fuse_module_conv_bn_relus(
    module: Module,
    inplace: bool = True,
    override_bn_subclasses_forward: Union[bool, str] = True,
) -> Module:
    """
    Performs fusion of Conv2d, BatchNorm2d, and ReLU layers found in the
    given module. To be fused, these layers must appear sequentially in
    module.named_modules() and be in the same submodule.
    Fuses either Conv2d -> BatchNorm2d or Conv2d -> BatchNorm2d -> ReLU blocks

    If this function does not fuse the model in the desired way, implement an
    in place fusing function for the model.

    :param module: the module to fuse
    :param inplace: set True to perform fusions in-place. default is True
    :param override_bn_subclasses_forward: if True, modules that are subclasses of
        BatchNorm2d will be modified to be BatchNorm2d but with the forward
        pass and state variables copied from the subclass. This is so these
        BN modules can pass PyTorch type checking when fusing. Can set to
        "override-only" and only parameters will be overwritten, not the
        forward pass. Default is True
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
                _set_submodule(module, name, ReLU(inplace=layer.inplace))
            if isinstance(layer, BatchNorm2d) and not type(layer) is BatchNorm2d:
                if not override_bn_subclasses_forward:
                    raise RuntimeError(
                        "Detected a Conv-BN block that uses a subclass of BatchNorm2d. "
                        "This will cause a type error when fusing with PyTorch, "
                        "set override_bn_subclasses_forward to True or 'override-only "
                        "to modify this BN subclass to be a BatchNorm2d object"
                    )
                # swap BN subclass with overwritten BN class that will pass torch
                # type checking
                overwritten_bn = _wrap_bn_sub_class(
                    layer,
                    override_forward=override_bn_subclasses_forward != "override-only",
                )
                _set_submodule(module, name, overwritten_bn),
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


def _set_submodule(root_module, sub_module_path, sub_module):
    current_module = root_module
    sub_module_path = sub_module_path.split(".")
    for child_module in sub_module_path[:-1]:
        current_module = getattr(current_module, child_module)
    setattr(current_module, sub_module_path[-1], sub_module)


def _wrap_bn_sub_class(bn_subclass, override_forward=True):
    batch_norm = BatchNorm2d(bn_subclass.num_features)
    batch_norm.__dict__ = bn_subclass.__dict__
    if override_forward:
        batch_norm.forward = bn_subclass.forward
    del bn_subclass
    return batch_norm
