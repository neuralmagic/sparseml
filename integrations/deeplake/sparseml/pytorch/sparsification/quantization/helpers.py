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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.intrinsic as nni
from packaging import version
from torch import quantization as torch_quantization
from torch.nn import BatchNorm2d, Conv2d, Embedding, Module, ReLU

from sparseml.pytorch.nn import ReLU as ReLU_nm
from sparseml.pytorch.sparsification.quantization.quantization_scheme import (
    QuantizationArgs,
    QuantizationScheme,
    get_observer,
)
from sparseml.pytorch.utils import get_layer


_PARSED_TORCH_VERSION = version.parse(torch.__version__)

__all__ = [
    "QATWrapper",
    "configure_module_bn_wrappers",
    "configure_module_default_qconfigs",
    "configure_module_qat_wrappers",
    "add_quant_dequant",
    "remove_activation_qat_by_layer_name",
    "get_qat_qconfig",
    "freeze_bn_stats",
    "fuse_module_conv_bn_relus",
    "prepare_embeddings_qat",
    "QConfigProperties",
    "LINEAR_ACTIVATION_NAMES",
    "CONV_ACTIVATION_NAMES",
]

LINEAR_ACTIVATION_NAMES = ["Linear", "LinearReLU"]
CONV_ACTIVATION_NAMES = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvBn1d",
    "ConvBn2d",
    "ConvBn3d",
    "ConvReLU1d",
    "ConvReLU2d",
    "ConvReLU3d",
    "ConvBnReLU1d",
    "ConvBnReLU2d",
    "ConvBnReLU3d",
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

_FUSED_MODULE_TYPES = (
    (
        # Conv based layers
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
        nni.LinearReLU,
    )
    if nni  # nni will always import if torch.quantization is available
    else tuple()
)


@dataclass
class QConfigProperties:
    """
    Dataclass that stores properties needed to define qconfig objects.
    Default values set here.

    :param symmetric_activations: if True, activations will have a symmetric
        quantization range with a pre-specified zero point
        (0 if activation_dtype=torch.qint8, 128 if activation_dtype=torch.quint8).
        Default is False.
    :param symmetric_weights: if True, weights will have a symmetric
        quantization range with a pre-specified zero point
        (0 if weight_dtype=torch.qint8, 128 if weight_dtype=torch.quint8).
        Default is True.
    :param reduce_range: if True, the quantization range will be reduced by one bit.
        This may prevent overflow issues with model execution on certain hardware.
        Default is False.
    :param activation_qconfig_kwargs: Additional kwargs for quantization of
        activations.
    :param weight_qconfig_kwargs: Additional kwargs for quantization of
        weights.
    :param activation_dtype: quantized activation data type.
        Default is torch.quint8.
    :param weight_dtype: quantized weights data type.
        Default is torch.qint8.
    :param activation_bits: number of bits for activations. Default is 8.
    :param weight_bits: number of bits for weights. Default is 8.
    :param tensorrt: if True sets quantization configuration for compatibility with
       explict quantization as supported by TensorRT 8.2.
    """

    _symmetric_activations: bool = False
    _symmetric_weights: bool = True
    reduce_range: bool = False
    activation_dtype: torch.dtype = torch.quint8
    weight_dtype: torch.dtype = torch.qint8
    activation_bits: int = 8
    weight_bits: int = 8
    activation_qconfig_kwargs: Dict[str, Any] = field(default_factory=dict)
    weight_qconfig_kwargs: Dict[str, Any] = field(default_factory=dict)
    tensorrt: bool = False

    @property
    def symmetric_activations(self) -> bool:
        # always use symmetric activations in tensorrt mode
        return self.tensorrt or self._symmetric_activations

    @symmetric_activations.setter
    def symmetric_activations(self, value: bool):
        self._symmetric_activations = value

    @property
    def symmetric_weights(self) -> bool:
        return self.tensorrt or self._symmetric_weights

    @symmetric_weights.setter
    def symmetric_weights(self, value: bool):
        self._symmetric_weights = value


class QATWrapper(Module):
    """
    Wraps inputs and outputs of a Module or function with QuantStubs for
    Quantization-Aware-Training (QAT)

    :param forward_fn: function to be wrapped, should generally accept and return
        torch Tensor(s)
    :param num_inputs: number of inputs of the forward function to add a QuantStub
        to. Will wrap the first num_inputs ordered inputs of the function. Default
        is 1
    :param kwarg_input_names: list of names of key word arguments to the forward pass
        that should be wrapped with a fake quantize operation. Defaults to empty
    :param num_outputs: number of outputs of the forward function to add a QuantStub
        to. Will wrap the first num_inputs ordered outputs of the function. Default
        is 1. Will also add a DeQuantStub for FP32 conversion if
        torch.quantization.convert is invoked
    :param input_qconfigs: QConfig to use for calibrating the input QuantStubs. Can
        be a single QConfig that will be copied to each QuantStub or a list of one
        QConfig for each input. Instead of a QConfig objects, the string 'asymmetric'
        or 'symmetric' may be used to use default UINT8 asymmetric and symmetric
        quantization respectively
    :param output_qconfigs: QConfig to use for calibrating the output QuantStubs. Can
        be a single QConfig that will be copied to each QuantStub or a list of one
        QConfig for each output. Instead of a QConfig objects, the string 'asymmetric'
        or 'symmetric' may be used to use default UINT8 asymmetric and symmetric
        quantization respectively
    :param qproperties: properties used to define QConfig. may also be a quantization
        scheme
    """

    @staticmethod
    def from_module(
        module: Module,
        qproperties: Union[QConfigProperties, QuantizationScheme],
    ) -> "QATWrapper":
        """
        :param module: torch Module to create a QATWrapper for
        :return: QATWrapper object created using the given Module as the forward
            function. Will attempt to find any other named parameter of the QATWrapper
            constructor from the attributes of the given Module
        """
        qat_wrapper_kwargs = (
            module.qat_wrapper_kwargs or {}
            if hasattr(module, "qat_wrapper_kwargs")
            else {}
        )

        # Remove qconfig from wrapped layer to avoid duplicate quantization
        module.qconfig = None
        return QATWrapper(
            forward_fn=module, qproperties=qproperties, **qat_wrapper_kwargs
        )

    def __init__(
        self,
        forward_fn: Callable[[Any], Any],
        qproperties: Union[QConfigProperties, QuantizationScheme],
        num_inputs: int = 1,
        kwarg_input_names: List[str] = None,
        num_outputs: int = 1,
        input_qconfigs: Union[
            "torch.quantization.QConfig", str, List["torch.quantization.QConfig"]
        ] = "asymmetric",
        output_qconfigs: Union[
            "torch.quantization.QConfig", str, List["torch.quantization.QConfig"]
        ] = "asymmetric",
    ):
        super().__init__()

        if torch_quantization is None:
            raise RuntimeError(
                "Unable to import package torch.quantization. "
                "Try upgrading your PyTorch version to >= 1.7.0."
            )

        if not callable(forward_fn):
            raise ValueError(
                "forward_fn of QATWrapper must be callable. "
                f"Received {type(forward_fn)}"
            )

        self.kwarg_input_names = kwarg_input_names or []
        num_input_quant_stubs = num_inputs + len(self.kwarg_input_names)

        self.forward_fn = forward_fn
        # Add weight qconfig to forward_fn (in case it has weights)
        qconfig_ = (
            get_qat_qconfig(qproperties)
            if isinstance(qproperties, QConfigProperties)
            else qproperties.get_qconfig()  # QuantizationScheme
        )
        qconfig = torch_quantization.QConfig(
            activation=torch.nn.Identity,
            weight=qconfig_.weight,
        )
        self.forward_fn.qconfig = qconfig

        self.input_qconfigs = self._load_qconfigs(
            name="input_qconfigs",
            expected_len=num_input_quant_stubs,
            qconfigs=input_qconfigs,
            qproperties=qproperties,
        )
        self.output_qconfigs = self._load_qconfigs(
            name="output_qconfigs",
            expected_len=num_outputs,
            qconfigs=output_qconfigs,
            qproperties=qproperties,
        )

        self.input_quant_stubs = torch.nn.ModuleList(
            [torch_quantization.QuantStub() for _ in range(num_input_quant_stubs)]
        )
        self.output_quant_stubs = torch.nn.ModuleList(
            [torch_quantization.QuantStub() for _ in range(num_outputs)]
        )
        self.output_dequant_stubs = torch.nn.ModuleList(
            [torch_quantization.DeQuantStub() for _ in range(num_outputs)]
        )

    def forward(self, *args, **kwargs) -> Any:
        """
        :param args: arguments to forward function; the first num_inputs of these args
            will be wrapped by a QuantStub
        :param kwargs: key word arguments to pass to the wrapped forward function
        :return: outputs of the forward function with a QuantStub applied to the first
            num_outputs outputs
        """

        if any(kwarg not in kwargs for kwarg in self.kwarg_input_names):
            raise ValueError(
                f"QATWrapper expected kwargs {self.kwarg_input_names} to be included "
                f"in forward function kwargs. Found {list(kwargs.keys())}. missing "
                f"{[kwarg for kwarg in self.kwarg_input_names if kwarg not in kwargs]}"
            )

        qat_args = []

        # fake quantize positional arguments
        num_args_stubs = len(self.input_quant_stubs) - len(self.kwarg_input_names)
        for idx, arg in enumerate(args):
            if idx < num_args_stubs:
                arg = self.input_quant_stubs[idx](arg)
            qat_args.append(arg)

        # fake quantize key word arguments
        for idx, kwarg in enumerate(self.kwarg_input_names):
            kwargs[kwarg] = self.input_quant_stubs[num_args_stubs + idx](kwargs[kwarg])

        # wrapped forward pass
        outputs = self.forward_fn(*qat_args, **kwargs)

        if len(self.output_quant_stubs) == 0:
            # no output wrapping
            return outputs

        if isinstance(outputs, torch.Tensor):
            if len(self.output_quant_stubs) > 1:
                raise ValueError(
                    f"QATWrapper expected {len(self.output_quant_stubs)} outputs in "
                    "forward pass. Found one output"
                )
            # output is a single Tensor
            qat_output = self.output_quant_stubs[0](outputs)
            return self.output_dequant_stubs[0](qat_output)

        qat_outputs = []

        for idx, output in enumerate(outputs):
            if idx < len(self.output_quant_stubs):
                output = self.output_quant_stubs[idx](output)
                output = self._output_deuant_stubs[idx](output)
            qat_outputs.append(output)

        return qat_outputs

    def configure_qconfig(self):
        """
        Sets the qconfigs of the quant stubs to the pre-initialized QConfigs
        """
        for quant_stub, qconfig in zip(self.input_quant_stubs, self.input_qconfigs):
            quant_stub.qconfig = qconfig
            if hasattr(qconfig, "quantization_stub"):
                quant_stub.quantization_stub = qconfig.quantization_stub

        for quant_stub, qconfig in zip(self.output_quant_stubs, self.output_qconfigs):
            quant_stub.qconfig = qconfig
            if hasattr(qconfig, "quantization_stub"):
                quant_stub.quantization_stub = qconfig.quantization_stub

    @staticmethod
    def _load_qconfigs(
        name: str,
        expected_len: int,
        qconfigs: Union["QConfig", str, List["QConfig"]],  # noqa: F821
        qproperties: QConfigProperties,
    ):
        if not isinstance(qconfigs, (str, torch_quantization.QConfig, List)):
            raise ValueError(
                f"QATWrapper {name} must be a string, torch.quantization.QConfig, "
                f"or a List of them. Received a {type(qconfigs)}"
            )

        if isinstance(qconfigs, (str, torch_quantization.QConfig)):
            qconfigs = [deepcopy(qconfigs) for _ in range(expected_len)]

        if len(qconfigs) != expected_len:
            raise ValueError(
                f"QATWrapper {name} should have exactly one qconfig or one for every "
                f"argument ({expected_len}). Given {len(qconfigs)}"
            )

        valid_qconfig_strs = ["asymmetric", "symmetric"]
        for idx, qconfig in enumerate(qconfigs):
            if not isinstance(qconfig, str):
                continue

            if qconfig not in valid_qconfig_strs:
                raise ValueError(
                    "QATWrapper qconfig names can either be "
                    "torch.quantization.QConfig objects or a string "
                    f"in {valid_qconfig_strs} that will be converted to a QConfig. "
                    f"Found string with value {qconfig} in {name}"
                )

            qconfig_idx = None
            if isinstance(qproperties, QConfigProperties):
                qproperties_idx = deepcopy(qproperties)
                qproperties_idx.symmetric_activations = qconfig == "symmetric"
                qconfig_idx = get_qat_qconfig(qproperties_idx)
            else:
                scheme_idx = deepcopy(qproperties)
                symmetric = qconfig == "symmetric"
                # always use output_activations of scheme because the activations
                # of the QuantStub() are the ones tracked
                if scheme_idx.output_activations is not None:
                    scheme_idx.input_activations.symmetric = symmetric
                else:
                    scheme_idx.output_activations = QuantizationArgs(
                        symmetric=symmetric
                    )
                qconfig_idx = scheme_idx.get_qconfig()
                qconfig_idx.quantization_scheme = scheme_idx

            qconfigs[idx] = qconfig_idx

        return qconfigs


def configure_module_bn_wrappers(module: Module):
    """
    Wrap any BatchNormalization modules that are not fused with convolutions
    with BNWrapper to enable freezing/unfreezing of BN statistics

    :param module: module to potentially wrap the submodules of
    """
    # wrap any children of the given module as a QATWrapper if required
    if not hasattr(module, "freeze_bn_stats"):
        for child_name, child_module in module.named_children():
            if type(child_module) in [
                torch.nn.BatchNorm1d,
                torch.nn.BatchNorm2d,
                torch.nn.BatchNorm3d,
            ]:
                setattr(module, child_name, _BNWrapper(child_module))
            # recurse on child module
            configure_module_bn_wrappers(child_module)


def configure_module_qat_wrappers(
    module: Module,
    qproperties: QConfigProperties,
):
    """
    if any submodule of the given module has the attribute wrap_qat == True,
    then it will be replaced by a QATWrapper of it created by QATWrapper.from_module.
    Other named kwargs to the QATWrapper constructor must be contained in a dictionary
    under an attributed named `qat_wrapper_kwargs`

    :param module: module to potentially wrap the submodules of
    :param qproperties: properties used to define QConfig.
    """
    # wrap any children of the given module as a QATWrapper if required
    for child_name, child_module in module.named_children():
        if hasattr(child_module, "wrap_qat") and child_module.wrap_qat:
            setattr(
                module,
                child_name,
                QATWrapper.from_module(
                    module=child_module,
                    qproperties=qproperties,
                ),
            )
        # recurse on child module
        configure_module_qat_wrappers(
            module=child_module,
            qproperties=qproperties,
        )


def configure_module_default_qconfigs(module: Module):
    """
    if any submodule of the given module has a configure_qconfig function,
    configure_qconfig will be called on that submodule to set the qconfig(s) of that
    module to its default

    :param module: module to set qconfigs for
    """
    for submodule in module.modules():
        if hasattr(submodule, "configure_qconfig") and callable(
            getattr(submodule, "configure_qconfig")
        ):
            submodule.configure_qconfig()


def add_quant_dequant(
    module: torch.nn.Module, name=None, parent_module=None, layer_class_names=None
):
    """
    Wraps all Conv and Linear submodule with a qconfig with a QuantWrapper
    :param module: the module to modify
    :param name: name of the module to modify; default to None
    :param parent_module: parent module containing the module to modify; default to None
    :param layer_class_names: list of module class names to be added to the
        list of quantizable modules
    :return: the modified module
    """
    named_children = module.named_children()
    is_quantizable = type(module) in _QUANTIZABLE_MODULE_TYPES
    if layer_class_names:
        is_quantizable = (
            is_quantizable or module.__class__.__name__ in layer_class_names
        )
    if is_quantizable and hasattr(module, "qconfig") and module.qconfig:
        module = torch_quantization.QuantWrapper(module)
        if parent_module is not None and len(list(named_children)) <= 0:
            if "." in name:
                # unwrap name under parent module, nested through multiple submodules
                name_parts = name.split(".")
                for name_part in name_parts[:-1]:
                    parent_module = getattr(parent_module, name_part)
                name = name_parts[-1]

            # set parent module child to the newly wrapped module
            setattr(parent_module, name, module)
    else:
        for name, child in named_children:
            setattr(
                module,
                name,
                add_quant_dequant(child, layer_class_names=layer_class_names),
            )
    return module


def remove_activation_qat_by_layer_name(module: Module, layer_class_names: List[str]):
    """
    Disables fake quantization of activations for all submodules of the given module
    with class name layer_class_names

    :param module: module to remove activation fake quantization for certain layers
    :param layer_class_names: list of layer class names that should be affected.
        e.x. ["Linear"]
    """
    for submodule in module.modules():
        if submodule.__class__.__name__ in layer_class_names and hasattr(
            submodule, "qconfig"
        ):
            submodule.qconfig = torch_quantization.QConfig(
                activation=torch.nn.Identity,
                weight=submodule.qconfig.weight,
            )


def get_qat_qconfig(qproperties: QConfigProperties) -> "torch.quantization.QConfig":
    """
    :param qproperties: properties used to define QConfig.
    """
    activation_observer = get_observer(
        qproperties.symmetric_activations,
        qproperties.activation_dtype,
        qproperties.activation_bits,
        qproperties.reduce_range,
        qproperties.activation_qconfig_kwargs,
    )

    weight_observer = get_observer(
        qproperties.symmetric_weights,
        qproperties.weight_dtype,
        qproperties.weight_bits,
        False,
        qproperties.weight_qconfig_kwargs,
    )

    return torch_quantization.QConfig(
        activation=activation_observer,
        weight=weight_observer,
    )


def freeze_bn_stats(module: Module):
    if hasattr(module, "freeze_bn_stats"):
        module.freeze_bn_stats()


def fuse_module_conv_bn_relus(
    module: Module,
    inplace: bool = True,
    override_bn_subclasses_forward: Union[bool, str] = True,
) -> Module:
    """
    Performs fusion of Conv2d, BatchNorm2d, and ReLU layers found in the
    given module. To be fused, these layers must appear sequentially in
    module.named_modules() and be in the same submodule.
    Fuses either Conv2d -> BatchNorm2d, Conv2d -> ReLU, or
    Conv2d -> BatchNorm2d -> ReLU blocks

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
            len(current_block) in [1, 2]  # [Conv2d] or [Conv2d, BatchNorm2d]
            and isinstance(layer, ReLU)
            and not isinstance(current_block[-1], ReLU)
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
    if len(current_block) > 1:
        conv_blocks.append(current_block)
    if conv_blocks:
        # manually save and move hooks surrounding fused blocks
        # into new fused modules due to torch.quantization
        # error when a module has more than one hook
        block_hooks = _delete_get_block_hooks(module, conv_blocks)

        # run torch fusion
        if _PARSED_TORCH_VERSION < version.parse("1.10.0"):
            torch_quantization.fuse_modules(module, conv_blocks, inplace=True)
        else:
            if module.training:
                torch.ao.quantization.fuse_modules_qat(
                    module, conv_blocks, inplace=True
                )
            else:
                torch.ao.quantization.fuse_modules(module, conv_blocks, inplace=True)

        # add hooks back
        _add_fused_block_hooks(module, block_hooks)

    return module


def prepare_embeddings_qat(
    module: Module,
    qproperties: Optional[QConfigProperties] = None,
    qconfig: Optional["torch.quantization.QConfig"] = None,
):
    """
    adds a fake quantize call to the weights of any Embedding modules in the given
    module. The used qconfig will have a heirarchy of

    submodule.qconfig -> qconfig -> qproperties

    :param module: module to run QAT for the embeddings of
    :param qconfig: qconfig to generate the fake quantize ops from if qconfig
        not set in moduleDefault uses INT8 asymmetric range
    :param qproperties: properties used to define QConfig if qconfig not present
    """
    if qconfig is None and qproperties is not None:
        qproperties.symmetric_weights = False
        qconfig = get_qat_qconfig(qproperties)
    for submodule in module.modules():
        submodule_qconfig = getattr(submodule, "qconfig", None)
        submodule_qconfig = submodule_qconfig or qconfig
        if type(submodule) is Embedding and submodule_qconfig is not None:
            _prepare_qat_embedding(submodule, submodule_qconfig)


def _delete_get_block_hooks(
    module: Module,
    fuse_blocks: List[List[str]],
) -> List[Tuple[Any, Any]]:
    block_hooks = []
    for block in fuse_blocks:
        pre_hooks = []
        post_hooks = []

        for name in block:
            # get Module objects in block by their names
            m = get_layer(name, module)

            # extract the hooks
            pre_hooks.extend(m._forward_pre_hooks.values())
            post_hooks.extend(m._forward_hooks.values())

            # de-register the hooks from this module
            m._forward_pre_hooks.clear()
            m._forward_hooks.clear()

        block_hooks.append((pre_hooks, post_hooks))

    return block_hooks


def _add_fused_block_hooks(module: Module, block_hooks: List[Tuple[Any, Any]]):
    fused_modules = [
        mod for mod in module.modules() if isinstance(mod, _FUSED_MODULE_TYPES)
    ]

    if len(fused_modules) != len(block_hooks):
        raise RuntimeError(
            f"Number of fused modules ({len(fused_modules)}) after layer fusion in "
            f"module {module.__class__.__name__}. does not match expected "
            f"({len(block_hooks)}). Module may have already been fused or block "
            "skipped during torch.quantization.fuse_modules"
        )

    for fused_module, (pre_hooks, post_hooks) in zip(fused_modules, block_hooks):
        for pre_hook in pre_hooks:
            fused_module.register_forward_pre_hook(pre_hook)
        for post_hook in post_hooks:
            fused_module.register_forward_hook(post_hook)


def _prepare_qat_embedding(embedding: Module, qconfig: "torch.quantization.QConfig"):
    embedding.weight_fake_quant = qconfig.weight()

    def _qat_forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight_fake_quant(self.weight)
        if weight.device != input.device:
            # torch DataParallel may not pick up overwritten bound method
            # send weight to correct device
            weight = weight.to(input.device)

        return torch.nn.functional.embedding(
            input,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    # bind qat forward to embedding
    qat_forward_bound = _qat_forward.__get__(embedding, embedding.__class__)
    embedding.to(embedding.weight.device)  # set weight_fake_quant to correct device
    setattr(embedding, "forward", qat_forward_bound)


def _set_submodule(root_module: Module, sub_module_path, sub_module: Module):
    sub_module.training = root_module.training
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


class _BNWrapper(Module):
    """
    Wraps BatchNormalization module to expose methods needed to enable
    freezing/unfreezing of statistics

    :param module: BatchNormalization module to be wrapped
    """

    def __init__(self, module: Module):
        super().__init__()
        self.bn = module
        self.freeze_bn = False

    @property
    def running_mean(self):
        return self.bn.running_mean

    @running_mean.setter
    def running_mean(self, value):
        self.bn.running_mean = value

    @property
    def running_var(self):
        return self.bn.running_var

    @running_var.setter
    def running_var(self, value):
        self.bn.running_var = value

    @property
    def weight(self):
        return self.bn.weight

    @weight.setter
    def weight(self, value):
        self.bn.weight = value

    @property
    def bias(self):
        return self.bn.bias

    @bias.setter
    def bias(self, value):
        self.bn.bias = value

    @property
    def gamma(self):
        return self.bn.gamma

    @gamma.setter
    def gamma(self, value):
        self.bn.gamma = value

    @property
    def beta(self):
        return self.bn.beta

    @beta.setter
    def beta(self, value):
        self.bn.beta = value

    @property
    def num_batches_tracked(self):
        return self.bn.num_batches_tracked

    @num_batches_tracked.setter
    def num_batches_tracked(self, value):
        self.bn.num_batches_tracked = value

    @property
    def eps(self):
        return self.bn.eps

    @eps.setter
    def eps(self, value):
        self.bn.eps = value

    @property
    def momentum(self):
        return self.bn.momentum

    @momentum.setter
    def momentum(self, value):
        self.bn.momentum = value

    def forward(self, x):
        return self.bn(x)

    def freeze_bn_stats(self):
        self.freeze_bn = True
        self.bn.training = False
        return self

    def reset_running_stats(self):
        self.bn.reset_running_stats()

    def train(self, mode=True):
        if not self.freeze_bn:
            self.bn.train(mode)
        return self

    def update_bn_stats(self):
        self.freeze_bn = False
        self.bn.training = True
        return self
