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
from typing import Any, Callable, List, Union

import torch
from torch.nn import BatchNorm2d, Conv2d, Embedding, Module, ReLU


try:
    import torch.nn.intrinsic as nni
    from torch import quantization as torch_quantization
except Exception:
    nni = None
    torch_quantization = None

from sparseml.pytorch.nn import ReLU as ReLU_nm


__all__ = [
    "QATWrapper",
    "configure_module_qat_wrappers",
    "configure_module_default_qconfigs",
    "add_quant_dequant",
    "get_qat_qconfig",
    "fuse_module_conv_bn_relus",
    "prepare_embeddings_qat",
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
    """

    @staticmethod
    def from_module(module: Module) -> "QATWrapper":
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

        return QATWrapper(forward_fn=module, **qat_wrapper_kwargs)

    def __init__(
        self,
        forward_fn: Callable[[Any], Any],
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
        self.input_qconfigs = self._load_qconfigs(
            "input_qconfigs", num_input_quant_stubs, input_qconfigs
        )
        self.output_qconfigs = self._load_qconfigs(
            "output_qconfigs", num_outputs, output_qconfigs
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

        for quant_stub, qconfig in zip(self.output_quant_stubs, self.output_qconfigs):
            quant_stub.qconfig = qconfig

    @staticmethod
    def _load_qconfigs(
        name: str,
        expected_len: int,
        qconfigs: Union["QConfig", str, List["QConfig"]],  # noqa: F821
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

            qconfigs[idx] = get_qat_qconfig(
                symmetric_activations=(qconfig == "symmetric")
            )

        return qconfigs


def configure_module_qat_wrappers(module: Module):
    """
    if any submodule of the given module has the attribute wrap_qat == True,
    then it will be replaced by a QATWrapper of it created by QATWrapper.from_module.
    Other named kwargs to the QATWrapper constructor must be contained in a dictionary
    under an attributed named `qat_wrapper_kwargs`

    :param module: module to potentially wrap the submodules of
    """
    # wrap any children of the given module as a QATWrapper if required
    for child_name, child_module in module.named_children():
        if hasattr(child_module, "wrap_qat") and child_module.wrap_qat:
            setattr(module, child_name, QATWrapper.from_module(child_module))
        # recurse on child module
        configure_module_qat_wrappers(child_module)


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


def get_qat_qconfig(
    symmetric_activations: bool = False,
    symmetric_weights: bool = True,
) -> "torch.quantization.QConfig":
    """
    :param symmetric_activations: if True, activations will have a symmetric
        UINT8 quantization range with zero point set to 128. Otherwise activations
        will use asymmetric quantization with any zero point. Default is False
    :param symmetric_weights: if True, weights will have a symmetric
        INT8 quantization range with zero point set to 0. Otherwise activations
        will use asymmetric quantization with any zero point. Default is True
    :return: A QAT fake quantization config for symmetric weight quantization and
        asymmetric activation quantization.  The difference between this and
        torch.quantization.default_qat_qconfig is that the activation observer
        will not have reduce_range enabled.
    """
    activation_qscheme = (
        torch.per_tensor_symmetric if symmetric_activations else torch.per_tensor_affine
    )
    activation_observer = torch_quantization.FakeQuantize.with_args(
        observer=torch_quantization.MovingAverageMinMaxObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=activation_qscheme,
        reduce_range=False,
    )
    weight_qscheme = (
        torch.per_tensor_symmetric if symmetric_weights else torch.per_tensor_affine
    )
    weight_observer = torch_quantization.FakeQuantize.with_args(
        observer=torch_quantization.MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=weight_qscheme,
        reduce_range=False,
    )
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


def prepare_embeddings_qat(
    module: Module,
    qconfig: "torch.quantization.QConfig" = None,
):
    """
    adds a fake quantize call to the weights of any Embedding modules in the given
    module

    :param module: module to run QAT for the embeddings of
    :param qconfig: qconfig to generate the fake quantize ops from. Default uses INT8
        asymmetric range
    """
    if qconfig is None:
        qconfig = get_qat_qconfig(symmetric_weights=False)
    for submodule in module.modules():
        if type(submodule) is Embedding:
            _prepare_qat_embedding(submodule, qconfig)


def _prepare_qat_embedding(embedding: Module, qconfig: "torch.quantization.QConfig"):
    embedding.weight_fake_quant = qconfig.weight()

    def _qat_forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(
            input,
            self.weight_fake_quant(self.weight),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    # bind qat forward to embedding
    qat_forward_bound = _qat_forward.__get__(embedding, embedding.__class__)
    setattr(embedding, "forward", qat_forward_bound)


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
