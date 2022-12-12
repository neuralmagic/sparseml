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
Schemas and types to support quantization
"""
from copy import deepcopy
from functools import partial
from typing import Any, Dict, Optional, Union

import torch
from packaging import version
from pydantic import BaseModel, Field
from torch.nn import Identity


try:
    from torch import quantization as torch_quantization
except Exception:
    torch_quantization = None


__all__ = [
    "DictQuantizationArgs",
    "DictQuantizationScheme",
    "QuantizationArgs",
    "QuantizationScheme",
    "QuantizationSchemeLoadable",
    "compute_range",
    "get_observer",
]


_PARSED_TORCH_VERSION = version.parse(torch.__version__)
_TORCH_PRE_112 = _PARSED_TORCH_VERSION < version.parse("1.12.0")


"""
Type definition aliases for defining QuantizationArgs and QuantizationScheme
as dictionaries for YAML serialization
"""
DictQuantizationArgs = Dict[str, Union[int, bool, Dict[str, Any]]]
DictQuantizationScheme = Dict[str, DictQuantizationArgs]

"""
Type definition for a type that is valid for loading a QuantizationScheme
using QuantizationScheme.load
"""
QuantizationSchemeLoadable = Union[
    "QuantizationScheme",
    DictQuantizationScheme,
    str,
    None,
]


class QuantizationArgs(BaseModel):
    """
    Class representing user facing arguments to define quantization Observers of
    activations or weights in a network
    """

    num_bits: int = Field(
        default=8, description="number of bits to target for quantization"
    )
    symmetric: bool = Field(
        default=False,
        description="set True to use symmetric quantization. Default False",
    )
    kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "optional dict of kwargs to be passed directly to torch quantization "
            "Observers constructor excluding quantization range or symmetry"
        ),
    )

    @classmethod
    def default_activation_args(cls):
        """
        :return: default 8 bits asymmetric settings
        """
        return cls(num_bits=8, symmetric=False)

    @classmethod
    def default_weight_args(cls):
        """
        :return: default 8 bits symmetric settings
        """
        return cls(num_bits=8, symmetric=True)

    def get_observer(self) -> "torch.quantization.FakeQuantize":
        """
        :return: torch quantization FakeQuantize built based on these QuantizationArgs
        """
        return get_observer(
            symmetric=self.symmetric,
            dtype=torch.qint8,
            bits=self.num_bits,
            reduce_range=self.kwargs.get("reduce_range", False),
            qconfig_kwargs=self.kwargs,
        )


class QuantizationScheme(BaseModel):
    """
    Class composed of QuantizationArgs to build QConfig and QuantWrapper objects for
    quantizing models. Provides a simple user interface for defining how inputs,
    weights, and outputs should be quantized
    """

    def __init__(self, *args, **kwargs):
        # support for loading from yaml str
        args = [arg if arg != "null" else None for arg in args]
        for key, val in kwargs.items():
            if val == "null":
                kwargs[key] = None
        super().__init__(*args, **kwargs)

    input_activations: Optional[QuantizationArgs] = Field(
        default_factory=QuantizationArgs.default_activation_args,
        description=(
            "target quantization setting for input activations. Set to None to "
            "not quantize input activations. Default is 8 bits asymmetric"
        ),
    )
    weights: Optional[QuantizationArgs] = Field(
        default_factory=QuantizationArgs.default_weight_args,
        description=(
            "target quantization setting for model weights. Set to None to "
            "not quantize weights. Default is 8 bits symmetric"
        ),
    )
    output_activations: Optional[QuantizationArgs] = Field(
        default=None,
        description=(
            "target quantization setting for output activations. Set to None to "
            "not quantize output activations. Default is None"
        ),
    )
    target_hardware: Optional[str] = Field(
        default=None,
        description=(
            "target deployment runtime/hardware name to be set by default "
            "classmethods. Default is None"
        ),
    )

    @classmethod
    def load(
        cls,
        scheme: QuantizationSchemeLoadable,
        default: Optional["QuantizationScheme"] = None,
    ) -> "QuantizationScheme":
        """
        :param scheme: QuantizationScheme, dict representation of scheme,
            or string alias of a scheme to load. Valid strings:
            ['default', 'deepsparse', 'tensorrt']
        :param default: default QuantizationScheme to override 'default' scheme
            with
        :return: constructed QuantizationScheme object from the given scheme;
            if given a dict, returns QuantizationScheme.parse_obj(scheme), string
            input will return the defualt QuantizationScheme if set to 'default'.
        """
        if isinstance(scheme, cls):
            return scheme
        elif scheme is None or scheme == "default":
            # if no default override, defaults to QuantizationScheme()
            return deepcopy(default) or cls()
        elif isinstance(scheme, str):
            if scheme == "deepsparse":
                return cls.deepsparse()
            elif scheme == "tensorrt":
                return cls.tensorrt()
            raise ValueError(
                f"Unrecognized QuantizationScheme string alias {scheme}. "
                "Valid strings: ['default', 'deepsparse', 'tensorrt']"
            )
        elif isinstance(scheme, dict):
            # default to dict
            scheme = {key: _parse_quantization_arg(arg) for key, arg in scheme.items()}
            return cls.parse_obj(scheme)
        else:
            raise ValueError(
                f"Unrecognized type {type(scheme)} for QuantizationScheme.load, "
                "expected one of: [QuantizationScheme, Dict, str, None]"
            )

    @classmethod
    def deepsparse(cls) -> "QuantizationScheme":
        """
        :return: QuantizationScheme for deepsparse targeted deployments -
            int8, symmetric weights, asymmetric inputs, no output quantization
        """
        return cls(
            input_activations=QuantizationArgs(num_bits=8, symmetric=False),
            weights=QuantizationArgs(num_bits=8, symmetric=True),
            output_activations=None,
            target_hardware="deepsparse",
        )

    @classmethod
    def tensorrt(cls) -> "QuantizationScheme":
        """
        :return: QuantizationScheme for tensorrt targeted deployments -
            compatibility with explict quantization as supported by TensorRT 8.2:
            int8, symmetric for both weights and inputs, no output quantization
        """
        return cls(
            input_activations=QuantizationArgs(num_bits=8, symmetric=True),
            weights=QuantizationArgs(num_bits=8, symmetric=True),
            output_activations=None,
            target_hardware="tensorrt",
        )

    def get_qconfig(self) -> "torch.quantization.QConfig":
        """
        :return: QConfig for Modules (output activations used,
            use QuantWrapper for inputs)
        """
        qconfig = _get_qconfig(self.output_activations, self.weights)
        # add reference to this quantization scheme for reference
        qconfig.quantization_scheme = self
        return qconfig

    def get_wrapper_qconfig(self) -> "torch.quantization.QConfig":
        """
        :return: QConfig for QuantWrapper objects (input activations used)
        """
        qconfig = _get_qconfig(self.input_activations, None)
        # add reference to this quantization scheme for reference
        qconfig.quantization_scheme = self
        return qconfig

    def __str__(self) -> str:
        """
        :return: YAML friendly string serialization
        """
        dict_repr = self.dict()
        dict_repr = {
            key: val if val is not None else "null" for key, val in dict_repr.items()
        }
        return str(dict_repr)


def compute_range(dtype: torch.dtype, bits: int):
    """
    compute quantization limits depending on data type and number of bits

    :param dtype: data type.
    :param bits: number of bits.
    :return: minimum limit, maximum limit, whether the range is customized
    """
    bits = bits if bits else 8
    is_custom = bits != 8
    if dtype == torch.qint8:
        quant_min = -(2 ** (bits - 1))
        quant_max = (2 ** (bits - 1)) - 1
    elif dtype == torch.quint8:
        quant_min = 0
        quant_max = (2 ** bits) - 1

    return quant_min, quant_max, is_custom


def get_observer(
    symmetric: bool,
    dtype: torch.dtype,
    bits: int,
    reduce_range: bool,
    qconfig_kwargs: Dict[str, Any],
):
    qscheme = torch.per_tensor_symmetric if symmetric else torch.per_tensor_affine
    quant_min, quant_max, is_custom_qrange = compute_range(dtype, bits)

    observer_cls = torch_quantization.MovingAverageMinMaxObserver
    observer_kwargs = dict(
        dtype=dtype,
        qscheme=qscheme,
        reduce_range=reduce_range,
    )

    """
    in torch 1.9.1, quant_min and quant_max are not passed to observer:
    https://github.com/pytorch/pytorch/blob/v1.9.1/torch/quantization/fake_quantize.py#L109
    however in 1.12.0, this is fixed so both are passed to observer:
    https://github.com/pytorch/pytorch/blob/v1.12.1/torch/ao/quantization/fake_quantize.py#L132

    Passing quant_min/quant_max to observer means the observer will have
    `self.has_customized_qrange == True` in both 1.9.1 and 1.12.0.

    For whatever reason, both versions calculate zero point for
    quint8 differently **if there is a customized_qrange**
    1. customized qrange has zero point of 127
    2. non-customized has zero point of 128.
    source:
    https://github.com/pytorch/pytorch/blob/v1.12.1/torch/ao/quantization/observer.py#L293

    **we want to ensure that the zero point is 128**
    see https://github.com/neuralmagic/sparseml/pull/604
    """
    if is_custom_qrange:
        # for both versions we need to include the custom min/max values in kwargs
        observer_kwargs["quant_min"] = quant_min
        observer_kwargs["quant_max"] = quant_max
        if _TORCH_PRE_112:
            # pre 1.12, the observer doesn't get passed the quant_min/quant_max values,
            # so we patch them in to the constructor of the observer
            observer_cls = partial(
                observer_cls, quant_min=quant_min, quant_max=quant_max
            )
    else:
        # if using a non custom qrange, we can rely on default values used by
        # the observers
        if _TORCH_PRE_112:
            # pre 1.12, the observer doesn't get passed the quant_min/quant_max values,
            # so we are safe to pass these to FakeQuantize
            observer_kwargs["quant_min"] = quant_min
            observer_kwargs["quant_max"] = quant_max
        else:
            # post 1.12 we cannot pass them to the observer since that will set
            # has_customized_qrange. instead we rely on the default values
            # being equal to the `quant_min` and `quant_max` here.
            pass

    observer_kwargs["observer"] = observer_cls
    observer_kwargs.update(qconfig_kwargs or {})
    observer = torch_quantization.FakeQuantize.with_args(
        **observer_kwargs,
    )

    return observer


def _get_qconfig(
    activation_args: Optional[QuantizationArgs], weight_args: Optional[QuantizationArgs]
) -> "torch.quantization.QConfig":
    return torch_quantization.QConfig(
        activation=activation_args.get_observer() if activation_args else Identity,
        weight=weight_args.get_observer() if weight_args else Identity,
    )


def _parse_quantization_arg(arg: Any):
    if arg == "None":
        return None
    return arg
