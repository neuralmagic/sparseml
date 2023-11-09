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
from typing import Any, Dict, Optional, Tuple, Union

import torch
from packaging import version
from pydantic import BaseModel, Field, validator
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
    strategy: str = Field(
        default="tensor",
        description=(
            "scope of the quantization to be applied."
            "Can be 'tensor', 'channel', or 'dynamic_token'"
        ),
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
            strategy=self.strategy,
            dtype=torch.qint8,
            bits=self.num_bits,
            reduce_range=self.kwargs.get("reduce_range", False),
            qconfig_kwargs=self.kwargs,
        )

    @validator("strategy")
    def validate_strategy(cls, value):
        valid_scopes = ["tensor", "channel"]
        if value not in valid_scopes:
            raise ValueError(f"`strategy` must be one of {valid_scopes}, got {value}")
        return value


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
        quant_max = (2**bits) - 1

    return quant_min, quant_max, is_custom


def get_observer(
    symmetric: bool,
    strategy: str,
    dtype: torch.dtype,
    bits: int,
    reduce_range: bool,
    qconfig_kwargs: Dict[str, Any],
):
    quant_min, quant_max, is_custom_qrange = compute_range(dtype, bits)

    if strategy == "channel":
        qscheme = torch.per_channel_symmetric if symmetric else torch.per_channel_affine
        observer_cls = torch_quantization.MovingAveragePerChannelMinMaxObserver
        fake_quantization_cls = torch_quantization.FakeQuantize
        fake_quantization_cls = torch_quantization.FakeQuantize
        observer_kwargs = dict(
            ch_axis=0,
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
        )
    elif strategy == "dynamic_token":
        observer_cls = PerTokenDynamicObserver
        fake_quantization_cls = DynamicFakeQuantize
        observer_kwargs = dict(
            ch_axis=-1,
            dtype=dtype,
            reduce_range=reduce_range,
            symmetric=symmetric,
        )
    else:  # default to tensor strategy
        qscheme = torch.per_tensor_symmetric if symmetric else torch.per_tensor_affine
        observer_cls = torch_quantization.MovingAverageMinMaxObserver
        fake_quantization_cls = torch_quantization.FakeQuantize
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
    observer = fake_quantization_cls.with_args(**observer_kwargs)

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


def _fake_quantize_per_token_affine(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    qmin: int,
    qmax: int,
):
    """
    Quantize tensor per token using scale and zero point.

    Args:
    :param x: Tensor to be quantized
    :param scale: Per-token quantization scale
    :param zero_point: Per-token zero point
    :param qmin: minimum quantized value
    :param qmax: maximum quantized value

    :return: Quantized tensor
    """
    scale = scale.to(x.dtype).to(x.device)
    zero_point = zero_point.to(x.dtype).to(x.device)
    q = torch.clip(torch.round((x / scale) + zero_point), min=qmin, max=qmax)
    return (q - zero_point) * scale


class PerTokenDynamicObserver(torch_quantization.ObserverBase):
    r"""
    Observer module for computing the quantization parameters for per token
    dynamic quantization. It uses current min/max values to determine quantization
    parameters. Differently from standard PyTorch observes, this observer does
    not create parameters since all computation depends on the current instance
    alone.

    :param ch_axis: Channel axis
    :param dtype: dtype of quantized tensors
    :param reduce_range: Boolean that determines whether to reduces the
        range of the quantized data type by 1 bit. Default is False.
    :param quant_min: Minimum quantized value.
        If unspecified, it will follow the 8-bit setup.
    :param quant_max: Maximum quantized value.
        If unspecified, it will follow the 8-bit setup.
    :param eps: Epsilon value for float32.
        Defaults to `torch.finfo(torch.float32).eps`.

    Given current min/max as :math:`x_\text{min}` and :math:`x_\text{max}`,
    scale :math:`s` and zero point :math:`z` are computed as:

    .. math::

        \begin{aligned}
            \text{if Symmetric:}&\\
            &s = 2 \max(|x_\text{min}|, x_\text{max}) /
                \left( Q_\text{max} - Q_\text{min} \right) \\
            &z = \begin{cases}
                0 & \text{if dtype is qint8} \\
                128 & \text{otherwise}
            \end{cases}\\
            \text{Otherwise:}&\\
                &s = \left( x_\text{max} - x_\text{min}  \right ) /
                    \left( Q_\text{max} - Q_\text{min} \right ) \\
                &z = Q_\text{min} - \text{round}(x_\text{min} / s)
        \end{aligned}

    where :math:`Q_\text{min}` and :math:`Q_\text{max}` are the minimum and
    maximum of the quantized data type.

    .. warning:: :attr:`dtype` can only take ``torch.qint8`` or ``torch.quint8``.

    .. note:: If the running minimum equals to the running maximum, the scale
              and zero_point are set to 1.0 and 0.
    """
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(
        self,
        ch_axis=-1,
        dtype=torch.quint8,
        symmetric=True,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        eps=torch.finfo(torch.float32).eps,
    ) -> None:
        super().__init__(dtype=dtype)
        self.ch_axis = ch_axis
        self.symmetric = symmetric
        self.reduce_range = reduce_range
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.eps = torch.tensor((eps,))
        self.min_val = None
        self.max_val = None
        if self.symmetric and self.reduce_range and self.dtype == torch.quint8:
            raise NotImplementedError(
                "Cannot reduce range for symmetric quantization for quint8"
            )
        self.has_customized_qrange = (quant_min is not None) and (quant_max is not None)
        if self.has_customized_qrange:
            assert (
                quant_min <= 0 <= quant_max
            ), "Used-specified quantization range must include 0."
            assert quant_min < quant_max, (
                "qmin must be strictly less than qmax for user-specified "
                "quantization range."
            )
        else:
            if dtype in [torch.qint8, torch.int8]:
                if reduce_range:
                    self.quant_min, self.quant_max = -64, 63
                else:
                    self.quant_min, self.quant_max = -128, 127
            elif dtype in [torch.quint8, torch.uint8]:
                if reduce_range:
                    self.quant_min, self.quant_max = 0, 127
                else:
                    self.quant_min, self.quant_max = 0, 255

    def forward(self, x_orig):
        """
        Records the per-token minimum and maximum values.

        :param x_orig: torch.Tensor

        :return: same tensor as input
        """
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        self.min_val, self.max_val = torch.aminmax(x, dim=self.ch_axis, keepdim=True)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the quantization parameters, given min and max
        value tensors. Works for both symmetric and asymmetric cases.

        :return: scales: Per-token scales tensor of shape
            (batch_size, tokens, 1)
        :return: zero_points: Per-token zero points tensor of shape
            (batch_size, tokens, 1)
        """
        if self.symmetric:
            max_val = torch.maximum(self.min_val.abs(), self.max_val.abs())
            scale = max_val / (float(self.quant_max - self.quant_min) / 2)
            scale = torch.maximum(scale, self.eps.to(scale.device))
            if self.dtype in [torch.quint8, torch.uint8]:
                if self.has_customized_qrange:
                    # When customized quantization range is used,
                    # down-rounded midpoint of the range is chosen.
                    zero_point = (self.quant_min + self.quant_max) // 2
                else:
                    zero_point = 128 * torch.ones_like(self.min_val, device=self.dtype)
        else:
            scale = (self.max_val - self.min_val) / float(
                self.quant_max - self.quant_min
            )
            zero_point = self.quant_min - torch.round(self.min_val / scale).to(
                torch.int
            )
            zero_point = torch.clamp(zero_point, self.quant_min, self.quant_max)

            # If scale is close to 0, set scale to min_val and zero point to zero
            zero_point = torch.where(
                scale > self.eps.to(scale.device),
                zero_point,
                torch.zeros_like(zero_point),
            )
            scale = torch.where(scale > self.eps.to(scale.device), scale, self.min_val)

        return scale, zero_point


class DynamicFakeQuantize(torch_quantization.FakeQuantizeBase):
    """
    Simulate the quantize and dequantize operations in training time.
    The output of this module is given by::

    x_out = (
        clamp(
            round(x/scale + zero_point), quant_min, quant_max)
        - zero_point
    ) * scale

    * :attr:`scale` defines the scale factor used for quantization.

    * :attr:`zero_point` specifies the quantized value to which 0
        in floating point maps to

    * :attr:`fake_quant_enabled` controls the application of fake
        quantization on tensors.

    * :attr:`dtype` specifies the quantized dtype that is being
        emulated with fake-quantization.
        Allowable values are torch.qint8 and torch.quint8.

    :param observer: Module for observing statistics on input tensors
        and calculating scale and zero-point.
    :param: observer_kwargs (optional): Arguments for the observer module

    Attributes:

        activation_post_process (Module): User provided module that collects
          statistics on the input tensor and provides a method to calculate
          scale and zero-point.

    """

    def __init__(
        self,
        observer=PerTokenDynamicObserver,
        quant_min=None,
        quant_max=None,
        **observer_kwargs,
    ):
        super().__init__()
        # Populate quant_min/quant_max to observer_kwargs if valid
        if quant_min is not None and quant_max is not None:
            assert (
                quant_min <= quant_max
            ), "quant_min must be less than or equal to quant_max"
            dtype = observer_kwargs.get("dtype", torch.quint8)
            if hasattr(observer, "p"):
                # In case observer is _PartialWrapper, dtype can be stored in
                # observer.p.keywords["dtype"]
                dtype = getattr(getattr(observer, "p", {}), "keywords", {}).get(
                    "dtype", dtype
                )
            assert torch.iinfo(dtype).min <= quant_min, "quant_min out of bound"
            assert quant_max <= torch.iinfo(dtype).max, "quant_max out of bound"
            observer_kwargs.update({"quant_min": quant_min, "quant_max": quant_max})
        self.activation_post_process = observer(**observer_kwargs)

        self.quant_min = self.activation_post_process.quant_min
        self.quant_max = self.activation_post_process.quant_max
        self.dtype = self.activation_post_process.dtype
        self.ch_axis = (
            self.activation_post_process.ch_axis
            if hasattr(self.activation_post_process, "ch_axis")
            else -1
        )

    @torch.jit.export
    def calculate_qparams(self):
        return self.activation_post_process.calculate_qparams()

    def forward(self, X):
        if self.fake_quant_enabled[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.calculate_qparams()
            X = _fake_quantize_per_token_affine(
                X,
                _scale,
                _zero_point,
                self.activation_post_process.quant_min,
                self.activation_post_process.quant_max,
            )
        return X

    @torch.jit.export
    def extra_repr(self):
        return (
            "fake_quant_enabled={}, "
            "quant_min={}, quant_max={}, dtype={}, ch_axis={}".format(
                self.fake_quant_enabled,
                self.activation_post_process.quant_min,
                self.activation_post_process.quant_max,
                self.dtype,
                self.ch_axis,
            )
        )
