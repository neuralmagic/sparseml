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

from typing import Any, Dict, List, Optional

from sparseml.core import Modifier, State
from sparseml.modifiers.quantization.utils.quantization_scheme import (
    QuantizationScheme,
    QuantizationSchemeLoadable,
)


__all__ = ["QuantizationModifier"]


class QuantizationModifier(Modifier):
    """
    Enables quantization aware training (QAT) for a given module or its submodules
    After the start epoch, the specified module(s) forward pass will emulate
    quantized execution and the modifier will be enabled until training is completed.

    | Sample yaml:
    |   QuantizationModifier:
    |       start: 0.0
    |       scheme:
    |           input_activations:
    |               num_bits: 8
    |               symmetric: False
    |           weights:
    |               num_bits: 8
    |               symmetric: True
    |       scheme_overrides:
    |           feature_extractor: "default"
    |           classifier:
    |               input_activations:
    |                   num_bits: 8
    |                   symmetric: False
    |               weights: null
    |           Conv2d:
    |               input_activations:
    |                   num_bits: 8
    |                   symmetric: True
    |       ignore: ["ReLU", "input"]
    |       disable_quantization_observer_epoch: 2.0
    |       freeze_bn_stats_epoch: 3.0
    |       model_fuse_fn_name: 'fuse_module'
    |       strict: True

    :param scheme: Default QuantizationScheme to use when enabling quantization
        in a module. May also be a dictionary to be loaded into the QuantizationScheme
        class. A string alias may also be used, supported aliases:
        ['default', 'deepsparse', 'tensorrt'].
        If None, the default scheme (`QuantizationScheme()`) will be used.
        Default is None
    :param scheme_overrides: optional mapping of module type names or submodule type
        names to quantization schemes to override them with. If a scheme is mapped to
        'default', then it will use the scheme set in the modifier scheme property
    :param ignore: optional list of module class names or submodule names
        to not quantize. Default is None
    :param disable_quantization_observer_epoch: Epoch to disable updates to the module
        quantization observers. At this point, quantized weights and zero points will
        not be updated. Leave None to not disable observers during QAT. Default is None
    :param freeze_bn_stats_epoch: Epoch to stop the tracking of batch norm stats. Leave
        None to not stop tracking batch norm stats during QAT. Default is None
    :param model_fuse_fn_name: Name of model function to fuse the model in place prior
        to performing QAT.  Set as None or 'no_fuse' to skip module fusing. Set as
         'conv_bv_relus' to use `sparseml.pytorch.utils.fuse_module_conv_bn_relus`.
        Default is None
    :param model_fuse_fn_kwargs: dictionary of keyword argument values to be passed
        to the model fusing function
    :param num_calibration_steps: Number of steps to run post training calibration for.
        When None, the entire calibration_dataloader is used
    :param strict: if True, will raise an error if any module types or submodules in
        scheme_overrides or ignore are not found in a given module. Default True
    """

    scheme: Optional[QuantizationSchemeLoadable] = None
    scheme_overrides: Optional[Dict[str, QuantizationSchemeLoadable]] = None
    ignore: Optional[List[str]] = None
    disable_quantization_observer_epoch: Optional[float] = None
    freeze_bn_stats_epoch: Optional[float] = None
    model_fuse_fn_name: Optional[str] = None
    model_fuse_fn_kwargs: Optional[Dict[str, Any]] = None
    num_calibration_steps: Optional[int] = None
    post_oneshot_calibration: Optional[bool] = False
    strict: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scheme = QuantizationScheme.load(self.scheme)
        self.scheme_overrides = _load_quantization_schemes_dict(
            self.scheme_overrides, self.scheme
        )
        if self.model_fuse_fn_kwargs is None:
            self.model_fuse_fn_kwargs = {}

    def on_initialize_structure(self, state: State, **kwargs):
        pass  # nothing needed for this modifier


class _QuantizationSchemesDict(dict):
    # wrapper class for dict to override the __str__ method for yaml serialization

    def __str__(self):
        return str({submodule: scheme.dict() for submodule, scheme in self.items()})


def _load_quantization_schemes_dict(
    schemes_dict: Optional[Dict[str, QuantizationSchemeLoadable]],
    default_scheme: QuantizationScheme,
) -> Dict[str, QuantizationScheme]:
    if schemes_dict is None:
        return {}
    return _QuantizationSchemesDict(
        {
            submodule: QuantizationScheme.load(scheme, default=default_scheme)
            for submodule, scheme in schemes_dict.items()
        }
    )
