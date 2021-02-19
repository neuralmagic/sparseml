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
Modifier for models through quantization aware training.

PyTorch version must support quantization (>=1.2, ONNX export support introduced in 1.7)
"""


from typing import Any, Dict, List, Union

from torch.nn import Module
from torch.optim.optimizer import Optimizer


try:
    from torch import quantization as torch_quantization
    from torch.nn import intrinsic as torch_intrinsic
except Exception:
    torch_quantization = None
    torch_intrinsic = None

from sparseml.optim import ModifierProp
from sparseml.pytorch.optim.modifier import PyTorchModifierYAML, ScheduledModifier
from sparseml.pytorch.optim.quantization import (
    add_quant_dequant,
    fuse_module_conv_bn_relus,
    get_qat_qconfig,
)


__all__ = [
    "QuantizationModifier",
]


@PyTorchModifierYAML()
class QuantizationModifier(ScheduledModifier):
    """
    Enables quantization aware training (QAT) for a given module or its submodules
    After the start epoch, the specified module(s)' forward pass will emulate
    quantized execution and the modifier will be enabled until training is completed.

    | Sample yaml:
    |   !QuantizationModifier
    |       start_epoch: 0.0
    |       submodules: ['blocks.0', 'blocks.2']
    |       model_fuse_fn_name: 'fuse_module'
    |       disable_quantization_observer_epoch: 2.0
    |       freeze_bn_stats_epoch: 3.0

    :param start_epoch: The epoch to start the modifier at
    :param submodules: List of submodule names to perform QAT on. Leave None to quantize
        entire model. Default is None
    :param model_fuse_fn_name: Name of model function to fuse the model in place prior
        to performing QAT.  Set as 'no_fuse' to skip module fusing. Leave None to use
        the default function `sparseml.pytorch.utils.fuse_module_conv_bn_relus`.
        Default is None
    :param disable_quantization_observer_epoch: Epoch to disable updates to the module's
        quantization observers. After this point, quantized weights and zero points will
        not be updated. Leave None to not disable observers during QAT. Default is None
    :param freeze_bn_stats_epoch: Epoch to stop the tracking of batch norm stats. Leave
        None to not stop tracking batch norm stats during QAT. Default is None
    :param end_epoch: Disabled, setting to anything other than -1 will raise an
        exception. For compatibility with YAML serialization only.
    :param model_fuse_fn_kwargs: dictionary of keyword argument values to be passed
        to the model fusing function
    """

    def __init__(
        self,
        start_epoch: float = -1.0,
        submodules: Union[List[str], None] = None,
        model_fuse_fn_name: Union[str, None] = None,
        disable_quantization_observer_epoch: Union[float, None] = None,
        freeze_bn_stats_epoch: Union[float, None] = None,
        end_epoch: float = -1,
        model_fuse_fn_kwargs: Dict[str, Any] = None,
    ):
        if torch_quantization is None or torch_intrinsic is None:
            raise RuntimeError(
                "Unable to import package torch.quantization and/or "
                "torch.nn.intrinsic. "
                "Try upgrading your PyTorch version to use the QuantizationModifier."
            )
        if end_epoch != -1:
            raise ValueError(
                "end_epoch is disabled for QuantizationModifier and can only be set to"
                " -1. Given {}".format(end_epoch)
            )

        super().__init__(start_epoch=start_epoch, end_epoch=-1.0, end_comparator=-1)

        self._start_epoch = start_epoch
        self._submodules = submodules
        self._model_fuse_fn_name = model_fuse_fn_name
        self._model_fuse_fn_kwargs = model_fuse_fn_kwargs or {}
        self._disable_quantization_observer_epoch = disable_quantization_observer_epoch
        self._freeze_bn_stats_epoch = freeze_bn_stats_epoch

        self._modules_to_quantize = None
        self._quantization_observer_disabled = False
        self._bn_stats_frozen = False

        if (
            isinstance(self._model_fuse_fn_name, str)
            and self._model_fuse_fn_name.lower() == "none"
        ):
            self._model_fuse_fn_name = None
        if isinstance(self._submodules, list):
            self._submodules = set(self._submodules)

        self._validate_params()

    @ModifierProp()
    def submodules(self) -> Union[List[str], None]:
        """
        :return: List of submodule names to perform QAT on. None quantizes the entire
            model
        """
        return list(self._submodules) if self._submodules is not None else None

    @submodules.setter
    def submodules(self, value: Union[List[str], None]):
        """
        :params value: List of submodule names to perform QAT on. Set None to quantize
        entire model
        """
        self._submodules = value
        if isinstance(self._submodules, list):
            self._submodules = set(self._submodules)
        self._validate_params()

    @ModifierProp()
    def model_fuse_fn_name(self) -> Union[str, None]:
        """
        :return: Name of model function to fuse the model in place prior
        to performing QAT. None to uses the default function
        `sparseml.pytorch.utils.fuse_module_conv_bn_relus`.
        """
        return self._model_fuse_fn_name

    @model_fuse_fn_name.setter
    def model_fuse_fn_name(self, value: Union[str, None]):
        """
        :params value: Name of model function to fuse the model in place prior
        to performing QAT. Set None to use the default function
        `sparseml.pytorch.utils.fuse_module_conv_bn_relus`.  Set as 'no_fuse'
        to skip module fusing.
        """
        self._model_fuse_fn_name = value
        if (
            isinstance(self._model_fuse_fn_name, str)
            and self._model_fuse_fn_name.lower() == "none"
        ):
            self._model_fuse_fn_name = None
        self._validate_params()

    @ModifierProp()
    def disable_quantization_observer_epoch(self) -> Union[float, None]:
        """
        :return: Epoch to disable updates to the module's
        quantization observers. After this point, quantized weights and zero points will
        not be updated. When None, observers never disabled during QAT
        """
        return self._disable_quantization_observer_epoch

    @disable_quantization_observer_epoch.setter
    def disable_quantization_observer_epoch(self, value: Union[float, None]):
        """
        :params value: Epoch to disable updates to the module's
        quantization observers. After this point, quantized weights and zero points will
        not be updated. Set None to not disable observers during QAT
        """
        self._disable_quantization_observer_epoch = value
        self._validate_params()

    @ModifierProp()
    def freeze_bn_stats_epoch(self) -> Union[float, None]:
        """
        :return: Epoch to stop the tracking of batch norm stats. When
        None, batch norm stats are track for all of training
        """
        return self._freeze_bn_stats_epoch

    @freeze_bn_stats_epoch.setter
    def freeze_bn_stats_epoch(self, value: Union[float, None]):
        """
        :params value: Epoch to stop the tracking of batch norm stats. Set
        None to not stop tracking batch norm stats during QAT
        """
        self._freeze_bn_stats_epoch = value
        self._validate_params()

    def initialize(self, module: Module, optimizer: Optimizer):
        """
        Grab the module / submodule to perform QAT on

        :param module: module to modify
        :param optimizer: optimizer to modify
        """
        super().initialize(module, optimizer)
        self._modules_to_quantize = []
        if self._submodules is not None:
            found_submodules = []
            for name, submodule in module.named_modules():
                if name in self._submodules:
                    self._modules_to_quantize.append(submodule)
                    found_submodules.append(name)
            if not len(found_submodules) == len(self._submodules):
                raise RuntimeError(
                    "Could not find all provided submodules to quantize"
                    "given: {}, found: {}".format(
                        list(self._submodules), found_submodules
                    )
                )
        else:
            self._modules_to_quantize.append(module)

    def update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        If start_pending(), fuses the model, sets the model quantization config,
        calls torch.quantization.prepare_qat on the model to begin QAT
        If end_pending(), updates the modules layers params to their original
        trainable state.

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().update(module, optimizer, epoch, steps_per_epoch)
        if self.start_pending(epoch, steps_per_epoch):
            if (
                self._model_fuse_fn_name is not None
                and self._model_fuse_fn_name != "no_fuse"
            ):  # module class fn
                module_fuse_fn = getattr(module, self._model_fuse_fn_name, None)
                if module_fuse_fn is None or not callable(module_fuse_fn):
                    raise ValueError(
                        "Invalid model_fuse_fn_name. "
                        "Module has no callable function {}".format(
                            self._model_fuse_fn_name
                        )
                    )
                module_fuse_fn(**self._model_fuse_fn_kwargs)
            elif self._model_fuse_fn_name is None:  # default auto fn
                self._model_fuse_fn_kwargs["inplace"] = True
                fuse_module_conv_bn_relus(module, **self._model_fuse_fn_kwargs)
            # prepare each module / submodule for quantization
            qconfig = get_qat_qconfig()
            for quant_module in self._modules_to_quantize:
                # set quantization config (asymmetric activations, symmetric weights)
                quant_module.qconfig = qconfig
                # wrap all conv / linear blocks in with quantization observers
                torch_quantization.propagate_qconfig_(quant_module)
                add_quant_dequant(quant_module)
                # set model to QAT mode
                torch_quantization.prepare_qat(quant_module, inplace=True)

        if self._disable_quantization_observer_update_ready(epoch):
            for quant_module in self._modules_to_quantize:
                quant_module.apply(torch_quantization.disable_observer)
            self._quantization_observer_disabled = True

        if self._freeze_bn_stats_update_ready(epoch):
            for quant_module in self._modules_to_quantize:
                quant_module.apply(torch_intrinsic.qat.freeze_bn_stats)
            self._bn_stats_frozen = True

    def _disable_quantization_observer_update_ready(self, epoch: float) -> bool:
        return (
            self._disable_quantization_observer_epoch is not None
            and epoch >= self._disable_quantization_observer_epoch
            and not self._quantization_observer_disabled
        )

    def _freeze_bn_stats_update_ready(self, epoch: float) -> bool:
        return (
            self._freeze_bn_stats_epoch is not None
            and epoch >= self._freeze_bn_stats_epoch
            and not self._bn_stats_frozen
        )

    def update_ready(self, epoch: float, steps_per_epoch: int) -> bool:
        """

        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        :return: True if the modifier is pending an update and update() should be called
        """
        if not self._initialized:
            raise RuntimeError("modifier must be initialized first")

        if not self._enabled:
            return False

        pending = (
            self.start_pending(epoch, steps_per_epoch)
            or self._disable_quantization_observer_update_ready(epoch)
            or self._freeze_bn_stats_update_ready(epoch)
        )

        return pending

    def _validate_params(self):
        if (
            self._disable_quantization_observer_epoch is not None
            and self._disable_quantization_observer_epoch < self._start_epoch
        ):
            raise ValueError(
                f"disable_quantization_observer_epoch may not be greater than "
                f"start_epoch for QuantizationModifier, received: "
                f"{self._disable_quantization_observer_epoch} with start_epoch "
                f"{self._start_epoch}"
            )

        if (
            self._freeze_bn_stats_epoch is not None
            and self._freeze_bn_stats_epoch < self._start_epoch
        ):
            raise ValueError(
                "freeze_bn_stats_epoch may not be greater than start_epoch"
                " for QuantizationModifier, received: {} with start_epoch {}".format(
                    self._freeze_bn_stats_epoch, self._start_epoch
                )
            )
