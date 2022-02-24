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
import logging
import warnings
from itertools import cycle
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import torch
from torch.nn import Module
from torch.optim.optimizer import Optimizer


try:
    from torch import quantization as torch_quantization
    from torch.nn import intrinsic as torch_intrinsic
except Exception:
    torch_quantization = None
    torch_intrinsic = None

from sparseml.optim import BaseModifier, ModifierProp
from sparseml.pytorch.optim.modifier import PyTorchModifierYAML, ScheduledModifier
from sparseml.pytorch.sparsification.quantization.helpers import (
    add_quant_dequant,
    configure_module_default_qconfigs,
    configure_module_qat_wrappers,
    fix_observer_quant_range,
    fuse_module_conv_bn_relus,
    get_qat_qconfig,
    prepare_embeddings_qat,
    remove_activation_qat_by_layer_name,
)
from sparseml.pytorch.utils import BaseLogger, tensors_module_forward, tensors_to_device
from sparseml.sparsification import SparsificationTypes


_LOGGER = logging.getLogger(__name__)

__all__ = [
    "QuantizationModifier",
]

_ModuleToQuantize = NamedTuple(
    "_ModuleToQuantize", [("name", Optional[str]), ("module", Module)]
)


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
    |       reduce_range: False
    |       activation_bits: False

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
    :param quantize_embeddings: if True, will perform QAT on torch.nn.Embedding layers
        using sparseml.pytorch.utils.quantization.prepare_embeddings_qat to fake
        quantize embedding weights. Default is True. Models without embedding layers
        will be unaffected
    :param reduce_range: if True, the quantization range will be reduced by one bit.
        This may prevent overflow issues with model execution on certain hardware
        Default is False
    :param quantize_linear_activations: if False, FakeQuantize ops will not be run
        for activations of fully connected layers. this is important for quantizing
        transformer based models such as BERT where the quantized MatMul outputs
        are kept at 32 bits of precision and fake quantizing the outputs harm training
        recovery. Default is True
    :param activation_bits: Number of bits to use for setting quant min/max values for
            activations. Default is None, which will quantize activations to 8 bits.
    :param num_calibration_steps: Number of steps to run post training calibration for.
            When None, the entire calibration_dataloader is used
    :param exclude_module_types: optional list of module class names
        to not propagate quantization configs to. Default is None
    :param activation_qconfig_kwargs: Additional kwargs for quantization of
            activations.
    :param weight_qconfig_kwargs: Additional kwargs for quantization of
            weights.
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
        quantize_embeddings: bool = True,
        reduce_range: bool = False,
        quantize_linear_activations: bool = True,
        activation_bits: Optional[int] = None,
        num_calibration_steps: Optional[int] = None,
        exclude_module_types: Union[List[str], None] = None,
        activation_qconfig_kwargs: Optional[Dict[str, Any]] = None,
        weight_qconfig_kwargs: Optional[Dict[str, Any]] = None,
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
        self._quantize_embeddings = quantize_embeddings
        self._reduce_range = reduce_range
        self._quantize_linear_activations = quantize_linear_activations
        self._activation_bits = activation_bits
        self._exclude_module_types = exclude_module_types

        self._modules_to_quantize = None
        self._qat_enabled = False
        self._quantization_observer_disabled = False
        self._bn_stats_frozen = False
        self._activation_qconfig_kwargs = activation_qconfig_kwargs
        self._weight_qconfig_kwargs = weight_qconfig_kwargs

        self._calibration_dataloader = None
        self._calibration_function = None
        self._num_calibration_steps = num_calibration_steps
        if (
            isinstance(self._model_fuse_fn_name, str)
            and self._model_fuse_fn_name.lower() == "none"
        ):
            self._model_fuse_fn_name = None
        if isinstance(self._submodules, list):
            self._submodules = set(self._submodules)

        self._validate_params()

    @BaseModifier.sparsification_types.getter
    def sparsification_types(self) -> List[SparsificationTypes]:
        """
        :return: the sparsification types this modifier instance will apply
        """
        return [SparsificationTypes.quantization, SparsificationTypes.structured]

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
            quantization observers. After this point, quantized weights and zero points
            will not be updated. When None, observers never disabled during QAT
        """
        return self._disable_quantization_observer_epoch

    @disable_quantization_observer_epoch.setter
    def disable_quantization_observer_epoch(self, value: Union[float, None]):
        """
        :params value: Epoch to disable updates to the module's
            quantization observers. After this point, quantized weights and zero points
            will not be updated. Set None to not disable observers during QAT
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

    @ModifierProp()
    def quantize_embeddings(self) -> bool:
        """
        :return: if True, will perform QAT on torch.nn.Embedding layers
            using sparseml.pytorch.utils.quantization.prepare_embeddings_qat to fake
            quantize embedding weights
        """
        return self._quantize_embeddings

    @quantize_embeddings.setter
    def quantize_embeddings(self, value: bool):
        """
        :params value: if True, will perform QAT on torch.nn.Embedding layers
            using sparseml.pytorch.utils.quantization.prepare_embeddings_qat to fake
            quantize embedding weights
        """
        self._quantize_embeddings = value

    @ModifierProp()
    def reduce_range(self) -> bool:
        """
        :return: if True, the quantization range will be reduced by one
            This may prevent overflow issues with model execution on certain hardware
        """
        return self._reduce_range

    @ModifierProp()
    def quantize_linear_activations(self) -> bool:
        """
        :return: if False, FakeQuantize ops will not be run
            for activations of fully connected layers. this is important for quantizing
            transformer based models such as BERT where the quantized MatMul outputs
            are kept at 32 bits of precision and fake quantizing the outputs harm
            training recovery
        """
        return self._quantize_linear_activations

    @ModifierProp()
    def exclude_module_types(self) -> Union[List[str], None]:
        """
        :return: optional list of module class names to not propagate
            quantization configs to. Default is None
        """
        return self._exclude_module_types

    @ModifierProp()
    def activation_bits(self) -> Optional[int]:
        """
        :return: Number of bits to be use for setting quant min/max values for
            activations. Default is None, which will quantize activations to 8 bits.
        """
        return self._activation_bits

    @ModifierProp()
    def activation_qconfig_kwargs(self) -> Dict[str, Any]:
        """
        :return: Dictionary with correct quant_min, quant_max, and dtype values
            for activations

        """
        return self._activation_qconfig_kwargs

    @ModifierProp()
    def weight_qconfig_kwargs(self) -> Dict[str, Any]:
        """
        :return: Dictionary with correct quant_min, quant_max, and dtype values
            for weights

        """
        return self._weight_qconfig_kwargs

    @ModifierProp()
    def num_calibration_steps(self) -> Optional[int]:
        """
        :return: Number of steps to run post training calibration for.
            When None, the entire calibration_dataloader is used
        """
        return self._num_calibration_steps

    def initialize(
        self,
        module: Module,
        epoch: float = 0,
        loggers: Optional[List[BaseLogger]] = None,
        calibration_dataloader: Optional[Iterable[Tuple[List, Dict[str, Any]]]] = None,
        calibration_function: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Grab the module / submodule to perform QAT on

        :param module: the PyTorch model/module to modify
        :param epoch: The epoch to initialize the modifier and module at.
            Defaults to 0 (start of the training process)
        :param loggers: Optional list of loggers to log the modification process to
        :param calibration_dataloader: optional dataloader for running post training
            quantization with the given model. if present, calibration will be run
            immediately after quantization is enabled
        :param calibration_function: An Optional callable to use for
            calibration of module parameters post training. Should be able to
            accept a batch of inputs along with a module.
            Example: func(batch, module), Defaults to tensors_module_forward
        :param kwargs: Optional kwargs to support specific arguments
            for individual modifiers.
        """
        super().initialize(module, epoch, loggers, **kwargs)
        self._modules_to_quantize = []
        self._calibration_dataloader = calibration_dataloader
        self._calibration_function = calibration_function
        if self._submodules is not None:
            found_submodules = []
            for name, submodule in module.named_modules():
                if name in self._submodules:
                    self._modules_to_quantize.append(_ModuleToQuantize(name, submodule))
                    found_submodules.append(name)
            if not len(found_submodules) == len(self._submodules):
                raise RuntimeError(
                    "Could not find all provided submodules to quantize"
                    "given: {}, found: {}".format(
                        list(self._submodules), found_submodules
                    )
                )
        else:
            self._modules_to_quantize.append(_ModuleToQuantize(None, module))

        self._check_quantization_update(module, epoch, steps_per_epoch=0)

    def finalize(
        self, module: Optional[Module] = None, reset_loggers: bool = True, **kwargs
    ):
        """
        Cleans up any state

        :param module: The model/module to finalize the modifier for.
            Marked optional so state can still be cleaned up on delete,
            but generally should always be passed in.
        :param reset_loggers: True to remove any currently attached loggers (default),
            False to keep the loggers attached.
        :param kwargs: Optional kwargs to support specific arguments
            for individual modifiers.
        """
        super().finalize(module, reset_loggers, **kwargs)
        self._modules_to_quantize = None

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
        self._check_quantization_update(module, epoch, steps_per_epoch)

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

    def _check_quantization_update(
        self, module: Module, epoch: float, steps_per_epoch: int
    ):
        if self.start_pending(epoch, steps_per_epoch) and not self._qat_enabled:
            self._enable_module_qat(module)

        if self._disable_quantization_observer_update_ready(epoch):
            for _, quant_module in self._modules_to_quantize:
                quant_module.apply(torch_quantization.disable_observer)
            self._quantization_observer_disabled = True

        if self._freeze_bn_stats_update_ready(epoch):
            for _, quant_module in self._modules_to_quantize:
                quant_module.apply(torch_intrinsic.qat.freeze_bn_stats)
            self._bn_stats_frozen = True

    def _enable_module_qat(self, module: Module):
        # fuse module Conv-BNs
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

        activation_qconfig_kwargs = self._get_updated_activation_qconfig_kwargs()

        # prepare each module / submodule for quantization
        qconfig = get_qat_qconfig(
            reduce_range=self._reduce_range,
            activation_qconfig_kwargs=activation_qconfig_kwargs,
            weight_qconfig_kwargs=self.weight_qconfig_kwargs,
        )
        for name, quant_module in self._modules_to_quantize:
            # wrap any modules with wrap_qat set to True as QATWrapper(s)
            configure_module_qat_wrappers(
                quant_module,
                reduce_range=self._reduce_range,
                activation_qconfig_kwargs=activation_qconfig_kwargs,
                weight_qconfig_kwargs=self.weight_qconfig_kwargs,
            )
            # set quantization config (asymmetric activations, symmetric weights)
            quant_module.qconfig = qconfig
            # wrap all conv / linear blocks in with quantization observers
            torch_quantization.propagate_qconfig_(quant_module)
            configure_module_default_qconfigs(quant_module)

            add_quant_dequant(quant_module, name, module)

            if not self._quantize_linear_activations:
                remove_activation_qat_by_layer_name(quant_module, ["Linear"])

        # remove qconfigs for module types in exclude_module_types
        if self._exclude_module_types:
            self._strip_excluded_module_qconfigs(module)

        # set modules with proper qconfigs to QAT mode
        torch_quantization.prepare_qat(module, inplace=True)
        if self._quantize_embeddings:
            prepare_embeddings_qat(
                module,
                reduce_range=self._reduce_range,
                activation_qconfig_kwargs=activation_qconfig_kwargs,
                weight_qconfig_kwargs=self.weight_qconfig_kwargs,
            )

        # propagate custom quant min/max range from FakeQuantize to Observer objects
        fix_observer_quant_range(module)

        self._qat_enabled = True
        self._calibrate_if_possible(module)

    def _calibrate_if_possible(self, module):
        if self.num_calibration_steps == 0 and self._calibration_dataloader:
            warnings.warn(
                f"num_calibration_steps is {self.num_calibration_steps}."
                f"Calibration data loader will not be used."
            )
        elif self.num_calibration_steps and not self._calibration_dataloader:
            raise ValueError(
                f"num_calibration_steps is {self.num_calibration_steps}. "
                "Calibration data loader is not set. Pass a "
                "calibration_data_loader with initialize(...) method."
            )

        elif not self._calibration_dataloader or not self._qat_enabled:
            return

        elif self._calibration_dataloader:
            self._calibrate(module)

    def _calibrate(self, module):
        _LOGGER.info("Running quantization calibration using calibration_dataloader")

        module_training = module.training
        module.eval()

        forward_fn: Callable = (
            self._calibration_function
            if self._calibration_function
            else tensors_module_forward
        )

        model_device = next(module.parameters()).device
        _dataloader = (
            self._calibration_dataloader
            if self.num_calibration_steps is None
            else cycle(self._calibration_dataloader)
        )

        for batch_idx, batch in enumerate(_dataloader):
            if self.num_calibration_steps and batch_idx >= self.num_calibration_steps:
                break
            batch = tensors_to_device(batch, model_device)
            with torch.no_grad():
                forward_fn(batch, module=module)

        if module_training:
            module.train()

    def _get_updated_activation_qconfig_kwargs(self):
        activation_qconfig_kwargs = (
            self.activation_qconfig_kwargs.copy()
            if self.activation_qconfig_kwargs
            else {}
        )

        # update qconfig_kwargs for activation_bits
        if self.activation_bits and (
            activation_qconfig_kwargs.get("quant_min")
            or activation_qconfig_kwargs.get("quant_max")
        ):
            raise ValueError(
                "Cannot override quant_max and quant_min with activation_bits enabled"
            )

        if self.activation_bits:
            quant_min = 0
            quant_max = 2 ** self.activation_bits - 1
            dtype = torch.quint8
            activation_qconfig_kwargs.update(
                dict(
                    quant_min=quant_min,
                    quant_max=quant_max,
                    dtype=dtype,
                )
            )
        return activation_qconfig_kwargs

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

    def _strip_excluded_module_qconfigs(self, module: Module):
        if not self._exclude_module_types:
            return
        excluded_classes = set(self._exclude_module_types)
        for submodule in module.modules():
            if submodule.__class__.__name__ in excluded_classes and hasattr(
                submodule, "qconfig"
            ):
                submodule.qconfig = None

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
