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
from sparseml.pytorch.sparsification.modifier import ScheduledModifier
from sparseml.pytorch.sparsification.quantization.helpers import (
    CONV_ACTIVATION_NAMES,
    LINEAR_ACTIVATION_NAMES,
    QConfigProperties,
    add_quant_dequant,
    configure_module_bn_wrappers,
    configure_module_default_qconfigs,
    configure_module_qat_wrappers,
    freeze_bn_stats,
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
        to performing QAT.  Set as None or 'no_fuse' to skip module fusing. Set as
         'conv_bv_relus' to use `sparseml.pytorch.utils.fuse_module_conv_bn_relus`.
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
    :param quantize_linear_activations: if True, FakeQuantize ops will be run
        for output activations of fully connected layers. Default is True.
    :param quantize_conv_activations: if True, FakeQuantize ops will be run
        for output activations of convolutional layers. Default is True.
    :param quantize_embedding_activations: if True, FakeQuantize ops will be run
        for output activations of embedding layers. Default is True.
    :param activation_bits: Number of bits to use for setting quant min/max values for
        activations. Default 8.
    :param weight_bits: Number of bits to use for setting quant min/max values for
        weights. Default is 8.
    :param num_calibration_steps: Number of steps to run post training calibration for.
        When None, the entire calibration_dataloader is used
    :param exclude_batchnorm: If True, do not propagate quantization qconfigs to
        batch-normalization modules
    :param exclude_module_types: optional list of module class names
        to not propagate quantization configs to. Default is None
    :param custom_quantizable_module_types: optional list of module class names
        to be added to the list of quantizable modules. Default is None
    :param activation_qconfig_kwargs: Additional kwargs for quantization of
        activations.
    :param weight_qconfig_kwargs: Additional kwargs for quantization of
        weights.
    :param tenssorrt: if True sets quantization configuration for compatibility with
       explict quantization as supported by TensorRT 8.2.
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
        quantize_conv_activations: bool = True,
        quantize_embedding_activations: bool = True,
        activation_bits: int = 8,
        weight_bits: int = 8,
        num_calibration_steps: Optional[int] = None,
        exclude_batchnorm: bool = True,
        exclude_module_types: Optional[List[str]] = None,
        custom_quantizable_module_types: Optional[List[str]] = None,
        activation_qconfig_kwargs: Optional[Dict[str, Any]] = None,
        weight_qconfig_kwargs: Optional[Dict[str, Any]] = None,
        tensorrt: bool = False,
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
        self._quantize_conv_activations = quantize_conv_activations
        self._quantize_embedding_activations = quantize_embedding_activations
        self._activation_bits = activation_bits
        self._weight_bits = weight_bits
        self._exclude_batchnorm = exclude_batchnorm
        self._exclude_module_types = exclude_module_types
        self._custom_quantizable_module_types = custom_quantizable_module_types

        self._modules_to_quantize = None
        self._qat_enabled = False
        self._quantization_observer_disabled = False
        self._bn_stats_frozen = False
        self._activation_qconfig_kwargs = activation_qconfig_kwargs
        self._weight_qconfig_kwargs = weight_qconfig_kwargs
        self._tensorrt = tensorrt

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
            to performing QAT. None sets to default function.
            If tensorrt flag is True, default is 'no_fuse', otherwise
            `sparseml.pytorch.utils.fuse_module_conv_bn_relus`.
        """
        if self.tensorrt:
            _LOGGER.info(
                "Overriding model_fuse_fn_name to False because tensorrt flag is True."
            )
            fuse_fn = (
                self._model_fuse_fn_name if self._model_fuse_fn_name else "no_fuse"
            )
        else:
            fuse_fn = (
                self._model_fuse_fn_name
                if self._model_fuse_fn_name
                else "conv_bn_relus"
            )
        return fuse_fn

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
        :return: if True, FakeQuantize ops will be run for output activations
            of fully connected layers
        """
        if self.tensorrt:
            _LOGGER.info(
                "Overriding quantize_linear_activations to False "
                "because tensorrt flag is True."
            )
            return False
        else:
            return self._quantize_linear_activations

    @ModifierProp()
    def quantize_conv_activations(self) -> bool:
        """
        :return: if True, FakeQuantize ops will be run for output activations
            of convolutional layers
        """
        if self.tensorrt:
            _LOGGER.info(
                "Overriding quantize_conv_activations to False "
                "because tensorrt flag is True."
            )
            return False
        else:
            return self._quantize_conv_activations

    @ModifierProp()
    def quantize_embedding_activations(self) -> bool:
        """
        :return: if True, FakeQuantize ops will be run for output activations
            of convolutional layers
        """
        if self.tensorrt:
            _LOGGER.info(
                "Overriding quantize_embedding_activations to False "
                "because tensorrt flag is True."
            )
            return False
        else:
            return self._quantize_embedding_activations

    @ModifierProp()
    def custom_quantizable_module_types(self) -> Union[List[str], None]:
        """
        :return: optional list of module class names to be included
            in list of quantizable modules. Default is None
        """
        return self._custom_quantizable_module_types

    @ModifierProp()
    def exclude_module_types(self) -> Union[List[str], None]:
        """
        :return: optional list of module class names to not propagate
            quantization configs to. Default is None
        """
        return self._exclude_module_types

    @ModifierProp()
    def exclude_batchnorm(self) -> bool:
        """
        :return: if True, do not propagate quantization qconfigs to
        batch-normalization modules
        """
        return self._exclude_batchnorm

    @ModifierProp()
    def activation_bits(self) -> Optional[int]:
        """
        :return: Number of bits to be use for setting quant min/max values for
            activations. Default is None, which will quantize activations to 8 bits.
        """
        return self._activation_bits

    @ModifierProp()
    def weight_bits(self) -> Optional[int]:
        """
        :return: Number of bits to be use for setting quant min/max values for
            weights. Default is None, which will quantize weights to 8 bits.
        """
        return self._weight_bits

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
        if (
            self._weight_qconfig_kwargs is not None
            and "observer" in self._weight_qconfig_kwargs
        ):
            kwargs = self._weight_qconfig_kwargs.copy()
            if kwargs["observer"] == "minmaxobserver":
                kwargs["observer"] = torch_quantization.MinMaxObserver
            return kwargs
        else:
            return self._weight_qconfig_kwargs

    @ModifierProp()
    def num_calibration_steps(self) -> Optional[int]:
        """
        :return: Number of steps to run post training calibration for.
            When None, the entire calibration_dataloader is used
        """
        return self._num_calibration_steps

    @ModifierProp()
    def tensorrt(self) -> bool:
        """
        :return: boolean. When set to True overrides quantization configs
        to be compatible with TensorRT.
        """
        return self._tensorrt

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

    def advance_epochs(self, ref_start_epoch: float = None):
        """
        Advance epoch attributes given a reference start epoch

        :param ref_start_epoch: the reference, i.e. new, start epoch
        """
        if ref_start_epoch is None:
            return

        super().advance_epochs(ref_start_epoch=ref_start_epoch)

        if self._disable_quantization_observer_epoch is not None:
            self._disable_quantization_observer_epoch = (
                max(0.0, self._disable_quantization_observer_epoch) + ref_start_epoch
            )

        if self._freeze_bn_stats_epoch is not None:
            self._freeze_bn_stats_epoch = (
                max(0.0, self._freeze_bn_stats_epoch) + ref_start_epoch
            )
        self._validate_params()

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
                quant_module.apply(freeze_bn_stats)
            self._bn_stats_frozen = True

    def _enable_module_qat(self, module: Module):
        # fuse module Conv-BNs
        if self.model_fuse_fn_name == "conv_bn_relus":
            self._model_fuse_fn_kwargs["inplace"] = True
            fuse_module_conv_bn_relus(module, **self._model_fuse_fn_kwargs)
        elif self.model_fuse_fn_name != "no_fuse":
            module_fuse_fn = getattr(module, self._model_fuse_fn_name, None)
            if module_fuse_fn is None or not callable(module_fuse_fn):
                raise ValueError(
                    "Invalid model_fuse_fn_name. "
                    "Module has no callable function {}".format(
                        self._model_fuse_fn_name
                    )
                )
            module_fuse_fn(**self._model_fuse_fn_kwargs)

        # build list of layer types that should not quantize output activations
        remove_activation_qat_layers = ["FloatFunctional"]
        if not self.quantize_linear_activations:
            remove_activation_qat_layers.extend(LINEAR_ACTIVATION_NAMES)

        if not self.quantize_conv_activations:
            remove_activation_qat_layers.extend(CONV_ACTIVATION_NAMES)

        if not self.quantize_embedding_activations:
            remove_activation_qat_layers.append("Embedding")

        # fix for freezing batchnorm statistics when not fusing BN with convs.
        # pytorch only supports freezing batchnorm statistics for fused modules.
        # this fix wraps BN modules adding with a new module class that supports
        # methods related to freezing/unfreezing BN statistics.
        configure_module_bn_wrappers(module)

        # set qconfig.
        # if tensorrt flag is used, set activation and weights to symmetric
        # quantization.
        # otherwise, use the default values set in QConfigProperties
        qproperties = QConfigProperties(
            activation_bits=self.activation_bits,
            weight_bits=self.weight_bits,
            activation_qconfig_kwargs=self.activation_qconfig_kwargs,
            weight_qconfig_kwargs=self.weight_qconfig_kwargs,
            reduce_range=self.reduce_range,
        )
        if self.tensorrt:
            _LOGGER.info(
                "Overriding quantization scheme to symmetric int8 "
                "for both weights and activations because tensorrt flag is True."
            )
            qproperties.tensorrt = True
            qproperties.activation_dtype = torch.qint8
            qproperties.weight_dtype = torch.qint8

        qconfig = get_qat_qconfig(qproperties)

        # prepare each module / submodule for quantization
        for name, quant_module in self._modules_to_quantize:
            # wrap any modules with wrap_qat set to True as QATWrapper(s)
            configure_module_qat_wrappers(quant_module, qproperties)

            # set quantization config (asymmetric activations, symmetric weights)
            quant_module.qconfig = qconfig

            # if for some reason the qconfig property is already set to None
            # in a submodule, the desired qconfig will not be propagated if
            # appropriate, calling helper function to delete these
            _clear_null_qconfigs(quant_module)

            # wrap all conv / linear blocks in with quantization observers
            torch_quantization.propagate_qconfig_(quant_module)
            configure_module_default_qconfigs(quant_module)
            add_quant_dequant(
                quant_module, name, module, self.custom_quantizable_module_types
            )

            # Remove output quantization from appropriate modules
            remove_activation_qat_by_layer_name(
                quant_module, remove_activation_qat_layers
            )

        # remove qconfigs for module types in exclude_module_types
        to_exclude = ["Softmax"]
        if self.exclude_module_types:
            to_exclude.extend(self.exclude_module_types)

        # if exclude_batchnorm flag is used, add batch norm layers to list of
        # modules to exclude qconfig
        if self.exclude_batchnorm:
            to_exclude.extend(["BatchNorm1d", "BatchNorm2d", "BatchNorm3d"])

        self._exclude_module_types = to_exclude
        if self.exclude_module_types:
            self._strip_excluded_module_qconfigs(module)

        # set modules with proper qconfigs to QAT mode
        self._prepare_qat(module, inplace=True)
        if self._quantize_embeddings:
            prepare_embeddings_qat(module, qproperties)

        self._qat_enabled = True
        self._calibrate_if_possible(module)

        # mark export mode for module Conv layers
        module.export_with_qlinearconv = self._quantize_conv_activations
        if hasattr(module, "module"):
            # for DP/DDP unwrapping
            module.module.export_with_qlinearconv = self._quantize_conv_activations

    def _prepare_qat(self, module, inplace=False):
        # Set training mode to satisfy a constraint during torch's prepare_qat
        prev_training_mode = module.training
        module.training = True
        torch_quantization.prepare_qat(module, inplace=inplace)
        module.training = prev_training_mode

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
        if not self.exclude_module_types:
            return
        excluded_classes = set(self.exclude_module_types)
        for submodule in module.modules():
            if submodule.__class__.__name__ in excluded_classes and hasattr(
                submodule, "qconfig"
            ):
                submodule.qconfig = None

    def _validate_params(self):
        self.validate_schedule()
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


def _clear_null_qconfigs(model: Module):
    for submodule in model.modules():
        if hasattr(submodule, "qconfig") and submodule.qconfig is None:
            del submodule.qconfig
