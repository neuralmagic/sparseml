"""
Code related to modifiers for enforcing kernel sparsity (model pruning) on models while pruning
"""

from typing import Union, List
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from neuralmagicML.utils import (
    ALL_TOKEN,
    INTERPOLATION_FUNCS,
    convert_to_bool,
    interpolate,
    validate_str_iterable,
)
from neuralmagicML.pytorch.utils import (
    get_terminal_layers,
    get_layer,
)
from neuralmagicML.pytorch.recal.modifier import (
    ModifierProp,
    ScheduledUpdateModifier,
    PytorchModifierYAML,
)
from neuralmagicML.pytorch.utils.logger import PytorchLogger
from neuralmagicML.pytorch.recal.kernel.analyzer import ModuleKSAnalyzer
from neuralmagicML.pytorch.recal.kernel.mask import ModuleParamKSMask


__all__ = ["GradualKSModifier"]


def _log_sparsity(
    analyzers: List[ModuleKSAnalyzer],
    loggers: List[PytorchLogger],
    epoch: float,
    steps_per_epoch: int,
):
    step = round(epoch) if steps_per_epoch <= 0 else round(epoch * steps_per_epoch)

    for logger in loggers:
        for analyzer in analyzers:
            logger.log_scalar(
                "Modifier KS/{}".format(analyzer.tag),
                analyzer.param_sparsity.item(),
                step,
            )


@PytorchModifierYAML()
class GradualKSModifier(ScheduledUpdateModifier):
    """
    Gradually applies kernel sparsity to a given layer or layers from init_sparsity until final_sparsity is reached
    over a given amount of time and applied with an interpolated function for each step taken

    Applies based on magnitude pruning without any structure to the pruning

    Sample yaml:
        !GradualKSModifier
            layers: __ALL__
            init_sparsity: 0.05
            final_sparsity: 0.8
            start_epoch: 0.0
            end_epoch: 10.0
            update_frequency: 1.0
            param: weight
            leave_enabled: True
            inter_func: cubic
            log_types: __ALL__
    """

    def __init__(
        self,
        layers: Union[str, List[str]],
        init_sparsity: float,
        final_sparsity: float,
        start_epoch: float,
        end_epoch: float,
        update_frequency: float,
        param: str = "weight",
        leave_enabled: bool = True,
        inter_func: str = "cubic",
        log_types: Union[str, List[str]] = ALL_TOKEN,
    ):
        """
        :param layers: str or list of str for the layers to apply the KS modifier to
                       can also use the token __ALL__ to specify all layers
        :param init_sparsity: the initial sparsity for the param to start with at start_epoch
        :param final_sparsity: the final sparsity for the param to end with at end_epoch
        :param start_epoch: The epoch to start the modifier at
        :param end_epoch: The epoch to end the modifier at
        :param update_frequency: The number of epochs or fraction of epochs to update at between start and end
        :param param: the name of the parameter to apply pruning to, generally 'weight' for linear and convs
        :param leave_enabled: True to continue masking the weights after end_epoch, False to stop masking
                              Should be set to False if exporting the result immediately after or doing some other prune
        :param inter_func: the type of interpolation function to use: [linear, cubic, inverse_cubic]
        :param log_types: The loggers to allow the learning rate to be logged to, default is __ALL__
        """
        super().__init__(
            log_types=log_types,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=update_frequency,
            min_end=0.0,
            end_comparator=1,
        )
        self._layers = validate_str_iterable(
            layers, "{} for layers".format(self.__class__.__name__)
        )
        self._init_sparsity = init_sparsity
        self._final_sparsity = final_sparsity
        self._param = param
        self._leave_enabled = convert_to_bool(leave_enabled)
        self._inter_func = inter_func
        self._module_masks = []  # type: List[ModuleParamKSMask]
        self._applied_sparsity = None
        self._last_logged_sparsity = None
        self._analyzers = None

        self.validate()

    def __del__(self):
        self._module_masks.clear()

    @ModifierProp()
    def layers(self) -> Union[str, List[str]]:
        """
        :return: str or list of str for the layers to apply the KS modifier to
                 can also use the token __ALL__ to specify all layers
        """
        return self._layers

    @layers.setter
    def layers(self, value: Union[str, List[str]]):
        """
        :param value: str or list of str for the layers to apply the KS modifier to
                      can also use the token __ALL__ to specify all layers
        """
        self._layers = validate_str_iterable(
            value, "{} for layers".format(self.__class__.__name__)
        )

    @ModifierProp()
    def init_sparsity(self) -> float:
        """
        :return: the initial sparsity for the param to start with at start_epoch
        """
        return self._init_sparsity

    @init_sparsity.setter
    def init_sparsity(self, value: float):
        """
        :param value: the initial sparsity for the param to start with at start_epoch
        """
        self._init_sparsity = value
        self.validate()

    @ModifierProp()
    def final_sparsity(self) -> float:
        """
        :return: the final sparsity for the param to end with at end_epoch
        """
        return self._final_sparsity

    @final_sparsity.setter
    def final_sparsity(self, value: float):
        """
        :param value: the final sparsity for the param to end with at end_epoch
        """
        self._final_sparsity = value
        self.validate()

    @ModifierProp()
    def param(self) -> str:
        """
        :return: the name of the parameter to apply pruning to, generally 'weight' for linear and convs
        """
        return self._param

    @param.setter
    def param(self, value: str):
        """
        :param value: the name of the parameter to apply pruning to, generally 'weight' for linear and convs
        """
        self._param = value

    @ModifierProp()
    def leave_enabled(self) -> bool:
        """
        :return: True to continue masking the weights after end_epoch, False to stop masking
                 Should be set to False if exporting the result immediately after or doing some other prune
        """
        return self._leave_enabled

    @leave_enabled.setter
    def leave_enabled(self, value: bool):
        """
        :param value: True to continue masking the weights after end_epoch, False to stop masking
                      Should be set to False if exporting the result immediately after or doing some other prune
        """
        self._leave_enabled = value

    @ModifierProp()
    def inter_func(self) -> str:
        """
        :return: the type of interpolation function to use: [linear, cubic, inverse_cubic]
        """
        return self._inter_func

    @inter_func.setter
    def inter_func(self, value: str):
        """
        :param value: the type of interpolation function to use: [linear, cubic, inverse_cubic]
        """
        self._inter_func = value
        self.validate()

    @ModifierProp(serializable=False)
    def applied_sparsity(self) -> float:
        """
        :return: the currently applied sparsity level to the contained params
        """
        return self._applied_sparsity

    def initialize(self, module: Module, optimizer: Optimizer):
        """
        Grab the layers' params to control kernel sparsity for

        :param module: module to modify
        :param optimizer: optimizer to modify
        """
        super(GradualKSModifier, self).initialize(module, optimizer)
        layers = (
            get_terminal_layers(module)
            if self._layers == ALL_TOKEN
            else {name: get_layer(name, module) for name in self._layers}
        )
        self._analyzers = []

        for name, layer in layers.items():
            found = False

            for param_name, par in layer.named_parameters():
                if param_name == self._param:
                    self._module_masks.append(ModuleParamKSMask(layer, self._param))
                    self._analyzers.append(ModuleKSAnalyzer(layer, name, param_name))
                    found = True
                    break

            if not found:
                raise ValueError(
                    "Could not find required param {} in layer {} for {}".format(
                        self._param, layer, self.__class__.__name__
                    )
                )

    def update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Update the sparsity mask for the selected parameters
        If start, enables the masks
        If end, disables the masks if leave_enabled is False

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """
        super().update(module, optimizer, epoch, steps_per_epoch)

        if self.start_pending(epoch, steps_per_epoch):
            for mask in self._module_masks:
                mask.enabled = True

        if self.end_pending(epoch, steps_per_epoch) and not self._leave_enabled:
            for mask in self._module_masks:
                mask.enabled = False

        # set the mask tensors according to the new sparsity
        self._applied_sparsity = interpolate(
            epoch,
            self.start_epoch,
            self.end_epoch,
            self._init_sparsity,
            self._final_sparsity,
            self._inter_func,
        )

        for mask in self._module_masks:
            mask.set_param_mask_from_sparsity(self._applied_sparsity)

    def log_update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Check whether to log an update for the learning rate of the modifier
        If constant logging is enabled, then will always log
        Otherwise checks for a change in the LR before logging

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """
        super().log_update(module, optimizer, epoch, steps_per_epoch)

        if self._applied_sparsity != self._last_logged_sparsity:
            self._last_logged_sparsity = self._applied_sparsity
            _log_sparsity(self._analyzers, self.loggers, epoch, steps_per_epoch)

    def optimizer_post_step(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Reapply the mask after the optimizer step in case the optimizer has momentum that may have moved weights from 0

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """
        super().optimizer_post_step(module, optimizer, epoch, steps_per_epoch)

        # be sure to apply mask again after optimizer update because weights may have changed
        # (optimizer with momentum, not masking gradient)
        for mask in self._module_masks:
            mask.apply()

    def validate(self):
        """
        Validate the values of the params for the current instance are valid
        """

        if not isinstance(self._init_sparsity, float):
            raise TypeError(
                "init_sparsity must be of float type for {}".format(
                    self.__class__.__name__
                )
            )

        if self._init_sparsity < 0.0 or self._init_sparsity > 1.0:
            raise ValueError(
                "final_sparsity value must be in the range [0.0, 1.0], given {} for {}".format(
                    self._init_sparsity, self.__class__.__name__
                )
            )

        if not isinstance(self._final_sparsity, float):
            raise TypeError(
                "final_sparsity must be of float type for {}".format(
                    self.__class__.__name__
                )
            )

        if self._final_sparsity < 0.0 or self._final_sparsity > 1.0:
            raise ValueError(
                "init_sparsity value must be in the range [0.0, 1.0], given {} for {}".format(
                    self._init_sparsity, self.__class__.__name__
                )
            )

        if self._inter_func not in INTERPOLATION_FUNCS:
            raise ValueError(
                "{} is not a supported inter_func in layers_settings, available are {} for {}".format(
                    self._inter_func, INTERPOLATION_FUNCS, self.__class__.__name__
                )
            )
