"""
Modifiers for inducing / enforcing kernel sparsity (model pruning)
on models while pruning.
"""
from typing import Union, List
import math
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
    get_named_layers_and_params_by_regex,
)
from neuralmagicML.pytorch.recal.modifier import (
    ModifierProp,
    ScheduledModifier,
    ScheduledUpdateModifier,
    PyTorchModifierYAML,
)
from neuralmagicML.pytorch.recal.sparsity_mask import (
    SparsityMaskCreator,
    load_mask_creator,
)
from neuralmagicML.pytorch.utils.logger import PyTorchLogger
from neuralmagicML.pytorch.recal.analyzer_ks import ModuleKSAnalyzer
from neuralmagicML.pytorch.recal.mask_ks import ModuleParamKSMask


__all__ = ["ConstantKSModifier", "GradualKSModifier"]


def _log_sparsity(
    analyzers: List[ModuleKSAnalyzer],
    loggers: List[PyTorchLogger],
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


@PyTorchModifierYAML()
class ConstantKSModifier(ScheduledModifier):
    """
    Holds the sparsity level and shape for a given parameter(s) constant while training.
    Useful for transfer learning use cases.

    | Sample yaml:
    |   !ConstantKSModifier
    |       start_epoch: 0.0
    |       end_epoch: 10.0
    |       params: ['re:.*weight']
    |       log_types: __ALL__

    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param params: A list of full parameter names or regex patterns of names to apply
        pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
        will match to all parameters.
    :param log_types: The loggers to allow the learning rate to be logged to,
        default is __ALL__
    """

    def __init__(
        self,
        start_epoch: float = -1.0,
        end_epoch: float = -1.0,
        params: Union[str, List[str]] = ["re:.*weight"],
        log_types: Union[str, List[str]] = ALL_TOKEN,
    ):
        super().__init__(
            log_types=log_types,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            end_comparator=-1,
        )
        self._params = validate_str_iterable(
            params, "{} for params".format(self.__class__.__name__)
        )
        self._module_masks = []  # type: List[ModuleParamKSMask]
        self._analyzers = None
        self._last_logged_epoch = None

    def __del__(self):
        for mask in self._module_masks:
            del mask

        self._module_masks.clear()

    @ModifierProp()
    def params(self) -> str:
        """
        :return: A list of full parameter names or regex patterns of names to apply
        pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
        will match to all parameters.
        """
        return self._params

    @params.setter
    def params(self, value: str):
        """
        :params value: A list of full parameter names or regex patterns of names to apply
        pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
        will match to all parameters.
        """
        self._params = validate_str_iterable(
            value, "{} for params".format(self.__class__.__name__)
        )

    def initialize(self, module: Module, optimizer: Optimizer):
        """
        Grab the params to control kernel sparsity for.

        :param module: module to modify
        :param optimizer: optimizer to modify
        """
        super().initialize(module, optimizer)

        param_names = (
            self._params
            if self._params != ALL_TOKEN and ALL_TOKEN not in self._params
            else ["re:.*"]
        )
        named_layers_and_params = get_named_layers_and_params_by_regex(
            module, param_names
        )

        self._analyzers = []
        for layer_name, layer, param_name, _ in named_layers_and_params:
            self._module_masks.append(ModuleParamKSMask(layer, param_name))
            self._analyzers.append(ModuleKSAnalyzer(layer, layer_name, param_name))

        if len(self._analyzers) == 0:
            raise ValueError(
                "Could not find any params matching {} in {}".format(
                    self._params, self.__class__.__name__
                )
            )

    def update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Update to enable and disable the mask when chosen.

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().update(module, optimizer, epoch, steps_per_epoch)

        if self.start_pending(epoch, steps_per_epoch):
            for mask in self._module_masks:
                mask.enabled = True

        if self.end_pending(epoch, steps_per_epoch):
            for mask in self._module_masks:
                mask.enabled = False

    def log_update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Check whether to log an update for the learning rate of the modifier.

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().log_update(module, optimizer, epoch, steps_per_epoch)

        if self._last_logged_epoch != math.floor(epoch):
            self._last_logged_epoch = math.floor(epoch)
            _log_sparsity(self._analyzers, self.loggers, epoch, steps_per_epoch)

    def optimizer_post_step(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Reapply the mask after the optimizer step in case the optimizer
        has momentum that may have moved weights from 0.

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().optimizer_post_step(module, optimizer, epoch, steps_per_epoch)

        # be sure to apply mask again after optimizer update because
        # weights may have changed (optimizer with momentum, not masking gradient)
        for mask in self._module_masks:
            mask.apply()


@PyTorchModifierYAML()
class GradualKSModifier(ScheduledUpdateModifier):
    """
    Gradually applies kernel sparsity to a given parameter or parameters from
    init_sparsity until final_sparsity is reached over a given amount of time
    and applied with an interpolated function for each step taken.

    Applies based on magnitude pruning unless otherwise specified by mask_type.

    | Sample yaml:
    |   !GradualKSModifier
    |       init_sparsity: 0.05
    |       final_sparsity: 0.8
    |       start_epoch: 0.0
    |       end_epoch: 10.0
    |       update_frequency: 1.0
    |       params: ["re:.*weight"]
    |       leave_enabled: True
    |       inter_func: cubic
    |       log_types: __ALL__
    |       mask_type: unstructured

    :param init_sparsity: the initial sparsity for the param to start with at
        start_epoch
    :param final_sparsity: the final sparsity for the param to end with at end_epoch
    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param update_frequency: The number of epochs or fraction of epochs to update at
        between start and end
    :param params: A list of full parameter names or regex patterns of names to apply
        pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
        will match to all parameters.
    :param leave_enabled: True to continue masking the weights after end_epoch,
        False to stop masking. Should be set to False if exporting the result
        immediately after or doing some other prune
    :param inter_func: the type of interpolation function to use:
        [linear, cubic, inverse_cubic]
    :param log_types: The loggers to allow the learning rate to be logged to,
        default is __ALL__
    :param mask_type: String to define type of sparsity (options: ['unstructured',
        'channel', 'filter']), List to define block shape of a parameter's in and out
        channels, or a SparsityMaskCreator object. default is 'unstructured'
    """

    def __init__(
        self,
        init_sparsity: float,
        final_sparsity: float,
        start_epoch: float,
        end_epoch: float,
        update_frequency: float,
        params: Union[str, List[str]] = ["re:.*weight"],
        leave_enabled: bool = True,
        inter_func: str = "cubic",
        log_types: Union[str, List[str]] = ALL_TOKEN,
        mask_type: Union[str, List[int], SparsityMaskCreator] = 'unstructured',
    ):
        super().__init__(
            log_types=log_types,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=update_frequency,
            min_end=0.0,
            end_comparator=1,
        )
        self._init_sparsity = init_sparsity
        self._final_sparsity = final_sparsity
        self._params = validate_str_iterable(
            params, "{} for params".format(self.__class__.__name__)
        )
        self._leave_enabled = convert_to_bool(leave_enabled)
        self._inter_func = inter_func
        self._mask_type = mask_type
        self._mask_creator = mask_type
        if not isinstance(mask_type, SparsityMaskCreator):
            self._mask_creator = load_mask_creator(mask_type)
        self._module_masks = []  # type: List[ModuleParamKSMask]
        self._applied_sparsity = None
        self._last_logged_sparsity = None
        self._analyzers = None

        self.validate()

    def __del__(self):
        for mask in self._module_masks:
            del mask

        self._module_masks.clear()

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
    def params(self) -> str:
        """
        :return: A list of full parameter names or regex patterns of names to apply
        pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
        will match to all parameters.
        """
        return self._params

    @params.setter
    def params(self, value: str):
        """
        :param value: A list of full parameter names or regex patterns of names to apply
        pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
        will match to all parameters.
        """
        self._params = validate_str_iterable(
            value, "{} for params".format(self.__class__.__name__)
        )

    @ModifierProp()
    def leave_enabled(self) -> bool:
        """
        :return: True to continue masking the weights after end_epoch,
            False to stop masking. Note, if set as False, sparsity will not be enforced
            and the model will likely deviate from the sparse solution
        """
        return self._leave_enabled

    @leave_enabled.setter
    def leave_enabled(self, value: bool):
        """
        :param value: True to continue masking the weights after end_epoch,
            False to stop masking. Note, if set as False, sparsity will not be enforced
            and the model will likely deviate from the sparse solution
        """
        self._leave_enabled = value

    @ModifierProp()
    def inter_func(self) -> str:
        """
        :return: the type of interpolation function to use:
            [linear, cubic, inverse_cubic]
        """
        return self._inter_func

    @inter_func.setter
    def inter_func(self, value: str):
        """
        :param value: the type of interpolation function to use:
            [linear, cubic, inverse_cubic]
        """
        self._inter_func = value
        self.validate()

    @ModifierProp()
    def mask_type(self) -> Union[str, List[int], SparsityMaskCreator]:
        """
        :return: the SparsityMaskCreator object used
        """
        return self._mask_type

    @mask_type.setter
    def mask_type(self, value: Union[str, List[int], SparsityMaskCreator]):
        """
        :param value: the SparsityMaskCreator object to use
        """
        self._mask_type = value
        self._mask_creator = value
        if not isinstance(value, SparsityMaskCreator):
            self._mask_creator = load_mask_creator(value)

    @ModifierProp(serializable=False)
    def applied_sparsity(self) -> float:
        """
        :return: the currently applied sparsity level to the contained params
        """
        return self._applied_sparsity

    def initialize(self, module: Module, optimizer: Optimizer):
        """
        Grab the params to control kernel sparsity for

        :param module: module to modify
        :param optimizer: optimizer to modify
        """
        super().initialize(module, optimizer)

        param_names = (
            self._params
            if self._params != ALL_TOKEN and ALL_TOKEN not in self._params
            else ["re:.*"]
        )
        named_layers_and_params = get_named_layers_and_params_by_regex(
            module, param_names
        )

        self._analyzers = []
        for layer_name, layer, param_name, _ in named_layers_and_params:
            self._module_masks.append(ModuleParamKSMask(
                layer, param_name, mask_creator=self._mask_creator
            ))
            self._analyzers.append(ModuleKSAnalyzer(layer, layer_name, param_name))

        if len(self._analyzers) == 0:
            raise ValueError(
                "Could not find any params matching {} in {}".format(
                    self._params, self.__class__.__name__
                )
            )

    def update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Update the sparsity mask for the selected parameters.
        If start, enables the masks.
        If end, disables the masks if leave_enabled is False.

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
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
        Check whether to log an update for the learning rate of the modifier.

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().log_update(module, optimizer, epoch, steps_per_epoch)

        if self._applied_sparsity != self._last_logged_sparsity:
            self._last_logged_sparsity = self._applied_sparsity
            _log_sparsity(self._analyzers, self.loggers, epoch, steps_per_epoch)

    def optimizer_post_step(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Reapply the mask after the optimizer step in case the optimizer has momentum
        that may have moved weights from 0.

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().optimizer_post_step(module, optimizer, epoch, steps_per_epoch)

        # be sure to apply mask again after optimizer update because weights may
        # have changed (optimizer with momentum, not masking gradient)
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
                (
                    "final_sparsity value must be in the range [0.0, 1.0],"
                    " given {} for {}"
                ).format(self._init_sparsity, self.__class__.__name__)
            )

        if not isinstance(self._final_sparsity, float):
            raise TypeError(
                "final_sparsity must be of float type for {}".format(
                    self.__class__.__name__
                )
            )

        if self._final_sparsity < 0.0 or self._final_sparsity > 1.0:
            raise ValueError(
                (
                    "init_sparsity value must be in the range [0.0, 1.0],"
                    " given {} for {}"
                ).format(self._init_sparsity, self.__class__.__name__)
            )

        if self._inter_func not in INTERPOLATION_FUNCS:
            raise ValueError(
                (
                    "{} is not a supported inter_func in layers_settings,"
                    " available are {} for {}"
                ).format(self._inter_func, INTERPOLATION_FUNCS, self.__class__.__name__)
            )
