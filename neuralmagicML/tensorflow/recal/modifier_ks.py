"""
Modifiers for inducing / enforcing kernel sparsity (model pruning)
on models while pruning.
"""

from typing import Union, List, Tuple
import hashlib

from neuralmagicML.utils import (
    ALL_TOKEN,
    validate_str_iterable,
    convert_to_bool,
)
from neuralmagicML.tensorflow.utils import tf_compat, VAR_INDEX_FROM_TRAINABLE
from neuralmagicML.tensorflow.recal.modifier import (
    ModifierProp,
    TensorFlowModifierYAML,
    ScheduledUpdateModifier,
)
from neuralmagicML.tensorflow.recal.mask_ks import (
    get_or_create_ks_schedule_ops,
    get_or_create_graph_ops_pruning,
    get_or_create_ks_update_op,
)


__all__ = ["GradualKSModifier"]


@TensorFlowModifierYAML()
class GradualKSModifier(ScheduledUpdateModifier):
    """
    Gradually applies kernel sparsity to a given layer or layers from
    init_sparsity until final_sparsity is reached over a given amount of time and
    applied with an interpolated function for each step taken.

    Applies based on magnitude pruning without any structure to the pruning.

    | Sample yaml:
    |   !GradualKSModifier
    |       layers: __ALL__
    |       init_sparsity: 0.05
    |       final_sparsity: 0.8
    |       start_epoch: 0.0
    |       end_epoch: 10.0
    |       update_frequency: 1.0
    |       param: from_trainable
    |       inter_func: cubic
    |       log_types: __ALL__

    :param layers: List of str for the layers (ops) to apply the KS modifier to
    :param init_sparsity: The initial sparsity for the param to
        start with at start_epoch
    :param final_sparsity: The final sparsity for the param to end with at end_epoch
    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param update_frequency: The number of epochs or fraction of epochs to
        update at between start and end
    :param param: The index to guide which input to grab from the operation.
        Can be set to an integer representing where the variable is, a string
        representing a name or portion of the name of the variable, or the default:
        "from_trainable" which tries to find from the trainable vars in the graph
    :param leave_enabled: True to continue masking the weights after end_epoch,
        False to stop masking. Should be set to False if exporting the result
        immediately after or doing some other prune
    :param inter_func: The type of interpolation function to use:
        [linear, cubic, inverse_cubic]
    :param log_types: The loggers to allow the learning rate to be logged to,
        default is __ALL__
    """

    def __init__(
        self,
        layers: List[str],
        init_sparsity: float,
        final_sparsity: float,
        start_epoch: float,
        end_epoch: float,
        update_frequency: float,
        param: Union[int, str] = VAR_INDEX_FROM_TRAINABLE,
        leave_enabled: bool = True,
        inter_func: str = "linear",
        log_types: Union[str, List[str]] = ALL_TOKEN,
    ):
        super(GradualKSModifier, self).__init__(
            log_types=log_types,
            start_epoch=start_epoch,
            min_start=-1.0,
            end_epoch=end_epoch,
            min_end=0.0,
            end_comparator=1,
            update_frequency=update_frequency,
            min_frequency=-1.0,
        )
        self._param = param
        self._layers = validate_str_iterable(
            layers, "{} for layers".format(self.__class__.__name__)
        )  # type: List[str]
        self._init_sparsity = init_sparsity
        self._final_sparsity = final_sparsity
        self._leave_enabled = convert_to_bool(leave_enabled)
        self._inter_func = inter_func

        self._check_setup()
        self._ks_group = self._create_group()
        self._exponent = None

        if self._inter_func == "linear":
            self._exponent = 1
        elif self._inter_func == "cubic":
            self._exponent = 3
        elif self._inter_func == "inverse_cubic":
            self._exponent = 1 / 3.0

    @ModifierProp()
    def layers(self) -> Union[str, List[str]]:
        """
        :return: List of str for the layers (ops) to apply the KS modifier to
        """
        return self._layers

    @layers.setter
    def layers(self, value: Union[str, List[str]]):
        """
        :param value: List of str for the layers (ops) to apply the KS modifier to
        """
        self._layers = value
        self.validate()

    @ModifierProp()
    def init_sparsity(self) -> float:
        """
        :return: The initial sparsity for the param to start with at start_epoch
        """
        return self._init_sparsity

    @init_sparsity.setter
    def init_sparsity(self, value: float):
        """
        :param value: The initial sparsity for the param to start with at start_epoch
        """
        self._init_sparsity = value
        self.validate()

    @ModifierProp()
    def final_sparsity(self) -> float:
        """
        :return: The final sparsity for the param to end with at end_epoch
        """
        return self._final_sparsity

    @final_sparsity.setter
    def final_sparsity(self, value: float):
        """
        :param value: The final sparsity for the param to end with at end_epoch
        """
        self._final_sparsity = value
        self.validate()

    @ModifierProp()
    def param(self) -> Union[int, str]:
        """
        :return: The index to guide which input to grab from the operation.
            Can be set to an integer representing where the variable is,
            a string representing a name or portion of the name of the variable,
            or the default: "from_trainable" which tries to find from
            the trainable vars in the graph
        """
        return self._param

    @param.setter
    def param(self, value: Union[int, str]):
        """
        :param value: The index to guide which input to grab from the operation.
            Can be set to an integer representing where the variable is,
            a string representing a name or portion of the name of the variable,
            or the default: "from_trainable" which tries to find from
            the trainable vars in the graph
        """
        self._param = value

    @ModifierProp()
    def leave_enabled(self) -> bool:
        """
        :return: True to continue masking the weights after end_epoch,
            False to stop masking. Should be set to False if exporting
            the result immediately after or doing some other prune
        """
        return self._leave_enabled

    @leave_enabled.setter
    def leave_enabled(self, value: bool):
        """
        :param value: True to continue masking the weights after end_epoch,
            False to stop masking. Should be set to False if exporting the result
            immediately after or doing some other prune
        """
        self._leave_enabled = value
        self.validate()

    @ModifierProp()
    def inter_func(self) -> str:
        """
        :return: The type of interpolation function to use:
            [linear, cubic, inverse_cubic]
        """
        return self._inter_func

    @inter_func.setter
    def inter_func(self, value: str):
        """
        :param value: The type of interpolation function to use:
            [linear, cubic, inverse_cubic]
        """
        self._inter_func = value
        self.validate()

    @ModifierProp(serializable=False)
    def ks_group(self) -> str:
        """
        :return: a hashed representation of the settings that identify this instance
        """
        props = self.props(only_serializable=True, format_str=False)
        props = ["{}={}".format(key, val) for key, val in props.items()]
        props.sort()
        props = "&".join(props)

        return "{}".format(hashlib.md5(bytes(props, encoding="utf8")).hexdigest())

    @ModifierProp(serializable=False)
    def exponent(self) -> float:
        """
        :return: the exponent to be used in for the sparsity schedule
        """

        if self._inter_func == "linear":
            return 1.0

        if self._inter_func == "cubic":
            return 3.0

        if self._inter_func == "inverse_cubic":
            return 1 / 3.0

        raise ValueError(
            "unrecognized value given for inter_func of {}".format(self._inter_func)
        )

    def create_ops(
        self,
        graph: tf_compat.Graph,
        steps_per_epoch: int,
        global_step: tf_compat.Variable,
    ) -> Tuple[tf_compat.Graph, List[tf_compat.Operation]]:
        """
        Create the sparsity ops to modify the training graph according to the settings
        for the current instance.

        :param graph: the graph to be modified
        :param steps_per_epoch: the number of steps (batches) per training epoch
        :param global_step: the global step used while training
        :return: a tuple containing the modified graph and extra ops
            to be run for modifying
        """
        # TODO: add support for tensorboard and tensorflow logging
        begin_step = round(self._start_epoch * steps_per_epoch)
        end_step = round(self._end_epoch * steps_per_epoch)
        update_step_freq = round(self._update_frequency * steps_per_epoch)
        update_ready, sparsity = get_or_create_ks_schedule_ops(
            begin_step,
            end_step,
            update_step_freq,
            self._init_sparsity,
            self._final_sparsity,
            self._exponent,
            global_step,
            self._ks_group,
        )
        mask_assign_ops, thresh_assign_ops = get_or_create_graph_ops_pruning(
            graph, self._layers, self._param, sparsity, self._ks_group
        )
        assign_ops = [*mask_assign_ops, *thresh_assign_ops]
        update_op = get_or_create_ks_update_op(update_ready, assign_ops, self._ks_group)

        return graph, [update_op]

    def complete_graph(self, graph: tf_compat.GraphDef) -> tf_compat.GraphDef:
        # TODO: fill out as needed for export to onnx support in the future
        return super().complete_graph(graph)

    def validate(self):
        """
        Validate the values of the params for the current instance are valid
        """

        if self._layers == ALL_TOKEN:
            raise ValueError(
                (
                    "layers cannot be set to {} for" " tensorflow implementation for {}"
                ).format(ALL_TOKEN, self.__class__.__name__)
            )

        if not self._leave_enabled:
            raise ValueError(
                "leave_enabled == True is only supported for {}".format(
                    self.__class__.__name__
                )
            )

        if not isinstance(self._init_sparsity, float):
            raise TypeError(
                "init_sparsity must be of float type for {}".format(
                    self.__class__.__name__
                )
            )

        if not 0.0 <= self._init_sparsity <= 1.0:
            raise ValueError(
                (
                    "init_sparsity value must be in the range"
                    " [0.0, 1.0], given {} for {}"
                ).format(self._init_sparsity, self.__class__.__name__)
            )

        if not isinstance(self._final_sparsity, float):
            raise TypeError(
                "final_sparsity must be of float type for {}".format(
                    self.__class__.__name__
                )
            )

        if not 0.0 <= self._final_sparsity <= 1.0:
            raise ValueError(
                (
                    "final_sparsity value must be in the range"
                    " [0.0, 1.0], given {} for {}"
                ).format(self._init_sparsity, self.__class__.__name__)
            )

        interpolation_funcs = ["linear", "cubic", "inverse_cubic"]

        if self._inter_func not in interpolation_funcs:
            raise ValueError(
                (
                    "{} is not a supported inter_func in layers_settings,"
                    " available are {} for {}"
                ).format(self._inter_func, interpolation_funcs, self.__class__.__name__)
            )
