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
Modifiers for inducing / enforcing kernel sparsity (model pruning)
on models while pruning.
"""

import hashlib
from typing import Any, Dict, List, Tuple, Union

from sparseml.tensorflow_v1.optim.mask_creator_pruning import (
    PruningMaskCreator,
    load_mask_creator,
)
from sparseml.tensorflow_v1.optim.mask_pruning import (
    PruningOpVars,
    apply_op_vars_masks,
    create_ks_scheduled_constant_graph_ops,
    create_summaries_pruning,
    get_or_create_ks_scheduled_graph_ops,
)
from sparseml.tensorflow_v1.optim.modifier import (
    EXTRAS_KEY_SUMMARIES,
    ModifierProp,
    ScheduledModifier,
    ScheduledUpdateModifier,
    TensorFlowModifierYAML,
)
from sparseml.tensorflow_v1.utils import (
    clean_tensor_name,
    get_ops_and_inputs_by_name_or_regex,
    tf_compat,
)
from sparseml.utils import ALL_TOKEN, convert_to_bool, validate_str_iterable


__all__ = ["ConstantPruningModifier", "GMPruningModifier"]


@TensorFlowModifierYAML()
class ConstantPruningModifier(ScheduledModifier):
    """
    Holds the sparsity level and shape for a given param constant while training.
    Useful for transfer learning use cases.

    | Sample yaml:
    |   !ConstantPruningModifier
    |       params: __ALL__
    |       start_epoch: 0.0
    |       end_epoch: 10.0
    |       log_types: __ALL__

    :param params: List of str names or regex patterns of names for the parameter
        variables to apply the pruning modifier to. Regex patterns must be specified
        with the prefix 're:'. Can also use the token __ALL__ to specify all
        prunable layers and weights
    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param log_types: The loggers to allow the learning rate to be logged to,
        default is __ALL__
    """

    def __init__(
        self,
        params: Union[str, List[str]],
        start_epoch: float = -1,
        end_epoch: float = -1,
        log_types: Union[str, List[str]] = ALL_TOKEN,
    ):
        super(ConstantPruningModifier, self).__init__(
            log_types=log_types,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            end_comparator=None,
        )
        self._params = validate_str_iterable(
            params, "{} for params".format(self.__class__.__name__)
        )  # type: List[str]
        self._prune_op_vars = None
        self._update_ready = None
        self._sparsity = None

    @ModifierProp()
    def params(self) -> Union[str, List[str]]:
        """
        :return: List of str for the variable names or regex patterns of names
            to apply the pruning modifier to. Regex patterns must be specified with
            the prefix 're:'.
        """
        return self._params

    @params.setter
    def params(self, value: Union[str, List[str]]):
        """
        :param value: List of str for the variable names or regex patterns of names
            to apply the pruning modifier to. Regex patterns must be specified with
            the prefix 're:'.
        """
        self._params = value

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

    @property
    def prune_op_vars(self) -> Union[None, List[PruningOpVars]]:
        """
        :return: the created pruning op vars in the graph if create_ops has been called,
            else None
        """
        return self._prune_op_vars

    @property
    def update_ready(self):
        """
        :return: the created update_ready tensor for setting the pruning ops
            if create_ops has been called, else None
        """
        return self._update_ready

    @property
    def sparsity(self) -> Union[None, tf_compat.Tensor]:
        """
        :return: the created sparsity tensor for setting the pruning ops
            if create_ops has been called, else None
        """
        return self._sparsity

    def create_ops(
        self,
        steps_per_epoch: int,
        global_step: tf_compat.Tensor,
        graph: tf_compat.Graph,
    ) -> Tuple[List[Union[tf_compat.Tensor, tf_compat.Operation]], Dict[str, Any]]:
        """
        Create the sparsity ops to modify the training graph according to the settings
        for the current instance.

        :param steps_per_epoch: the number of steps (batches) per training epoch
        :param global_step: the global step used while training
        :param graph: the graph to be modified
        :return: a tuple (list of ops, dict of named ops / tensors)
            to be run or used for modifying the training process.
        """
        mod_ops, mod_extras = super().create_ops(graph, None, None)
        start_step, end_step = self.start_end_steps(steps_per_epoch, after_optim=True)

        params = (
            self._params
            if self._params != ALL_TOKEN
            else [
                clean_tensor_name(var.name)
                for _, var in
                # Have ALL_TOKEN match to all variable names for now
                get_ops_and_inputs_by_name_or_regex(["re:.*"], graph)
            ]
        )

        with graph.as_default():
            update_op, prune_op_vars = create_ks_scheduled_constant_graph_ops(
                graph,
                global_step,
                params,
                start_step,
                end_step,
                self.ks_group,
            )

            if self.log_types == ALL_TOKEN or "tensorboard" in self.log_types:
                mod_extras[EXTRAS_KEY_SUMMARIES] = create_summaries_pruning(
                    prune_op_vars
                )

        mod_ops.append(update_op)
        self._prune_op_vars = prune_op_vars
        # self._update_ready = tf_compat.constant(False, name="nm_update_ready")

        return mod_ops, mod_extras

    def initialize_session(self, sess: tf_compat.Session):
        """
        Initialize the mask variables for pruning.

        :param sess: the session to use for initializing
        """
        super().initialize_session(sess)
        masks = [op_vars.mask for op_vars in self._prune_op_vars]

        if masks:
            sess.run(tf_compat.variables_initializer(masks))

    def complete_graph(self, graph: tf_compat.Graph, sess: tf_compat.Session):
        """
        Complete modifying the graph.
        Resets the pruned op's variables using the created masks to zero out
        the pruned weights for saving.

        :param graph: the modified graph that should be completed and cleaned.
            if not supplied, then will use the default graph
        :param sess: the session to use for completing the modified graph.
            if not supplied, then will use the default session
        :return: the cleaned graph
        """
        super().complete_graph(graph, sess)

        with graph.as_default():
            apply_op_vars_masks(self.prune_op_vars, self.ks_group, sess)


@TensorFlowModifierYAML()
class GMPruningModifier(ScheduledUpdateModifier):
    """
    Gradually applies kernel sparsity to a given variable or variables from
    init_sparsity until final_sparsity is reached over a given amount of time and
    applied with an interpolated function for each step taken.

    Applies based on magnitude pruning without any structure to the pruning.

    | Sample yaml:
    |   !GMPruningModifier
    |       params: __ALL__
    |       init_sparsity: 0.05
    |       final_sparsity: 0.8
    |       start_epoch: 0.0
    |       end_epoch: 10.0
    |       update_frequency: 1.0
    |       inter_func: cubic
    |       log_types: __ALL__
    |       mask_type: unstructured
    |       leave_enabled: True

    :param params: List of str names or name regex patterns for the variables in the
        graph to apply the pruning modifier to.  Regex patterns must be specified with
        the prefix 're:'.  __ALL__ will match to all parameters.
    :param init_sparsity: The initial sparsity for the variable to
        start with at start_epoch
    :param final_sparsity: The final sparsity for the variable to end with at end_epoch
    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param update_frequency: The number of epochs or fraction of epochs to
        update at between start and end
    :param leave_enabled: True to continue masking the weights after end_epoch,
        False to stop masking. Should be set to False if exporting the result
        immediately after or doing some other prune
    :param inter_func: The type of interpolation function to use:
        [linear, cubic, inverse_cubic]
    :param log_types: The loggers to allow the learning rate to be logged to,
        default is __ALL__
    :param mask_type: String to define type of sparsity (options: ['unstructured',
        'channel', 'filter']), List to define block shape of a parameter's in and out
        channels, or a SparsityMaskCreator object. default is 'unstructured'
    :param leave_enabled: True to continue masking the weights after end_epoch,
        False to stop masking. Should be set to False if exporting the result
        immediately after or doing some other prune
    """

    def __init__(
        self,
        params: Union[str, List[str]],
        init_sparsity: float,
        final_sparsity: float,
        start_epoch: float,
        end_epoch: float,
        update_frequency: float,
        inter_func: str = "cubic",
        log_types: Union[str, List[str]] = ALL_TOKEN,
        mask_type: Union[str, List[int], PruningMaskCreator] = "unstructured",
        leave_enabled: bool = True,
    ):
        super(GMPruningModifier, self).__init__(
            log_types=log_types,
            start_epoch=start_epoch,
            min_start=-1.0,
            end_epoch=end_epoch,
            min_end=0.0,
            end_comparator=1,
            update_frequency=update_frequency,
            min_frequency=-1.0,
        )
        self._params = validate_str_iterable(
            params, "{} for params".format(self.__class__.__name__)
        )  # type: List[str]
        self._init_sparsity = init_sparsity
        self._final_sparsity = final_sparsity
        self._leave_enabled = convert_to_bool(leave_enabled)
        self._inter_func = inter_func
        self._mask_type = mask_type
        self._mask_creator = mask_type
        self._leave_enabled = convert_to_bool(leave_enabled)
        if not isinstance(mask_type, PruningMaskCreator):
            self._mask_creator = load_mask_creator(mask_type)
        self._prune_op_vars = None
        self._update_ready = None
        self._sparsity = None
        self._mask_initializer = None

        self.validate()

    @ModifierProp()
    def params(self) -> Union[str, List[str]]:
        """
        :return: List of str for the variable names or regex patterns of names
            to apply the pruning modifier to. Regex patterns must be specified with
            the prefix 're:'.
        """
        return self._params

    @params.setter
    def params(self, value: Union[str, List[str]]):
        """
        :param value: List of str for the variable names or regex patterns of names
            to apply the pruning modifier to. Regex patterns must be specified with
            the prefix 're:'.
        """
        self._params = value
        self.validate()

    @ModifierProp()
    def init_sparsity(self) -> float:
        """
        :return: The initial sparsity for the variable to start with at start_epoch
        """
        return self._init_sparsity

    @init_sparsity.setter
    def init_sparsity(self, value: float):
        """
        :param value: The initial sparsity for the variable to start with at start_epoch
        """
        self._init_sparsity = value
        self.validate()

    @ModifierProp()
    def final_sparsity(self) -> float:
        """
        :return: The final sparsity for the variable to end with at end_epoch
        """
        return self._final_sparsity

    @final_sparsity.setter
    def final_sparsity(self, value: float):
        """
        :param value: The final sparsity for the variable to end with at end_epoch
        """
        self._final_sparsity = value
        self.validate()

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

    @ModifierProp()
    def mask_type(self) -> Union[str, List[int], PruningMaskCreator]:
        """
        :return: the SparsityMaskCreator object used
        """
        return self._mask_type

    @mask_type.setter
    def mask_type(self, value: Union[str, List[int], PruningMaskCreator]):
        """
        :param value: the SparsityMaskCreator object to use
        """
        self._mask_type = value
        self._mask_creator = value
        if not isinstance(value, PruningMaskCreator):
            self._mask_creator = load_mask_creator(value)

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

    @property
    def prune_op_vars(self) -> Union[None, List[PruningOpVars]]:
        """
        :return: the created pruning op vars in the graph if create_ops has been called,
            else None
        """
        return self._prune_op_vars

    @property
    def update_ready(self):
        """
        :return: the created update_ready tensor for setting the pruning ops
            if create_ops has been called, else None
        """
        return self._update_ready

    @property
    def sparsity(self) -> Union[None, tf_compat.Tensor]:
        """
        :return: the created sparsity tensor for setting the pruning ops
            if create_ops has been called, else None
        """
        return self._sparsity

    def create_ops(
        self,
        steps_per_epoch: int,
        global_step: tf_compat.Tensor,
        graph: tf_compat.Graph,
    ) -> Tuple[List[Union[tf_compat.Tensor, tf_compat.Operation]], Dict[str, Any]]:
        """
        Create the sparsity ops to modify the training graph according to the settings
        for the current instance.

        :param steps_per_epoch: the number of steps (batches) per training epoch
        :param global_step: the global step used while training
        :param graph: the graph to be modified
        :return: a tuple (list of ops, dict of named ops / tensors)
            to be run or used for modifying the training process.
        """
        mod_ops, mod_extras = super().create_ops(graph, steps_per_epoch, global_step)
        start_step, end_step = self.start_end_steps(steps_per_epoch, after_optim=True)
        update_frequency_step = self.update_frequency_steps(steps_per_epoch)
        params = (
            self._params
            if self._params != ALL_TOKEN
            else [
                clean_tensor_name(var.name)
                for _, var in
                # Have ALL_TOKEN match to all variable names for now
                get_ops_and_inputs_by_name_or_regex(["re:.*"], graph)
            ]
        )

        with graph.as_default():
            (
                update_op,
                prune_op_vars,
                update_ready,
                sparsity,
            ) = get_or_create_ks_scheduled_graph_ops(
                graph,
                global_step,
                params,
                start_step,
                end_step,
                update_frequency_step,
                self._init_sparsity,
                self._final_sparsity,
                self.exponent,
                self._leave_enabled,
                self.ks_group,
                self._mask_creator,
            )

            if self.log_types == ALL_TOKEN or "tensorboard" in self.log_types:
                mod_extras[EXTRAS_KEY_SUMMARIES] = create_summaries_pruning(
                    prune_op_vars
                )

        mod_ops.append(update_op)
        self._prune_op_vars = prune_op_vars
        self._update_ready = update_ready
        self._sparsity = sparsity

        # Create and cache the mask initializers to be run
        # through initialize_session. When using the estimator,
        # the initialization is done as part of the init_fn of
        # the training scaffold object, at which the graph cannot
        # be changed (hence the creation and caching)
        masks = [op_vars.mask for op_vars in self._prune_op_vars]
        self._mask_initializer = (
            tf_compat.variables_initializer(masks) if masks else None
        )

        return mod_ops, mod_extras

    def initialize_session(self, sess: tf_compat.Session):
        """
        Initialize the mask variables for pruning.

        :param sess: the session to use for initializing
        """
        super().initialize_session(sess)
        if self._mask_initializer:
            sess.run(self._mask_initializer)

    def complete_graph(self, graph: tf_compat.Graph, sess: tf_compat.Session):
        """
        Complete modifying the graph.
        Resets the pruned op's variables using the created masks to zero out
        the pruned weights for saving.

        :param graph: the modified graph that should be completed and cleaned.
            if not supplied, then will use the default graph
        :param sess: the session to use for completing the modified graph.
            if not supplied, then will use the default session
        :return: the cleaned graph
        """
        super().complete_graph(graph, sess)

        with graph.as_default():
            apply_op_vars_masks(self.prune_op_vars, self.ks_group, sess)

    def validate(self):
        """
        Validate the values of the params for the current instance are valid
        """

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
