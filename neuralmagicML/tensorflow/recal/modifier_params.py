"""
Modifier for changing the state of a module's params while training according to
certain update formulas or patterns.
"""

import re
from typing import Union, List, Tuple, Dict, Any

import tensorflow.contrib.graph_editor as ge

from neuralmagicML.utils import (
    ALL_TOKEN,
    convert_to_bool,
    flatten_iterable,
)
from neuralmagicML.tensorflow.utils import (
    tf_compat,
    get_prunable_ops,
    any_str_or_regex_matches_tensor_name,
)
from neuralmagicML.tensorflow.recal.modifier import (
    ModifierProp,
    TensorFlowModifierYAML,
    ScheduledModifier,
    EXTRAS_KEY_VAR_LIST,
)

__all__ = ["TrainableParamsModifier"]


@TensorFlowModifierYAML()
class TrainableParamsModifier(ScheduledModifier):
    """
    Modifier to control the params for a given list of parameters.  Applies the
    trainability over all epochs.
    To select all params in the graph, set to the ALL_TOKEN string: __ALL__

    | Sample yaml:
    |   !TrainableParamsModifier:
    |       params: ["conv_net/conv1/weight"]
    |       trainable: True

    :param params: A list of full parameter names or regex patterns of names to apply
        pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
        will match to all parameters. Can also use the token __ALL__ to specify all
        params
    :param trainable: True if the param(s) should be made trainable,
        False to make them non-trainable
    :param params_strict: True if the given param(s) must be found in each layer
        -- will raise an err if not found,
        False if missing params are ok -- will not raise an err
    """

    def __init__(
        self,
        params: Union[str, List[str]],
        trainable: bool,
        params_strict: bool = True,
        start_epoch: float = -1.0,
        end_epoch: float = -1.0,
    ):
        super(TrainableParamsModifier, self).__init__(
            start_epoch=-1, end_epoch=-1, end_comparator=-1,
        )
        self._params = self._validate_params(params)
        self._trainable = convert_to_bool(trainable)
        self._params_strict = convert_to_bool(params_strict)
        self._vars_to_trainable_orig = {}
        self.validate()

    def _validate_params(self, params: Union[str, List[Union[int, str]]]):
        """
        :param val: the value to validate, check that params is a list (and flattens it),
            otherwise checks that it's an ALL string, otherwise raises a ValueError
        :return: the validated version of the param
        """
        if isinstance(params, str):
            if params.upper() == ALL_TOKEN:
                return params.upper()

        if isinstance(params, List):
            return flatten_iterable(params)

        raise ValueError(
            "unsupported type ({}) given in {}".format(
                params, "{} for layers".format(self.__class__.__name__)
            )
        )

    @ModifierProp()
    def params(self) -> Union[str, List[str]]:
        """
        :return: A list of full parameter names or regex patterns of names to apply
        pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
        will match to all parameters. Can also use the token __ALL__ to specify all
        params
        """
        return self._params

    @params.setter
    def params(self, value: Union[str, List[str]]):
        """
        :param value: A list of full parameter names or regex patterns of names to apply
        pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
        will match to all parameters. Can also use the token __ALL__ to specify all
        params
        """
        self._params = self._validate_params(value)
        self.validate()

    @ModifierProp()
    def trainable(self) -> bool:
        """
        :return: True if the param(s) should be made trainable,
            False to make them non-trainable
        """
        return self._trainable

    @trainable.setter
    def trainable(self, value: bool):
        """
        :param value: True if the param(s) should be made trainable,
            False to make them non-trainable
        """
        self._trainable = value
        self.validate()

    @ModifierProp()
    def params_strict(self) -> bool:
        """
        :return: True if the given param(s) must be found in each layer
            -- will raise an err if not found.
            False if missing params are ok -- will not raise an err
        """
        return self._params_strict

    @params_strict.setter
    def params_strict(self, value: bool):
        """
        :param value: True if the given param(s) must be found in each layer
            -- will raise an err if not found.
            False if missing params are ok -- will not raise an err
        """
        self._params_strict = value

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
        trainable_vars_ref = tf_compat.get_collection_ref(
            tf_compat.GraphKeys.TRAINABLE_VARIABLES
        )
        match_params = self._params != ALL_TOKEN
        # Iterate through all variables to find names that match self._params
        all_variables = (
            tf_compat.global_variables()
            if match_params
            else tf_compat.trainable_variables()
        )
        for variable in tf_compat.global_variables():
            if match_params and not any_str_or_regex_matches_tensor_name(
                variable.name, self._params
            ):
                continue  # Skip any variables that do not match any of self._params
            is_var_trainable = variable in trainable_vars_ref
            # record original state of variable and modify trainable_vars_ref
            self._vars_to_trainable_orig[variable] = is_var_trainable
            if self._trainable and not is_var_trainable:
                trainable_vars_ref.append(variable)
            elif not self._trainable and is_var_trainable:
                trainable_vars_ref.remove(variable)
        if (
            not match_params
            and self._params_strict
            and len(self._vars_to_trainable_orig) < len(self._params)
        ):
            found_var_names = [
                v.name for v in list(self._vars_to_trainable_orig.keys())
            ]
            raise ValueError(
                "Could not find all required params: {}. Found: {}".format(
                    self._params, found_var_names
                )
            )

        # Populate var_list extra
        mod_extras[EXTRAS_KEY_VAR_LIST] = list(self._vars_to_trainable_orig.keys())
        return mod_ops, mod_extras

    def complete_graph(self, graph: tf_compat.Graph, sess: tf_compat.Session):
        """
        Complete modifying the graph.
        Resets the object's filtered variables to their original trainability

        :param graph: the modified graph that should be completed and cleaned.
            if not supplied, then will use the default graph
        :param sess: the session to use for completing the modified graph.
            if not supplied, then will use the default session
        """
        super().complete_graph(graph, sess)

        trainable_vars_ref = collection = tf_compat.get_collection_ref(
            tf_compat.GraphKeys.TRAINABLE_VARIABLES
        )
        for variable, was_trainable in self._vars_to_trainable_orig.items():
            if was_trainable and variable not in trainable_vars_ref:
                trainable_vars_ref.append(variable)
            elif not was_trainable and variable in trainable_vars_ref:
                trainable_vars_ref.remove(variable)

    def validate(self):
        """
        Validate the values of the params for the current instance are valid
        """

        if self._trainable and self._params == ALL_TOKEN:
            raise ValueError(
                "params == {} not supported when trainable == True"
                " please provide a list of parameter names instead".format(ALL_TOKEN,)
            )
