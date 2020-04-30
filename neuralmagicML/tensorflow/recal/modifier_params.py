"""
Modifier for changing the state of a module's params while training according to
certain update formulas or patterns.
"""

from typing import Union, List, Tuple, Dict, Any

import tensorflow.contrib.graph_editor as ge

from neuralmagicML.utils import (
    ALL_TOKEN,
    validate_str_iterable,
    convert_to_bool,
    flatten_iterable,
)
from neuralmagicML.tensorflow.utils import (
    tf_compat,
    get_prunable_ops,
    clean_tensor_name,
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
    Modifier to control the params for a given list of layers to include in the
    trainable variables group and var_list for optimizers.  Applies the
    trainability over all epochs.
    To set all params in the given layers, set to the ALL_TOKEN string: __ALL__
    To set all layers in the given module, set to the ALL_TOKEN string: __ALL__

    | Sample yaml:
    |   !TrainableParamsModifier:
    |       params:
    |           - weight
    |           - bias
    |       layers: __ALL__
    |       trainable: True

    :param params: int or str or list of them for the params to apply the trainable
        modifier to. Can be set to an integer representing where the variable is,
        a string representing a name or portion of the name of the variable.
        can also use the token __ALL__ to specify all params
    :param layers: str or list of str for the layers to apply the trainable modifier to
        can also use the token __ALL__ to specify all layers
    :param trainable: True if the param(s) should be made trainable,
        False to make them non-trainable
    :param params_strict: True if the given param(s) must be found in each layer
        -- will raise an err if not found,
        False if missing params are ok -- will not raise an err
    """

    def __init__(
        self,
        params: Union[str, List[Union[int, str]]],
        layers: Union[str, List[str]],
        trainable: bool,
        params_strict: bool = True,
        start_epoch: float = -1.0,
        end_epoch: float = -1.0,
    ):
        super(TrainableParamsModifier, self).__init__(
            start_epoch=-1, end_epoch=-1, end_comparator=-1,
        )
        self._params = self._validate_params(params)
        self._layers = validate_str_iterable(
            layers, "{} for layers".format(self.__class__.__name__)
        )
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
        :return: str or list of str for the params to apply the trainable modifier to.
            Can also use the token __ALL__ to specify all params
        """
        return self._params

    @params.setter
    def params(self, value: Union[str, List[str]]):
        """
        :param value: str or list of str for the params to apply the trainable modifier
            to.Can also use the token __ALL__ to specify all params
        """
        self._params = self._validate_params(value)
        self.validate()

    @ModifierProp()
    def layers(self) -> Union[str, List[str]]:
        """
        :return: str or list of str for the layers to apply the trainable modifier to.
            Can also use the token __ALL__ to specify all layers
        """
        return self._layers

    @layers.setter
    def layers(self, value: Union[str, List[str]]):
        """
        :param value: str or list of str for the layers to apply the trainable modifier
            to. Can also use the token __ALL__ to specify all layers
        """
        self._layers = validate_str_iterable(
            value, "{} for layers".format(self.__class__.__name__)
        )

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

        layers = (
            self._layers
            if self._layers != ALL_TOKEN
            else [op.name for op in graph.get_operations()]
        )
        include_all_params = self._params == ALL_TOKEN
        param_idx_set, param_name_set = set(), set()
        if not include_all_params:
            for param in set(self._params):
                if isinstance(param, int):
                    param_idx_set.add(param)
                else:
                    param_name_set.add(param)
        # Get cleaned names of all variables that pass this object's filters
        cleaned_var_names = []
        for layer in layers:
            num_vars_start = len(cleaned_var_names)
            op = graph.get_operation_by_name(layer)
            for idx, var in enumerate(ge.sgv(op).inputs):
                cleaned_var_name = clean_tensor_name(var.name)
                if (
                    include_all_params
                    or idx in param_idx_set
                    or any(param in cleaned_var_name for param in param_name_set)
                ):
                    cleaned_var_names.append(cleaned_var_name)
            # Perform params stricgt check
            if self._params_strict and not include_all_params:
                num_vars_found = len(cleaned_var_names) - num_vars_start
                if num_vars_found != len(self._params):
                    raise ValueError(
                        (
                            "Could not find all required params for layer {}"
                            "found {} required {} for {}"
                        ).format(
                            layer,
                            cleaned_var_names[-num_vars_found:],
                            self._params,
                            self.__class__.__name__,
                        )
                    )
        # process global variables according to which ones were found in the layers
        trainable_vars_ref = tf_compat.get_collection_ref(
            tf_compat.GraphKeys.TRAINABLE_VARIABLES
        )
        all_variables = tf_compat.global_variables()
        if include_all_params:  # reduce search space for ALL_TOKEN
            all_variables = tf_compat.trainable_variables()
        for var in all_variables:
            name = clean_tensor_name(var.name)
            if name not in cleaned_var_names:
                continue
            is_var_trainable = var in trainable_vars_ref
            # record original state of variable
            self._vars_to_trainable_orig[var] = is_var_trainable
            if self._trainable and not is_var_trainable:
                trainable_vars_ref.append(var)
            elif not self._trainable and is_var_trainable:
                trainable_vars_ref.remove(var)
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
