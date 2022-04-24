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
Base Modifiers for inducing / enforcing kernel sparsity (model pruning)
on models while pruning.
"""


from typing import List, Union

from sparseml.optim.modifier import (
    BaseModifier,
    BaseScheduled,
    BaseUpdate,
    ModifierProp,
)
from sparseml.sparsification.types import SparsificationTypes
from sparseml.utils import FROM_PARAM_TOKEN, convert_to_bool, validate_str_iterable


__all__ = [
    "ConstantPruningModifier",
    "GMPruningModifier",
]


class ConstantPruningModifier(BaseModifier, BaseScheduled):
    """
    Base modifier for holding the sparsity level and shape for a given
    param constant while training. Useful for transfer learning use cases.

    | Sample yaml:
    |   !ConstantPruningModifier
    |       params: __ALL__
    |       start_epoch: 0.0
    |       end_epoch: 10.0

    :param params: List of str names or regex patterns of names for the parameter
        variables to apply the KS modifier to. Regex patterns must be specified
        with the prefix 're:'. Can also use the token __ALL__ to specify all
        prunable layers and weights
    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    """

    def __init__(
        self,
        params: Union[str, List[str]],
        start_epoch: float = -1,
        end_epoch: float = -1,
        **kwargs,
    ):
        kwargs["end_comparator"] = kwargs.get("end_comparator", None)
        super().__init__(start_epoch=start_epoch, end_epoch=end_epoch, **kwargs)

        self._params = validate_str_iterable(
            params, "{} for params".format(self.__class__.__name__)
        )  # type: List[str]

    @BaseModifier.sparsification_types.getter
    def sparsification_types(self) -> List[SparsificationTypes]:
        """
        :return: the sparsification types this modifier instance will apply
        """
        return [SparsificationTypes.pruning]

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
        self._params = validate_str_iterable(
            value, "{} for params".format(self.__class__.__name__)
        )


class GMPruningModifier(BaseModifier, BaseScheduled, BaseUpdate):
    """
    Base Modifier to gradually applies kernel sparsity to a given variable or variables
    from init_sparsity until final_sparsity is reached over a given amount of time and
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
    |       mask_type: unstructured
    |       leave_enabled: True

    :param params: List of str names or name regex patterns for the variables in the
        graph to apply the KS modifier to.  Regex patterns must be specified with
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
    :param mask_type: String to define type of sparsity (options: ['unstructured',
        'channel', 'filter']), List to define block shape of a parameter's in and out
        channels. default is 'unstructured'
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
        mask_type: Union[str, List[int]] = "unstructured",
        leave_enabled: bool = True,
        **kwargs,
    ):
        kwargs["min_frequency"] = kwargs.get("min_frequency", -1.0)
        super().__init__(
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=update_frequency,
            **kwargs,
        )
        self._params = validate_str_iterable(
            params, "{} for params".format(self.__class__.__name__)
        )  # type: List[str]
        self._init_sparsity = init_sparsity
        self._final_sparsity = final_sparsity
        self._leave_enabled = convert_to_bool(leave_enabled)
        self._inter_func = inter_func
        self._mask_type = mask_type
        if (
            isinstance(self._mask_type, List)
            and len(self._mask_type) > 2
            and all(i == 1 for i in self.mask_type[2:])
        ):
            self._mask_type = self._mask_type[:2]
        self._leave_enabled = convert_to_bool(leave_enabled)

        self.validate()

    @BaseModifier.sparsification_types.getter
    def sparsification_types(self) -> List[SparsificationTypes]:
        """
        :return: the sparsification types this modifier instance will apply
        """
        return [SparsificationTypes.pruning]

    @ModifierProp()
    def params(self) -> Union[str, List[str]]:
        """
        :return: List of str for the variable names or regex patterns of names
            to apply the KS modifier to. Regex patterns must be specified with
            the prefix 're:'.
        """
        return self._params

    @params.setter
    def params(self, value: Union[str, List[str]]):
        """
        :param value: List of str for the variable names or regex patterns of names
            to apply the KS modifier to. Regex patterns must be specified with
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
    def mask_type(self) -> Union[str, List[int]]:
        """
        :return: the mask type used
        """
        return self._mask_type

    @mask_type.setter
    def mask_type(self, value: Union[str, List[int]]):
        """
        :param value: the mask type to use
        """
        self._mask_type = value

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

        if not isinstance(self.init_sparsity, float) and (
            self.init_sparsity != FROM_PARAM_TOKEN
        ):
            raise TypeError(
                "init_sparsity must be of float type or be '{}' for {}".format(
                    self.__class__.__name__, FROM_PARAM_TOKEN
                )
            )

        if isinstance(self.init_sparsity, float) and (
            not 0.0 <= self.init_sparsity <= 1.0
        ):
            raise ValueError(
                (
                    "init_sparsity value must be in the range"
                    " [0.0, 1.0], given {} for {}"
                ).format(self.init_sparsity, self.__class__.__name__)
            )

        if not isinstance(self._final_sparsity, (float, List)):
            raise TypeError(
                "final_sparsity must be of float type for {}".format(
                    self.__class__.__name__
                )
            )

        final_sparsities = (
            self._final_sparsity
            if isinstance(self._final_sparsity, List)
            else [self._final_sparsity]
        )
        if not all(0.0 < sparsity < 1.0 for sparsity in final_sparsities):
            raise ValueError(
                (
                    "final_sparsity value(s) must be in the range (0.0, 1.0),"
                    " given {} for {}"
                ).format(self._final_sparsity, self.__class__.__name__)
            )

        interpolation_funcs = ["linear", "cubic", "inverse_cubic"]

        if self._inter_func not in interpolation_funcs:
            raise ValueError(
                (
                    "{} is not a supported inter_func in layers_settings,"
                    " available are {} for {}"
                ).format(self._inter_func, interpolation_funcs, self.__class__.__name__)
            )
