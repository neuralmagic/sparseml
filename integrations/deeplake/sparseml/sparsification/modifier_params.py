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
Base Modifier for changing the state of a modules params while training according to
certain update formulas or patterns.
"""

from typing import List, Union

from sparseml.optim.modifier import BaseModifier, BaseScheduled, ModifierProp
from sparseml.sparsification.types import SparsificationTypes
from sparseml.utils import convert_to_bool, validate_str_iterable


__all__ = ["TrainableParamsModifier"]


class TrainableParamsModifier(BaseModifier, BaseScheduled):
    """
    Base Modifier to control the trainability for a given list of parameters.

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
    :param params_strict: True if the given param(s) must be found in each layer and
        will raise an err if not found,
        False if missing params are ok and will not raise an err
    :param start_epoch: The epoch to start the modifier at
        (set to -1.0 so it starts immediately)
    :param end_epoch: The epoch to end the modifier at (set to -1.0 so it never ends),
        if > 0 then will revert to the original value for the params after this epoch"""

    def __init__(
        self,
        params: Union[str, List[str]],
        trainable: bool,
        params_strict: bool = True,
        start_epoch: float = -1.0,
        end_epoch: float = -1.0,
        **kwargs,
    ):
        kwargs["end_comparator"] = kwargs.get("end_comparator", -1)
        super(TrainableParamsModifier, self).__init__(
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            **kwargs,
        )
        self._params = validate_str_iterable(
            params, "{} for params".format(self.__class__.__name__)
        )
        self._trainable = convert_to_bool(trainable)
        self._params_strict = convert_to_bool(params_strict)
        self._vars_to_trainable_orig = {}
        self.validate()

    @BaseModifier.sparsification_types.getter
    def sparsification_types(self) -> List[SparsificationTypes]:
        """
        :return: the sparsification types this modifier instance will apply
        """
        return [SparsificationTypes.general]

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
        self._params = validate_str_iterable(
            value, "{} for params".format(self.__class__.__name__)
        )
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
        :return: True if the given param(s) must be found in each layer and
            will raise an err if not found,
            False if missing params are ok and will not raise an err
        """
        return self._params_strict

    @params_strict.setter
    def params_strict(self, value: bool):
        """
        :param value: True if the given param(s) must be found in each layer and
            will raise an err if not found,
            False if missing params are ok and will not raise an err
        """
        self._params_strict = value
        self.validate()

    def validate(self):
        """
        Validate the values of the params for the current instance are valid,
        Should be overriden by framework specific implementations
        """
        pass
