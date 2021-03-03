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
Modifier for changing the state of a modules params while training according to
certain update formulas or patterns.
"""

from typing import List, Union

import tensorflow

from sparseml.keras.optim.modifier import (
    KerasModifierYAML,
    ModifierProp,
    ScheduledModifier,
)
from sparseml.keras.optim.utils import get_layer_name_from_param
from sparseml.utils import ALL_TOKEN, convert_to_bool, flatten_iterable


__all__ = ["TrainableParamsModifier"]


class TrainableParamsCallback(tensorflow.keras.callbacks.Callback):
    def __init__(self, model, optimizer, layers, trainable, start_step, end_step):
        self.model = model
        self.optimizer = optimizer
        self.layers = layers
        self.prev_trainables = [layer.trainable for layer in self.layers]
        self.trainable = trainable
        self.start_step = start_step
        self.end_step = end_step
        self.step = None

    def on_train_begin(self, logs=None):
        self.step = tensorflow.keras.backend.get_value(self.optimizer.iterations)

    def on_train_batch_begin(self, batch, logs=None):
        if self.step == self.start_step:
            for layer in self.layers:
                layer.trainable = self.trainable
        if self.step == self.end_step:
            assert self.end_step > -1
            for idx, layer in enumerate(self.layers):
                layer.trainable = self.prev_trainables[idx]

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1


@KerasModifierYAML()
class TrainableParamsModifier(ScheduledModifier):
    """
    Modifier to control the params for a given list of parameters.
    Applies the trainability over all epochs.
    To select all params in the graph, set to the ALL_TOKEN string: __ALL__

    | Sample yaml:
    |   !TrainableParamsModifier:
    |       params: ["conv2d_1/kernel:0", "conv2d_5/kernel:0"]
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
            start_epoch=-1,
            end_epoch=-1,
            end_comparator=-1,
        )
        self._params = self._validate_params(params)
        self._layer_names = [get_layer_name_from_param(p) for p in self._params]
        self._trainable = convert_to_bool(trainable)
        self._params_strict = convert_to_bool(params_strict)
        self._vars_to_trainable_orig = {}
        self.validate()

    def _validate_params(self, params: Union[str, List[Union[int, str]]]):
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

    @property
    def layer_names(self) -> List[str]:
        return self._layer_names

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

    def validate(self):
        """
        Validate the values of the params for the current instance are valid
        """

        if self._trainable and self._params == ALL_TOKEN:
            raise ValueError(
                "params == {} not supported when trainable == True"
                " please provide a list of parameter names instead".format(
                    ALL_TOKEN,
                )
            )

    def modify(
        self,
        model,
        optimizer,
        steps_per_epoch: int,
        input_tensors: tensorflow.Tensor = None,
    ):
        model, optimizer, callback = super(TrainableParamsModifier, self).modify(
            model, optimizer, steps_per_epoch, input_tensors=input_tensors
        )
        start_step, end_step = self.start_end_steps(steps_per_epoch, after_optim=False)
        layers = [layer for layer in model.layers if layer.name in self.layer_names]
        trainable_param_callback = TrainableParamsCallback(
            model, optimizer, layers, self.trainable, start_step, end_step
        )
        return model, optimizer, trainable_param_callback
