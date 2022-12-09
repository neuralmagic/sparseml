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

from abc import ABC, abstractmethod
from typing import Any


class BaseTransform(ABC):
    """
    Interface that all transforms must implement.
    A transform is an object that, when applied to a model,
    does a single modification to it
    and returns a modified model.
    """

    def __call__(self, model: Any) -> Any:
        return self.apply(model)

    def apply(self, model: Any) -> Any:
        """
        The logic for applying the transform to the model
        should follow the following steps:
        1. Validate the input model
        2. Apply the transform to the model
        3. Validate the resulting model and return it

        :param model: The input model to be
            validated, transformed and once again, validated
        :return: The transformed model
        """
        model = self.pre_validate(model)
        model = self.transform(model)
        model = self.post_validate(model)
        return model

    @abstractmethod
    def transform(self, model: Any) -> Any:
        """
        Logic for applying the transformation to the model.

        :param model: The input model to be transformed
        :return: The transformed model
        """
        raise NotImplementedError

    @abstractmethod
    def pre_validate(self, model: Any) -> Any:
        """
        Validate the input model before applying the transform

        :param model: The input model to be validated
        """
        raise NotImplementedError

    @abstractmethod
    def post_validate(self, model: Any) -> Any:
        """
        Validate the model after applying the transform

        :param model: The transformed model to be validated
        """
        raise NotImplementedError
