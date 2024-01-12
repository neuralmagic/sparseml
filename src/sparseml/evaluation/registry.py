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


import logging
from typing import Callable

from sparsezoo.evaluation import EvaluationRegistry
from sparsezoo.evaluation.results import Result


__all__ = ["SparseMLEvaluationRegistry", "collect_integrations"]

_LOGGER = logging.getLogger(__name__)


class SparseMLEvaluationRegistry(EvaluationRegistry):
    """
    This class is used to register and retrieve evaluation integrations for
    SparseML. It is a subclass of the SparseZoo EvaluationRegistry class.
    """

    @classmethod
    def resolve(cls, name: str, *args, **kwargs) -> Callable[..., Result]:
        """
        Resolve an evaluation integration by name.

        :param name: The name of the evaluation integration to resolve
        :param args: The arguments to pass to the evaluation integration
        :param kwargs: The keyword arguments to pass to the evaluation integration
        :return: The evaluation integration associated with the name
        """

        collect_integrations()
        return cls.get_value_from_registry(name=name)


def collect_integrations(name: str):
    """
    This function is used to collect integrations based on name, this method
    is responsible for triggering the registration with the SparseML Evaluation
    registry.

    This is done by importing the module associated with each integration.
    This function is called automatically when the registry is accessed.
    The specific integration(s) `callable` must be decorated with the
    `@SparseMLEvaluationRegistry.register` decorator.

    :param name: The name of the integration to collect, is case insentitive
    """

    # The integration locations must be hardcoded here to be collected
    # this is a time being solution until we move to a config based
    # solution

    _LOGGER.debug(
        f"Auto collection of {name} integration for eval failed. "
        "The integration must be registered and collected/imported "
        "manually."
    )
