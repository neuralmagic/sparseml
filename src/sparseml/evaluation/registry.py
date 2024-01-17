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


from typing import Callable

from sparsezoo.evaluation import EvaluationRegistry
from sparsezoo.evaluation.results import Result


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


def collect_integrations():
    """
    This function is used to collect all integrations that are registered
    with the SparseML evaluation registry. This is done by importing the module
    associated with each integration.
    """
