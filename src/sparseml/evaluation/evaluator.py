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
from typing import Any, List, Optional, Union

from sparsezoo.evaluation import EvaluationRegistry
from sparsezoo.evaluation.results import Result


__all__ = ["evaluate"]
_LOGGER = logging.getLogger(__name__)


def evaluate(
    target: Any,
    datasets: Union[str, List[str]],
    integration: Optional[str] = None,
    batch_size: int = 1,
    **kwargs,
) -> Result:

    collect_integrations()
    eval_integration = EvaluationRegistry.get_value_from_registry(name=integration)
    _LOGGER.info(
        f"Starting evaluation with target {target}, datasets {datasets}, "
        f"integration {eval_integration}, batch_size {batch_size}, kwargs {kwargs}"
    )
    return eval_integration(target, datasets, batch_size, **kwargs)


def collect_integrations():
    """
    This function is used to collect all integrations that are registered
    with the evaluation registry. This is done by importing the module
    associated with each integration.
    """
