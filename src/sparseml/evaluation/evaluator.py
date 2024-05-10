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


from typing import Optional

from sparseml.core.session import create_session
from sparseml.evaluation.registry import SparseMLEvaluationRegistry
from sparsezoo.evaluation.results import Result


__all__ = ["evaluate"]


def evaluate(
    model_path: str,
    integration: str,
    datasets: Optional[str] = None,
    batch_size: int = 1,
    **kwargs,
) -> Result:
    """
    Evaluate a target model on a dataset using the specified integration.

    :param model_path: Path to the model folder or the model stub from
        SparseZoo/HuggingFace to evaluate. For example,
        `mgoin/llama2.c-stories15M-quant-pt`
    :param datasets: The dataset(s) to evaluate on. For example,
        `open_platypus`. If None, it is left upto the integration
        to handle the default dataset.
    :param integration: Name of the eval integration to use.
        Example, `perplexity`
    :param batch_size: The batch size to use for evals, defaults to 1
    :return: The evaluation result as a Result object
    """
    with create_session():
        eval_integration = SparseMLEvaluationRegistry.resolve(
            name=integration, datasets=datasets
        )

        if datasets is None:
            # let the integration handle the default dataset
            return eval_integration(
                model_path=model_path, batch_size=batch_size, **kwargs
            )

        return eval_integration(
            model_path=model_path, datasets=datasets, batch_size=batch_size, **kwargs
        )
