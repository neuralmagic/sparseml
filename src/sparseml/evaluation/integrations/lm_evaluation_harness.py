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
from typing import Any, Dict, List, Union

from sparseml.evaluation.registry import SparseMLEvaluationRegistry
from sparsezoo.evaluation.results import Dataset, Evaluation, Metric, Result


try:
    from lm_eval import evaluator, tasks, utils
    from lm_eval.models.huggingface import HFLM
except ImportError as import_error:
    HFLM = object
    raise ImportError(
        "package `lm_eval` not found. Please install it via "
        "`pip install lm-eval==0.4.1;pip uninstall transformers &&"
        " pip install sparseml[transformers,torch]`"
    ) from import_error
try:
    # This needs to be imported after lm_eval to ensure right transformers
    # version is installed for SparseML
    from sparseml.transformers import SparseAutoTokenizer
    from sparseml.transformers.sparsification.sparse_config import SparseAutoConfig
    from sparseml.transformers.sparsification.sparse_model import (
        SparseAutoModelForCausalLM,
    )
except ImportError as import_error:
    raise import_error

__all__ = ["lm_eval_harness", "SparseMLLM"]
_LOGGER = logging.getLogger(__name__)

LM_EVALUATION_HARNESS: str = "lm-evaluation-harness"
LM_EVALUATION_HARNESS_ALIASES: List[str] = ["lm-eval-harness"]


@SparseMLEvaluationRegistry.register(
    LM_EVALUATION_HARNESS, alias=LM_EVALUATION_HARNESS_ALIASES
)
def lm_eval_harness(
    model_path,
    datasets: Union[str, List[str]] = "wikitext",
    batch_size: int = 1,
    **kwargs,
) -> Result:
    """
    Run the lm-evaluation-harness on the given target model

    :param model-path: the target model to evaluate, can be path to
        a local model directory or a SparseZoo/Huggingface stub
    :param datasets: the datasets to evaluate on, can be a string or
        list of strings, or a command separated string
    :param batch_size: the batch size to use for evaluation
    :param kwargs: additional keyword arguments to pass to the
        lm-evaluation-harness. For example, `limit`
    """
    kwargs["limit"] = int(limit) if (limit := kwargs.get("limit")) else None

    tokenizer = SparseAutoTokenizer.from_pretrained(model_path)
    model = SparseMLLM(pretrained=model_path, tokenizer=tokenizer, **kwargs)

    if kwargs.get("limit"):
        _LOGGER.warning(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. "
            "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )
    tasks.initialize_tasks()
    if datasets is None:
        task_names = tasks.ALL_TASKS
    else:
        datasets = datasets if isinstance(datasets, str) else ",".join(datasets)
        task_names = utils.pattern_match(datasets.split(","), tasks.ALL_TASKS)

    _LOGGER.info(f"Selected Tasks: {task_names}")

    results_raw = evaluator.simple_evaluate(
        model=model,
        tasks=task_names,
        batch_size=batch_size,
        **kwargs,
    )

    results = Result(
        raw=results_raw,
        formatted=_format_lm_eval_raw_results(results_raw),
    )

    return results


class SparseMLLM(HFLM):
    """
    SparseML is an open-source model optimization toolkit that enables you to create
    inference-optimized sparse models using pruning, quantization, and distillation
    algorithms. Models optimized with SparseML can then be exported to the ONNX and
    deployed with DeepSparse for GPU-class performance on CPU hardware.

    This class is a wrapper around the HuggingFace LM class to enable SparseML
    integration with the lm-evaluation-harness
    """

    def _create_model(
        self,
        pretrained: str,
        **kwargs,
    ) -> None:
        model_kwargs = kwargs if kwargs else {}
        relevant_kwarg_names = [
            "revision",
            "trust_remote_code",
            "offload_folder",
            "device",
        ]

        relevant_kwargs = {
            k: v for k, v in model_kwargs.items() if k in relevant_kwarg_names
        }

        model = SparseAutoModelForCausalLM.from_pretrained(
            pretrained, **relevant_kwargs
        )
        self._model = model

    def _get_config(self, pretrained: str, **kwargs) -> None:
        self._config = SparseAutoConfig.from_pretrained(
            pretrained_model_name_or_path=pretrained, **kwargs
        )


def _format_lm_eval_raw_results(results: Dict[str, Any]) -> List[Evaluation]:
    """
    Format the raw results from lm_evaluation_harness into a list of
    Evaluation objects.

    :param results: the raw results from lm_evaluation_harness
    :return: the formatted results as a list of Evaluation objects
    """
    formatted_results = []
    for dataset_name, dataset_result in results["results"].items():
        metrics = [
            Metric(name=metric_name, value=metric_value)
            for metric_name, metric_value in dataset_result.items()
            if isinstance(metric_value, (float, int))
        ]
        dataset = Dataset(
            type=None, name=dataset_name, config=results["config"], split=None
        )
        evaluation = Evaluation(
            task=LM_EVALUATION_HARNESS,
            dataset=dataset,
            metrics=metrics,
            samples=None,
        )
        formatted_results.append(evaluation)
    return formatted_results
