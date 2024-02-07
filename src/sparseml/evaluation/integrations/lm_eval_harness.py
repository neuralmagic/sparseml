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
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from sparseml.evaluation.registry import SparseMLEvaluationRegistry
from sparseml.transformers.utils.sparse_model import SparseAutoModelForCausalLM
from sparsezoo.evaluation.results import Dataset, Evaluation, Metric, Result


try:
    from lm_eval import evaluator, tasks, utils
    from lm_eval.models.huggingface import HFLM
except ImportError as err:
    HFLM = object
    raise Exception(
        "package `lm_eval` is not installed. "
        "Please install it via "
        "`pip install lm-eval==0.4.0"
    ) from err

__all__ = ["lm_eval_harness", "SparseMLLM", "LMEvalHarnessEvaluatorInputSchema"]
_LOGGER = logging.getLogger(__name__)


@SparseMLEvaluationRegistry.register("lm-eval-harness")
def lm_eval_harness(
    model_path,
    datasets: str = "wikitext",
    batch_size: int = 1,
    device: str = "cuda",
    **kwargs,
) -> Result:
    """
    Run the lm-evaluation-harness on the given target model

    :param model-path: the target model to evaluate, can be path to
        a local model directory or a SparseZoo/Huggingface stub
    :param datasets: the datasets to evaluate on, can be a comma separated
        list of dataset names or a pattern to match against
    :param batch_size: the batch size to use for evaluation
    :param device: the device to use for evaluation
    :param kwargs: additional keyword arguments to pass to the
        lm-evaluation-harness
    """
    model = SparseMLLM(pretrained=model_path, device=device, **kwargs)
    kwargs.pop("nsamples", None)

    if kwargs.get("limit"):
        _LOGGER.warning(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. "
            "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )
    tasks.initialize_tasks()
    if datasets is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = utils.pattern_match(datasets.split(","), tasks.ALL_TASKS)

    _LOGGER.info(f"Selected Tasks: {task_names}")

    evaluator_input = LMEvalHarnessEvaluatorInputSchema(
        model=model,
        tasks=task_names,
        batch_size=batch_size,
        device=device,
        **kwargs,
    )

    results_raw = evaluator.simple_evaluate(**evaluator_input.dict())

    results = Result(
        raw=dict(output=results_raw, input=_filter_evaluator_input(evaluator_input)),
        formatted=_format_lm_eval_raw_results(results_raw),
    )

    return results


class LMEvalHarnessEvaluatorInputSchema(BaseModel):
    model: Any = Field(description="The name of the model.")
    tasks: List[str] = Field(
        description="The task (or multiple tasks) to evaluate the target on."
    )
    batch_size: int = Field(description="The batch size to use for evaluation.")
    model_args: str = Field(
        "", description="Additional arguments for the evaluated model."
    )
    num_fewshot: int = Field(0, description="The number of few shots to use.")
    max_batch_size: Optional[int] = Field(
        None, description="Maximal batch size to try with --batch_size auto."
    )
    device: Optional[str] = Field(None, description="Device to use for evaluation.")
    use_cache: Optional[str] = Field(
        None,
        description="A path to a sqlite db "
        "file for caching model responses. `None` if not caching.",
    )
    limit: Optional[float] = Field(
        None,
        description="Limit the number of examples per task. If <1, "
        "limit is a percentage of the total number of "
        "examples.",
    )
    decontamination_ngrams_path: Optional[str] = Field(
        None, description="Specify the path for decontamination n-grams."
    )
    check_integrity: bool = Field(
        False, description="Include this flag to check integrity."
    )
    write_out: bool = Field(False, description="Include this flag to write out.")


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


def _filter_evaluator_input(
    evaluator_input: "LMEvalHarnessEvaluatorInputSchema",
) -> Dict[str, Any]:  # noqa: F821
    """
    Filter the evaluator input to remove the model field.
    The model field is a complex object that cannot be serialized.

    :param evaluator_input: the evaluator input to filter
    :return: the filtered evaluator input
    """
    evaluator = evaluator_input.dict()
    del evaluator["model"]

    return evaluator


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
            if isinstance(metric_value, float)
        ]
        dataset = Dataset(
            type=None, name=dataset_name, config=results["config"], split=None
        )
        evaluation = Evaluation(
            task="lm_evaluation_harness",
            dataset=dataset,
            metrics=metrics,
            samples=None,
        )
        formatted_results.append(evaluation)
    return formatted_results
