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
######
Command help:
$ sparseml.evaluate --help
Usage: sparseml.evaluate [OPTIONS] MODEL_PATH [INTEGRATION_ARGS]...

  Evaluate a model using a specified integration.

  Where model is path to a local directory containing pytorch model (including
  all the auxiliary files) or a SparseZoo/HugginFace stub

Options:
  -d, --datasets TEXT              The name of dataset to evaluate on
  -i, --integration TEXT          Name of the evaluation integration to use.
                                  Must be a valid integration name that is
                                  registered in the evaluation registry
                                  [required]
  -s, --save_path TEXT            The path to save the evaluation results. The
                                  results will be saved under the name
                                  'result.yaml`/'result.json' depending on the
                                  serialization type. If argument is not
                                  provided, the results will be saved in the
                                  current directory
  -t, --type_serialization [yaml|json]
                                  The serialization type to use save the
                                  evaluation results. The default is json
  -b, --batch_size INTEGER        The batch size to use for the evaluation.
                                  Must be greater than 0
  --limit INTEGER              The number of samples to evaluate on. Must
                                  be greater than 0
  --help                          Show this message and exit.

INTEGRATION_ARGS:
    Additional, unstructured arguments to pass to the evaluation integration.
#########
EXAMPLES
#########
1. Use A Huggingface stub with lm-evaluation_harness
    sparseml.evaluate \
        "mgoin/llama2.c-stories15M-quant-pt" \
        -d hellaswag -i lm-evaluation-harness

"""  # noqa: E501
import logging
from pathlib import Path
from typing import Any, Dict

import click
from sparseml.evaluation.evaluator import evaluate
from sparseml.utils import parse_kwarg_tuples
from sparsezoo.evaluation.results import Result, save_result


_LOGGER = logging.getLogger(__name__)


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
        token_normalize_func=lambda x: x.replace("-", "_"),
    )
)
@click.argument(
    "model_path", type=click.Path(dir_okay=True, file_okay=True), required=True
)
@click.option(
    "-d",
    "--datasets",
    type=str,
    default=None,
    help="The name of dataset to evaluate on",
)
@click.option(
    "-i",
    "--integration",
    type=str,
    required=True,
    help="Name of the evaluation integration to use. Must be a valid "
    "integration name that is registered in the evaluation registry",
)
@click.option(
    "-s",
    "--save_path",
    type=click.UNPROCESSED,
    default=None,
    help="The path to save the evaluation results. The results will "
    "be saved under the name 'result.yaml`/'result.json' depending on "
    "the serialization type. If argument is not provided, the results "
    "will be saved in the current directory",
)
@click.option(
    "-t",
    "--type_serialization",
    type=click.Choice(["yaml", "json"]),
    default="json",
    help="The serialization type to use save the evaluation results. "
    "The default is json",
)
@click.option(
    "-b",
    "--batch_size",
    type=int,
    default=1,
    help="The batch size to use for the evaluation. Must be greater than 0",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="The number of samples to evaluate on. Must be greater than 0",
)
@click.argument("integration_args", nargs=-1, type=click.UNPROCESSED)
def main(
    model_path,
    datasets,
    integration,
    save_path,
    type_serialization,
    batch_size,
    limit,
    integration_args,
):
    """
    Evaluate a model using a specified integration.

    Where model is path to a local directory
    containing pytorch model (including all the
    auxiliary files) or a SparseZoo/HugginFace stub
    """

    # format kwargs to a  dict
    integration_args: Dict[str, Any] = parse_kwarg_tuples(integration_args)

    _LOGGER.info(
        f"Datasets to evaluate on: {datasets}\n"
        f"Batch size: {batch_size}\n"
        f"Additional integration arguments supplied: {integration_args}"
    )

    result: Result = evaluate(
        model_path=model_path,
        datasets=datasets,
        integration=integration,
        batch_size=batch_size,
        limit=limit,
        **integration_args,
    )

    _LOGGER.info(f"Evaluation done. Results:\n{result}")

    save_path = (
        Path.cwd().absolute() / f"results.{type_serialization}"
        if not save_path
        else Path(save_path).absolute().with_suffix(f".{type_serialization}")
    )

    if save_path:
        _LOGGER.info(f"Saving the evaluation results to {save_path}")
        save_result(
            result=result, save_path=str(save_path), save_format=type_serialization
        )


if __name__ == "__main__":
    main()
