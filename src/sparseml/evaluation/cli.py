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
Command help: sparseml.eval --help                                                                                                                      (add-evaluator|✚3-2…1⚑1)
Usage: sparseml.eval [OPTIONS] [INTEGRATION_ARGS]...

  CLI utility for evaluating models on the various  supported evaluation
  integrations

Options:
  --target TEXT                   A SparseZoo stub or local directory
                                  containing torch model (including all the
                                  auxiliary files)  [required]
  -d, --dataset TEXT              The name of dataset to evaluate on. The user
                                  may pass multiple datasets names by passing
                                  the option multiple times.
  -i, --integration TEXT          Optional name of the evaluation integration
                                  to use. Must be a valid integration name
                                  that is registered in the evaluation
                                  registry
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
  --help                          Show this message and exit.

INTEGRATION_ARGS:
    Additional, unstructured arguments to pass to the evaluation integration.

#########
EXAMPLES
#########

##########
Example command for evaluating a quantized MPT model from SparseZoo.
The evaluation will be run using torch
sparseml.eval   --target zoo:mpt-7b-mpt_pretrain-base_quantized \
                --dataset hellaswag \
                --dataset gsm8k \
                --integration perlexity

"""  # noqa: E501
import logging
from pathlib import Path

import click
from sparseml.evaluation.evaluator import evaluate
from sparsezoo.evaluation.results import Result, save_result


_LOGGER = logging.getLogger(__name__)


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
    )
)
@click.option(
    "--target",
    type=str,
    required=True,
    help="A SparseZoo stub or local directory containing torch model "
    "(including all the auxiliary files)",
)
@click.option(
    "-d",
    "--dataset",
    type=str,
    multiple=True,
    help="The name of dataset to evaluate on. The user may pass multiple "
    "datasets names by passing the option multiple times.",
)
@click.option(
    "-i",
    "--integration",
    type=str,
    required=False,
    help="Optional name of the evaluation integration to use. Must be a valid "
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
@click.argument("integration_args", nargs=-1, type=click.UNPROCESSED)
def main(
    target,
    dataset,
    integration,
    save_path,
    type_serialization,
    batch_size,
    integration_args,
):
    """
    CLI utility for evaluating models on the various
    supported evaluation integrations
    """
    result: Result = evaluate(
        target=target,
        datasets=dataset,
        integration=integration,
        batch_size=batch_size,
        **integration_args,
    )
    _LOGGER.info(f"Evaluation done. Results:\n{result}")

    save_path: Path = Path(save_path) if save_path else Path.cwd()
    save_path = save_path / f"result.{type_serialization}"

    _LOGGER.info(f"Saving the evaluation results to {save_path}")
    save_result(result=result, save_path=save_path, save_format=type_serialization)


if __name__ == "__main__":
    main()
