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
$ sparseml.eval --help
Usage: sparseml.eval [OPTIONS] [INTEGRATION_ARGS]...

Options:
  --model_path PATH               A path to a local directory containing
                                  pytorch model(including all the auxiliary
                                  files) or a SparseZoo/HugginFace stub
                                  [required]
  -d, --dataset TEXT              The name of dataset to evaluate on. The user
                                  may pass multiple datasets names by passing
                                  the option multiple times.
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
  --nsamples INTEGER              The number of samples to evaluate on. Must
                                  be greater than 0
  --help                          Show this message and exit.

INTEGRATION_ARGS:
    Additional, unstructured arguments to pass to the evaluation integration.

#########
EXAMPLES
#########

"""  # noqa: E501
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import click
from sparseml.evaluation.evaluator import evaluate
from sparsezoo.evaluation.results import Result, save_result


_LOGGER = logging.getLogger(__name__)


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
        token_normalize_func=lambda x: x.replace("-", "_"),
    )
)
@click.option(
    "--model_path",
    type=click.Path(dir_okay=True, file_okay=True),
    required=True,
    help="A path to a local directory containing pytorch model"
    "(including all the auxiliary files) or a SparseZoo/HugginFace stub",
)
@click.option(
    "-d",
    "--dataset",
    type=str,
    default=None,
    help="The name of dataset to evaluate on. The user may pass multiple "
    "datasets names by passing the option multiple times.",
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
    "--nsamples",
    type=int,
    default=None,
    help="The number of samples to evaluate on. Must be greater than 0",
)
@click.argument("integration_args", nargs=-1, type=click.UNPROCESSED)
def main(
    model_path,
    dataset,
    integration,
    save_path,
    type_serialization,
    batch_size,
    nsamples,
    integration_args,
):

    # format kwargs to a  dict
    integration_args = _args_to_dict(integration_args)

    _LOGGER.info(
        f"Datasets to evaluate on: {dataset}\n"
        f"Batch size: {batch_size}\n"
        f"Additional integration arguments supplied: {integration_args}"
    )

    result: Result = evaluate(
        model_path=model_path,
        datasets=dataset,
        integration=integration,
        batch_size=batch_size,
        nsamples=nsamples,
        **integration_args,
    )

    _LOGGER.info(f"Evaluation done. Results:\n{result}")

    save_path = (
        Path.cwd() / f"results.{type_serialization}"
        if not save_path
        else Path(save_path).absolute().with_suffix(f".{type_serialization}")
    )

    if save_path:
        _LOGGER.info(f"Saving the evaluation results to {save_path}")
        save_result(
            result=result, save_path=str(save_path), save_format=type_serialization
        )


def _args_to_dict(args: Tuple[Any, ...]) -> Dict[str, Any]:
    """
    Convert a tuple of args to a dict of args.

    :param args: The args to convert. Should be a tuple of alternating
        arg names and arg values e.g.('--arg1', 1, 'arg2', 2, -arg3', 3).
        The names can optionally have a '-' or `--` in front of them.
    :return: The converted args as a dict.
    """
    # Note: this function will ne moved to
    # nm_utils soon

    if len(args) == 0:
        return {}
    # names are uneven indices, values are even indices
    args_names = args[0::2]
    args_values = args[1::2]
    # remove any '-' or '--' from the names
    args_names = [name.lstrip("-") for name in args_names]

    return dict(zip(args_names, args_values))


if __name__ == "__main__":
    main()
