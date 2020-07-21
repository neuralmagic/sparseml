"""
Script to run a pruning (kernel sparsity) loss sensitivity for a given ONNX.
This will impose kernel sparsity and then measure the affect on the loss
or approximate the affect on the loss.


##########
Command help:
usage: model_pruning_loss_sensitivity.py [-h] {approximate,one_shot} ...

Run a pruning (kernel sparsity) loss sensitivity for a given ONNX

positional arguments:
  {approximate,one_shot}

optional arguments:
  -h, --help            show this help message and exit


##########
approximate command help:
usage: model_pruning_loss_sensitivity.py approximate [-h] --onnx-file-path
                                                     ONNX_FILE_PATH
                                                     [--json-path JSON_PATH]
                                                     [--plot-path PLOT_PATH]

Approximate the effect of pruning on the loss by using the weight magnitude
approach

optional arguments:
  -h, --help            show this help message and exit
  --onnx-file-path ONNX_FILE_PATH
                        Path to the local onnx file to analyze
  --json-path JSON_PATH
                        Path to save the output json file to, defaults to save
                        next to the onnx-file-path
  --plot-path PLOT_PATH
                        Path to save the output visual plot file to, defaults
                        to save next to the onnx-file-path


##########
one_shot command help
usage: model_pruning_loss_sensitivity.py one_shot [-h] --onnx-file-path
                                                  ONNX_FILE_PATH
                                                  [--json-path JSON_PATH]
                                                  [--plot-path PLOT_PATH]
                                                  --data-glob DATA_GLOB
                                                  --batch-size BATCH_SIZE
                                                  [--steps-per-measurement STEPS_PER_MEASUREMENT]

Use a one shot approach (no retraining) for estimating the effect of pruning
on the loss

optional arguments:
  -h, --help            show this help message and exit
  --onnx-file-path ONNX_FILE_PATH
                        Path to the local onnx file to analyze
  --json-path JSON_PATH
                        Path to save the output json file to, defaults to save
                        next to the onnx-file-path
  --plot-path PLOT_PATH
                        Path to save the output visual plot file to, defaults
                        to save next to the onnx-file-path
  --data-glob DATA_GLOB
                        glob pattern to grab sample data files to feed through
                        the model, sample data must be numpy files with one
                        model input per file as either an array in an npy file
                        or dictionary in an npz file
  --batch-size BATCH_SIZE
                        The batch size to run the analysis for
  --steps-per-measurement STEPS_PER_MEASUREMENT
                        The number of steps (batches) to run for each sparsity
                        measurement


##########
approximate example:
python scripts/onnx/model_pruning_loss_sensitivity.py approximate \
    --onnx-file-path /PATH/TO/MODEL.onnx


##########
one shot example:
python scripts/onnx/model_pruning_loss_sensitivity.py one_shot \
    --onnx-file-path /PATH/TO/MODEL.onnx \
    --data-glob /PATH/TO/DATA/*.npz \
    --batch-size 16
    --steps-per-measurement 10
"""

import argparse

from neuralmagicML import get_main_logger
from neuralmagicML.onnx.utils import DataLoader
from neuralmagicML.onnx.recal import (
    approx_ks_loss_sensitivity,
    one_shot_ks_loss_sensitivity,
)


LOGGER = get_main_logger()
APPROXIMATE_COMMAND = "approximate"
ONE_SHOT_COMMAND = "one_shot"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a pruning (kernel sparsity) loss sensitivity for a given ONNX"
    )

    subparsers = parser.add_subparsers(dest="command")

    approximate_parser = subparsers.add_parser(
        APPROXIMATE_COMMAND,
        description="Approximate the effect of pruning on the loss by using "
        "the weight magnitude approach",
    )
    one_shot_parser = subparsers.add_parser(
        ONE_SHOT_COMMAND,
        description="Use a one shot approach (no retraining) for "
        "estimating the effect of pruning on the loss",
    )

    for par in [approximate_parser, one_shot_parser]:
        par.add_argument(
            "--onnx-file-path",
            type=str,
            required=True,
            help="Path to the local onnx file to analyze",
        )
        par.add_argument(
            "--json-path",
            type=str,
            default=None,
            help="Path to save the output json file to, "
            "defaults to save next to the onnx-file-path",
        )
        par.add_argument(
            "--plot-path",
            type=str,
            default=None,
            help="Path to save the output visual plot file to, "
            "defaults to save next to the onnx-file-path",
        )

    one_shot_parser.add_argument(
        "--data-glob",
        type=str,
        required=True,
        help="glob pattern to grab sample data files to feed through the model, "
        "sample data must be numpy files with one model input per file "
        "as either an array in an npy file or dictionary in an npz file",
    )
    one_shot_parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="The batch size to run the analysis for",
    )
    one_shot_parser.add_argument(
        "--steps-per-measurement",
        type=int,
        default=5,
        help="The number of steps (batches) to run for each sparsity measurement",
    )

    return parser.parse_args()


def main(args):
    if args.command == APPROXIMATE_COMMAND:
        LOGGER.info("running approximated loss sensitivity...")
        sensitivity = approx_ks_loss_sensitivity(args.onnx_file_path)
        save_key = "loss_approx"
    elif args.command == ONE_SHOT_COMMAND:
        LOGGER.info("creating data...")
        if args.data_glob is not None:
            data = DataLoader(
                data=args.data_glob,
                labels=None,
                batch_size=args.batch_size,
                iter_steps=-1,
            )
        else:
            data = DataLoader.from_model_random(
                args.onnx_file_path, batch_size=args.batch_size, iter_steps=-1
            )

        LOGGER.info("running one shot loss sensitivity...")
        sensitivity = one_shot_ks_loss_sensitivity(
            args.onnx_file_path, data, args.batch_size, args.steps_per_measurement
        )
        save_key = "loss_one_shot"
    else:
        raise ValueError("unknown command given of {}".format(args.command))

    json_path = (
        args.json_path
        if args.json_path is not None
        else "{}.{}-sensitivity.json".format(args.onnx_file_path, save_key)
    )
    LOGGER.info("saving json to {}".format(json_path))
    sensitivity.save_json(json_path)
    LOGGER.info("saved json to {}".format(json_path))

    plot_path = (
        args.plot_path
        if args.plot_path is not None
        else "{}.{}-sensitivity.png".format(args.onnx_file_path, save_key)
    )
    LOGGER.info("saving plot to {}".format(plot_path))
    sensitivity.plot(plot_path, plot_integral=True, normalize=False)
    LOGGER.info("saved plot to {}".format(plot_path))


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
