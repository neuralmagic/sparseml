"""
Script to run a pruning (kernel sparsity) performance sensitivity for a given ONNX.
This will impose kernel sparsity and then measure the layer by layer performance
changes due to the kernel sparsity.


##########
Command help:
usage: model_pruning_perf_sensitivity.py [-h] --onnx-file-path ONNX_FILE_PATH
                                         [--data-glob DATA_GLOB] --batch-size
                                         BATCH_SIZE [--num-cores NUM_CORES]
                                         [--iterations-per-check ITERATIONS_PER_CHECK]
                                         [--warmup-iterations-per-check WARMUP_ITERATIONS_PER_CHECK]
                                         [--json-path JSON_PATH]
                                         [--plot-path PLOT_PATH]

Run a pruning (kernel sparsity) performance sensitivity for a given ONNX

optional arguments:
  -h, --help            show this help message and exit
  --onnx-file-path ONNX_FILE_PATH
                        Path to the local onnx file to analyze
  --data-glob DATA_GLOB
                        Optional glob pattern to grab sample data files to
                        feed through the model, defaults to random
  --batch-size BATCH_SIZE
                        The batch size to run the analysis for
  --num-cores NUM_CORES
                        The number of physical cores to run the analysis on,
                        defaults to all physical cores available on the system
  --iterations-per-check ITERATIONS_PER_CHECK
                        The number of iterations to run for each performance
                        check / timing
  --warmup-iterations-per-check WARMUP_ITERATIONS_PER_CHECK
                        The number of warmup iterations to run for before
                        checking performance / timing
  --json-path JSON_PATH
                        Path to save the output json file to, defaults to save
                        next to the onnx-file-path
  --plot-path PLOT_PATH
                        Path to save the output visual plot file to, defaults
                        to save next to the onnx-file-path


##########
Example:
python scripts/onnx/model_pruning_perf_sensitivity.py \
    --onnx-file-path /PATH/TO/MODEL.onnx \
    --batch-size 1
"""

import argparse

from neuralmagicML import get_main_logger
from neuralmagicML.onnx.utils import DataLoader, max_available_cores
from neuralmagicML.onnx.recal import one_shot_ks_perf_sensitivity


LOGGER = get_main_logger()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a pruning (kernel sparsity) performance sensitivity "
        "for a given ONNX"
    )
    parser.add_argument(
        "--onnx-file-path",
        type=str,
        required=True,
        help="Path to the local onnx file to analyze",
    )
    parser.add_argument(
        "--data-glob",
        type=str,
        default=None,
        help="Optional glob pattern to grab sample data files to feed through the model"
        ", defaults to random",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="The batch size to run the analysis for",
    )
    parser.add_argument(
        "--num-cores",
        type=int,
        default=-1,
        help="The number of physical cores to run the analysis on, "
        "defaults to all physical cores available on the system",
    )
    parser.add_argument(
        "--iterations-per-check",
        type=int,
        default=10,
        help="The number of iterations to run for each performance check / timing",
    )
    parser.add_argument(
        "--warmup-iterations-per-check",
        type=int,
        default=3,
        help="The number of warmup iterations to run for before checking "
        "performance / timing",
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default=None,
        help="Path to save the output json file to, "
        "defaults to save next to the onnx-file-path",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default=None,
        help="Path to save the output visual plot file to, "
        "defaults to save next to the onnx-file-path",
    )

    return parser.parse_args()


def main(args):
    if args.data_glob is not None:
        LOGGER.info("creating data from glob {}".format(args.data_glob))
        data = DataLoader(data=args.data_glob, labels=None, batch_size=args.batch_size)
    else:
        LOGGER.info("creating random data...")
        data = DataLoader.from_model_random(
            args.onnx_file_path, batch_size=args.batch_size
        )

    num_cores = args.num_cores if args.num_cores > 0 else max_available_cores()

    LOGGER.info("running performance sensitivity...")
    sensitivity = one_shot_ks_perf_sensitivity(
        args.onnx_file_path,
        data,
        args.batch_size,
        num_cores,
        args.iterations_per_check,
        args.warmup_iterations_per_check,
    )

    json_path = (
        args.json_path
        if args.json_path is not None
        else "{}.perf-sensitivity_bs-{}_c-{}.json".format(
            args.onnx_file_path, args.batch_size, num_cores
        )
    )
    LOGGER.info("saving json to {}".format(json_path))
    sensitivity.save_json(json_path)
    LOGGER.info("saved json to {}".format(json_path))

    plot_path = (
        args.plot_path
        if args.plot_path is not None
        else "{}.perf-sensitivity_bs-{}_c-{}.png".format(
            args.onnx_file_path, args.batch_size, num_cores
        )
    )
    LOGGER.info("saving plot to {}".format(plot_path))
    sensitivity.plot(plot_path)
    LOGGER.info("saved plot to {}".format(plot_path))


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
