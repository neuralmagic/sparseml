"""
Script to benchmark an ONNX model in Neural Magic or ONNXRuntime inference systems.
By default will use random data. Real data can be supplied as numpy files.


##########
Command help:
usage: model_benchmark.py [-h] {neuralmagic,onnxruntime} ...

Benchmark the inference for an ONNX model

positional arguments:
  {neuralmagic,onnxruntime}

optional arguments:
  -h, --help            show this help message and exit


##########
neuralmagic command help:
usage: model_benchmark.py neuralmagic [-h] --onnx-file-path ONNX_FILE_PATH
                                      [--data-glob DATA_GLOB] --batch-size
                                      BATCH_SIZE [--num-cores NUM_CORES]
                                      [--iterations-per-check ITERATIONS_PER_CHECK]
                                      [--warmup-iterations-per-check WARMUP_ITERATIONS_PER_CHECK]

Run benchmarks in the Neural Magic inference engine

optional arguments:
  -h, --help            show this help message and exit
  --onnx-file-path ONNX_FILE_PATH
                        Path to the local onnx file to analyze
  --data-glob DATA_GLOB
                        Optional glob pattern to grab sample data files to
                        feed through the model, sample data must be numpy
                        files with one model input per file as either an array
                        in an npy file or dictionary in an npz file, defaults
                        to random
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


##########
onnxruntime command help:
usage: model_benchmark.py onnxruntime [-h] --onnx-file-path ONNX_FILE_PATH
                                      [--data-glob DATA_GLOB] --batch-size
                                      BATCH_SIZE
                                      [--iterations-per-check ITERATIONS_PER_CHECK]
                                      [--warmup-iterations-per-check WARMUP_ITERATIONS_PER_CHECK]
                                      [--no-batch-override]

Run benchmarks in onnxruntime

optional arguments:
  -h, --help            show this help message and exit
  --onnx-file-path ONNX_FILE_PATH
                        Path to the local onnx file to analyze
  --data-glob DATA_GLOB
                        Optional glob pattern to grab sample data files to
                        feed through the model, sample data must be numpy
                        files with one model input per file as either an array
                        in an npy file or dictionary in an npz file, defaults
                        to random
  --batch-size BATCH_SIZE
                        The batch size to run the analysis for
  --iterations-per-check ITERATIONS_PER_CHECK
                        The number of iterations to run for each performance
                        check / timing
  --warmup-iterations-per-check WARMUP_ITERATIONS_PER_CHECK
                        The number of warmup iterations to run for before
                        checking performance / timing
  --no-batch-override   Do not override batch dimension of the ONNX model
                        before running in ORT


##########
neuralmagic example:
python scripts/onnx/model_benchmark.py neuralmagic \
    --onnx-file-path /PATH/TO/MODEL.onnx \
    --batch-size 1


##########
onnxruntime example:
python scripts/onnx/model_benchmark.py onnxruntime \
    --onnx-file-path /PATH/TO/MODEL.onnx \
    --batch-size 1
"""

import argparse
import numpy

from neuralmagicML import get_main_logger
from neuralmagicML.onnx.utils import DataLoader, NMModelRunner, ORTModelRunner


LOGGER = get_main_logger()
NEURALMAGIC_COMMAND = "neuralmagic"
ONNXRUNTIME_COMMAND = "onnxruntime"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark the inference for an ONNX model"
    )

    subparsers = parser.add_subparsers(dest="command")

    neuralmagic_parser = subparsers.add_parser(
        NEURALMAGIC_COMMAND,
        description="Run benchmarks in the Neural Magic inference engine",
    )
    onnxruntime_parser = subparsers.add_parser(
        ONNXRUNTIME_COMMAND, description="Run benchmarks in onnxruntime",
    )

    for index, par in enumerate([neuralmagic_parser, onnxruntime_parser]):
        par.add_argument(
            "--onnx-file-path",
            type=str,
            required=True,
            help="Path to the local onnx file to analyze",
        )
        par.add_argument(
            "--data-glob",
            type=str,
            default=None,
            help="Optional glob pattern to grab sample data files to feed through the "
            "model, sample data must be numpy files with one model input per file "
            "as either an array in an npy file or dictionary in an npz file, "
            "defaults to random",
        )
        par.add_argument(
            "--batch-size",
            type=int,
            required=True,
            help="The batch size to run the analysis for",
        )

        if index == 0:
            par.add_argument(
                "--num-cores",
                type=int,
                default=-1,
                help="The number of physical cores to run the analysis on, "
                "defaults to all physical cores available on the system",
            )

        par.add_argument(
            "--iterations-per-check",
            type=int,
            default=20,
            help="The number of iterations to run for each performance check / timing",
        )
        par.add_argument(
            "--warmup-iterations-per-check",
            type=int,
            default=5,
            help="The number of warmup iterations to run for before checking "
            "performance / timing",
        )

    onnxruntime_parser.add_argument(
        "--no-batch-override",
        action='store_true',
        help="Do not override batch dimension of the ONNX model before running in ORT"
    )

    return parser.parse_args()


def main(args):
    if args.command == NEURALMAGIC_COMMAND:
        LOGGER.info("running benchmarking in neuralmagic...")
        runner = NMModelRunner(args.onnx_file_path, args.batch_size, args.num_cores)
    elif args.command == ONNXRUNTIME_COMMAND:
        LOGGER.info("running benchmarking in onnxruntime...")
        if args.no_batch_override:
            runner = ORTModelRunner(args.onnx_file_path)
        else:
            runner = ORTModelRunner(args.onnx_file_path, batch_size=args.batch_size)
    else:
        raise ValueError("unknown command given of {}".format(args.command))

    if args.data_glob is not None:
        LOGGER.info("creating data from glob {}".format(args.data_glob))
        data = DataLoader(
            data=args.data_glob, labels=None, batch_size=args.batch_size, iter_steps=-1
        )
    else:
        LOGGER.info("creating random data...")
        data = DataLoader.from_model_random(
            args.onnx_file_path, batch_size=args.batch_size, iter_steps=-1
        )

    _, times = runner.run(
        data,
        desc="benchmarking",
        max_steps=args.iterations_per_check + args.warmup_iterations_per_check,
    )

    if args.warmup_iterations_per_check > 0:
        times = times[args.warmup_iterations_per_check :]

    secs_per_batch = numpy.mean(times).item()
    batches_per_sec = 1.0 / secs_per_batch
    secs_per_item = secs_per_batch / args.batch_size
    items_per_sec = 1.0 / secs_per_item

    # print out results instead of log so they can't be filtered
    print(
        "benchmarking complete for batch_size {} num_core {}".format(
            args.batch_size, args.num_cores if hasattr(args, "num_cores") else None
        )
    )
    print("batch times: {}".format(times))
    print("ms/batch: {}".format(secs_per_batch * 1000))
    print("batches/sec: {}".format(batches_per_sec))
    print("ms/item: {}".format(secs_per_item * 1000))
    print("items/sec: {}".format(items_per_sec))


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
