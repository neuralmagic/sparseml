"""
Script to benchmark an ONNX model in Neural Magic, ONNXRuntime or OpenVINO inference systems.
By default will use random data. Real data can be supplied as numpy files.


##########
Command help:
usage: model_benchmark.py [-h] {neuralmagic, onnxruntime, openvino} ...

Benchmark the inference for an ONNX model

positional arguments:
  {neuralmagic,onnxruntime, openvino}

optional arguments:
  -h, --help            show this help message and exit


##########
neuralmagic command help:
usage: model_benchmark.py neuralmagic [-h] --onnx-file-path ONNX_FILE_PATH
                                      [--data-glob DATA_GLOB] --batch-size
                                      BATCH_SIZE [--num-cores NUM_CORES]
                                      [--iterations-per-check ITERATIONS_PER_CHECK]
                                      [--warmup-iterations-per-check WARMUP_ITERATIONS_PER_CHECK]
                                      [--num-samples NUM_SAMPLES]
                                      [--random_seed SEED]
                                      [--stats-file STATS_FILE]

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
                        defaults to all physical cores available on the system. This option is
                        used for neuralmagic only.
  --iterations-per-check ITERATIONS_PER_CHECK
                        The number of iterations to run for each performance
                        check / timing
  --warmup-iterations-per-check WARMUP_ITERATIONS_PER_CHECK
                        The number of warmup iterations to run for before
                        checking performance / timing
  --num-samples NUM_SAMPLES
                        Number of samples for the randomly generated data set (default to 100)
  --random-seed SEED
                        Random seed used for random data generation
  --stats-file STATS_FILE
                        Path to output CSV file to store the benchmark results

##########
onnxruntime command help:
usage: model_benchmark.py onnxruntime [-h] --onnx-file-path ONNX_FILE_PATH
                                      [--data-glob DATA_GLOB] --batch-size BATCH_SIZE
                                      [--nthreads NUM_THREADS]
                                      [--iterations-per-check ITERATIONS_PER_CHECK]
                                      [--warmup-iterations-per-check WARMUP_ITERATIONS_PER_CHECK]
                                      [--no-batch-override]
                                      [--num-samples NUM_SAMPLES]
                                      [--random_seed SEED]
                                      [--stats-file STATS_FILE]


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
                        The list of batch size to run the analysis for
  --nthreads NUM_THREADS
                        The list of threads to run the model for benchmarking.
                        Note: If ORT was built with OpenMP, use OpenMP env variable such as
                        OMP_NUM_THREADS to control the number of threads.
                        See: https://github.com/microsoft/onnxruntime/blob/master/docs/ONNX_Runtime_Perf_Tuning.md
  --iterations-per-check ITERATIONS_PER_CHECK
                        The number of iterations to run for each performance
                        check / timing
  --warmup-iterations-per-check WARMUP_ITERATIONS_PER_CHECK
                        The number of warmup iterations to run for before
                        checking performance / timing
  --no-batch-override   Do not override batch dimension of the ONNX model
                        before running in ORT
  --num-samples NUM_SAMPLES
                        Number of samples for the randomly generated data set (default to 100)
  --random-seed SEED
                        Random seed used for random data generation
  --stats-file STATS_FILE
                        Path to output CSV file to store the benchmark results


##########
openvino command help:
usage: model_benchmark.py openvino [-h] --model-file-path MODEL_FILE_PATH
                                      [--data-glob DATA_GLOB] --batch-size BATCH_SIZE
                                      [--nthreads NUM_THREADS]
                                      [--iterations-per-check ITERATIONS_PER_CHECK]
                                      [--warmup-iterations-per-check WARMUP_ITERATIONS_PER_CHECK]
                                      [--no-batch-override]
                                      [--num-samples NUM_SAMPLES]
                                      [--random_seed SEED]
                                      [--stats-file STATS_FILE]


Run benchmarks in openvino

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
                        The list of batch size to run the analysis for
  --nthreads NUM_THREADS
                        The list of threads to run the model for benchmarking
  --iterations-per-check ITERATIONS_PER_CHECK
                        The number of iterations to run for each performance
                        check / timing
  --warmup-iterations-per-check WARMUP_ITERATIONS_PER_CHECK
                        The number of warmup iterations to run for before
                        checking performance / timing
  --no-batch-override   Do not override batch dimension of the ONNX model
                        before running in ORT
  --num-samples NUM_SAMPLES
                        Number of samples for the randomly generated data set (default to 100)
  --random-seed SEED
                        Random seed used for random data generation
  --stats-file STATS_FILE
                        Path to output CSV file to store the benchmark results


##########
neuralmagic example:
python scripts/onnx/model_benchmark.py neuralmagic \
    --onnx-file-path /PATH/TO/MODEL.onnx \
    --batch-size 1 64
    --num-cores 1 16


##########
onnxruntime example:
python scripts/onnx/model_benchmark.py onnxruntime \
    --onnx-file-path /PATH/TO/MODEL.onnx \
    --batch-size 1 64


##########
openvino example
python scripts/onnx/model_benchmark.py openvino \
    --model-file-path /PATH/TO/MODEL.xml \
    --batch-size 1 64
    --nthreads 1 16 24
    --stats-file /PATH/TO/STATS_FILE.csv
"""
import os
import argparse
import itertools
import pandas as pd
import numpy
import collections
import time
from neuralmagicML import get_main_logger
from neuralmagicML.onnx.utils import (
    DataLoader,
    NMModelRunner,
    ORTModelRunner,
    OpenVINOModelRunner,
)

LOGGER = get_main_logger()
NEURALMAGIC_COMMAND = "neuralmagic"
ONNXRUNTIME_COMMAND = "onnxruntime"
OPENVINO_COMMAND = "openvino"


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
        ONNXRUNTIME_COMMAND,
        description="Run benchmarks in onnxruntime",
    )

    openvino_parser = subparsers.add_parser(
        OPENVINO_COMMAND,
        description="Run benchmarks in OpenVINO",
    )

    for index, par in enumerate(
        [neuralmagic_parser, onnxruntime_parser, openvino_parser]
    ):
        if par == openvino_parser:
            par.add_argument(
                "--model-file-path",
                type=str,
                required=True,
                help="Path to the converted/quantized model by OpenVINO",
            )
        else:
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
            nargs="+",
            type=int,
            required=True,
            help="The batch size to run the analysis for",
        )
        if par == neuralmagic_parser:
            par.add_argument(
                "--num-cores",
                nargs="+",
                type=int,
                default=[-1],
                help="The number of physical cores to run the analysis on, "
                "defaults to all physical cores available on the system",
            )
        else:
            par.add_argument(
                "--nthreads",
                nargs="+",
                type=int,
                default=[1],
                help="The number of threads to run the models",
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
        par.add_argument(
            "--random-seed",
            type=int,
            default=None,
            help="Radom seed for randomly generated numpy data (default None)",
        )
        par.add_argument(
            "--num-samples", type=int, default=100, help="Number of random data points"
        )
        par.add_argument("--stats-file", type=str, default=None, help="Output CSV file")

    onnxruntime_parser.add_argument(
        "--no-batch-override",
        action="store_true",
        help="Do not override batch dimension of the ONNX model before running in ORT",
    )
    return parser.parse_args()


def benchmark(num_cores, batch_size, args):
    """
    Run benchmark with specific number of cores/threads and batch size

    :param num_cores: number of cores to run the model
    :param batch_size: the batch size
    :param args: arguments passed in by user

    :return: list of median, average, standard deviation latency in ms,
        and number of images processed per second
    """
    if args.command == NEURALMAGIC_COMMAND:
        LOGGER.info("running benchmarking in neuralmagic...")
        runner = NMModelRunner(args.onnx_file_path, batch_size, num_cores)
    elif args.command == ONNXRUNTIME_COMMAND:
        LOGGER.info("running benchmarking in onnxruntime...")
        if args.no_batch_override:
            runner = ORTModelRunner(args.onnx_file_path, nthreads=num_cores)
        else:
            runner = ORTModelRunner(
                args.onnx_file_path, batch_size=batch_size, nthreads=num_cores
            )
    elif args.command == OPENVINO_COMMAND:
        LOGGER.info("running benchmarking in OpenVINO...")
        runner = OpenVINOModelRunner(
            args.model_file_path, nthreads=num_cores, batch_size=batch_size
        )
    else:
        raise ValueError("unknown command given of {}".format(args.command))

    if args.data_glob is not None:
        LOGGER.info("creating data from glob {}".format(args.data_glob))
        data = DataLoader(
            data=args.data_glob, labels=None, batch_size=batch_size, iter_steps=-1
        )
    else:
        LOGGER.info("creating random data...")
        if args.command == OPENVINO_COMMAND:
            data_shapes = runner.network_input_shapes()
            data = DataLoader.from_random(
                data_shapes,
                None,
                batch_size=batch_size,
                iter_steps=-1,
                num_samples=args.num_samples,
            )
        else:
            data = DataLoader.from_model_random(
                args.onnx_file_path,
                batch_size=batch_size,
                iter_steps=-1,
                num_samples=args.num_samples,
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
    secs_per_item = secs_per_batch / batch_size
    items_per_sec = 1.0 / secs_per_item

    # print out results instead of log so they can't be filtered
    print(
        "benchmarking complete for batch_size {} num_core {}".format(
            batch_size, num_cores
        )
    )
    print("batch times: {}".format(times))
    print("ms/batch: {}".format(secs_per_batch * 1000))
    print("batches/sec: {}".format(batches_per_sec))
    print("ms/item: {}".format(secs_per_item * 1000))
    print("items/sec: {}".format(items_per_sec))

    ms_times = [t * 1000 for t in times]
    median_ms_per_batch = numpy.median(ms_times)
    mean_ms_per_batch = numpy.mean(ms_times)
    std_ms_per_batch = numpy.std(ms_times)

    return median_ms_per_batch, mean_ms_per_batch, std_ms_per_batch, items_per_sec


def main(args):

    numpy.random.seed(args.random_seed)

    stats_dict = collections.OrderedDict(
        [
            ("framework", []),
            ("model", []),
            ("num_cores", []),
            ("batch_size", []),
            ("median_ms_per_batch", []),
            ("mean_ms_latency", []),
            ("std_ms_latency", []),
            ("images_per_sec", []),
        ]
    )
    num_cores_list = (
        args.num_cores if args.command == NEURALMAGIC_COMMAND else args.nthreads
    )
    for num_cores, batch_size in itertools.product(num_cores_list, args.batch_size):
        (
            median_ms_per_batch,
            mean_ms_per_batch,
            std_ms_per_batch,
            items_per_sec,
        ) = benchmark(num_cores, batch_size, args)
        stats_dict["framework"].append(args.command)
        model = (
            args.model_file_path
            if args.command == OPENVINO_COMMAND
            else args.onnx_file_path
        )
        stats_dict["model"].append(model)
        stats_dict["num_cores"].append(num_cores)
        stats_dict["batch_size"].append(batch_size)
        stats_dict["median_ms_per_batch"].append(round(median_ms_per_batch, 3))
        stats_dict["mean_ms_latency"].append(round(mean_ms_per_batch, 3))
        stats_dict["std_ms_latency"].append(round(std_ms_per_batch, 3))
        stats_dict["images_per_sec"].append(round(items_per_sec, 3))
        time.sleep(3)

    if args.stats_file is not None:
        df = pd.DataFrame.from_dict(stats_dict)
        if os.path.exists(args.stats_file):
            existing_df = pd.read_csv(args.stats_file, sep="|")
            df = existing_df.append(df)
        df.to_csv(args.stats_file, sep="|", index=False)


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
