"""
Benchmarking script for YOLOv3 ONNX models with the DeepSparse engine.


##########
Command help:
usage: benchmark.py [-h] --data-path DATA_PATH
                    [--image-shape IMAGE_SHAPE [IMAGE_SHAPE ...]]
                    [-b BATCH_SIZE] [-c NUM_CORES] [-i NUM_ITERATIONS]
                    [-w NUM_WARMUP_ITERATIONS] [-q]
                    onnx_filepath

Benchmark sparsified YOLOv3 models

positional arguments:
  onnx_filepath         The full filepath of the ONNX model file or SparseZoo
                        stub to the model

optional arguments:
  -h, --help            show this help message and exit
  --data-path DATA_PATH
                        Filepath to image examples to run the benchmark on.
                        Can be path to directory, single image jpg file, or a
                        glob path. All files should be in jpg format.
  --image-shape IMAGE_SHAPE [IMAGE_SHAPE ...]
                        Image shape to benchmark with, must be two integers.
                        Default is 640 640
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        The batch size to run the benchmark for
  -c NUM_CORES, --num-cores NUM_CORES
                        The number of physical cores to run the benchmark on,
                        defaults to all physical cores available on the system
  -i NUM_ITERATIONS, --num-iterations NUM_ITERATIONS
                        The number of iterations the benchmark will be run for
  -w NUM_WARMUP_ITERATIONS, --num-warmup-iterations NUM_WARMUP_ITERATIONS
                        The number of warmup iterations that will be executed
                        before the actual benchmarking
  -q, --quantized-inputs
                        Set flag to execute benchmark with int8 inputs instead
                        of float32

##########
Example command for running a benchmark on a pruned quantized YOLOv3:
python benchmark.py
    zoo:cv/detection/yolo_v3-spp/pytorch/ultralytics/coco/pruned_quant-aggressive_90
    --data_path /PATH/TO/COCO/DATASET/val2017
    --batch-size 32
"""


import argparse
import cv2
import glob
import numpy
import os
import time

from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Iterable, Any

from deepsparse import compile_model, cpu
from deepsparse.benchmark import BenchmarkResults

from deepsparse_utils import (
    preprocess_images,
    pre_nms_postprocess,
    postprocess_nms,
)


CORES_PER_SOCKET, AVX_TYPE, _ = cpu.cpu_details()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark sparsified YOLOv3 models"
    )

    parser.add_argument(
        "onnx_filepath",
        type=str,
        help="The full filepath of the ONNX model file or SparseZoo stub to the model",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help=(
            "Filepath to image examples to run the benchmark on. Can be path to "
            "directory, single image jpg file, or a glob path. All files should be "
            "in jpg format."
        ),
    )

    parser.add_argument(
        "--image-shape",
        type=int,
        default=(640, 640),
        nargs="+",
        help="Image shape to benchmark with, must be two integers. Default is 640 640",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=32,
        help="The batch size to run the benchmark for",
    )
    parser.add_argument(
        "-c",
        "--num-cores",
        type=int,
        default=CORES_PER_SOCKET,
        help=(
            "The number of physical cores to run the benchmark on, "
            "defaults to all physical cores available on the system"
        ),
    )
    parser.add_argument(
        "-i",
        "--num-iterations",
        help="The number of iterations the benchmark will be run for",
        type=int,
        default=80,
    )
    parser.add_argument(
        "-w",
        "--num-warmup-iterations",
        help=(
            "The number of warmup iterations that will be executed before the actual"
            " benchmarking"
        ),
        type=int,
        default=15,
    )
    parser.add_argument(
        "-q",
        "--quantized-inputs",
        help=(
            "Set flag to execute benchmark with int8 inputs instead of float32"
        ),
        action="store_true"
    )

    return parser.parse_args()


def _load_file_paths(dataset_path: str) -> List[str]:
    # from yolov5/utils/datasets.py
    path = str(Path(dataset_path).absolute())  # os-agnostic absolute path
    if '*' in path:
        files = sorted(glob.glob(path, recursive=True))  # glob
    elif os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, '*.*')))  # dir
    elif os.path.isfile(v):
        files = [path]  # files
    else:
        raise Exception(f"ERROR: {p} does not exist")

    numpy.random.shuffle(files)
    return files


def _iter_batches(
    dataset: Iterable[Any], batch_size: int, iterations: int
) -> Iterable[Any]:
    iteration = 0
    batch = []
    while iteration < iterations:
        for item in dataset:
            batch.append(item)

            if len(batch) == batch_size:
                yield batch
                batch = []
                iteration += 1

                if iteration >= iterations:
                    break


def benchmark_yolo(args):
    model = compile_model(
        args.onnx_filepath, args.batch_size, args.num_cores
    )
    dataset = _load_file_paths(args.data_path)
    total_iterations = args.num_iterations + args.num_warmup_iterations
    data_loader = _iter_batches(dataset, args.batch_size, total_iterations)

    print(
        (
            f"Running for {args.num_warmup_iterations} warmup iterations "
            f"and {args.num_iterations} benchmarking iterations"
        ),
        flush=True
    )

    results = BenchmarkResults()
    progress_bar = tqdm(total=args.num_iterations)
    for iteration, batch in enumerate(data_loader):
        iter_start = time.time()

        # pre-processing
        batch = [cv2.imread(filepath) for filepath in batch]
        inputs = [preprocess_images(batch)]

        # inference
        outputs = model.run(inputs)

        # post-processing
        outputs = pre_nms_postprocess(outputs)

        # NMS
        outputs = postprocess_nms(outputs)

        iter_end = time.time()

        if iteration >= args.num_warmup_iterations:
            results.append_batch(
                time_start=iter_start,
                time_end=iter_end,
                batch_size=args.batch_size,
            )
            progress_bar.update(1)

    progress_bar.close()

    print(f"Benchmarking complete. End-toend results:\n{results}")

    print(f"End-to-end per image time: {results.ms_per_batch / args.batch_size}ms")


def main():
    args = parse_args()
    assert len(args.image_shape) == 2

    benchmark_yolo(args)


if __name__ == "__main__":
    main()






