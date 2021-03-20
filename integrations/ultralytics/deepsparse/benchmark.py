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
Benchmarking script for YOLOv3 ONNX models with the DeepSparse engine.


##########
Command help:
usage: benchmark.py [-h] [-e {deepsparse,onnxruntime,torch}] --data-path
                    DATA_PATH [--image-shape IMAGE_SHAPE [IMAGE_SHAPE ...]]
                    [-b BATCH_SIZE] [-c NUM_CORES] [-i NUM_ITERATIONS]
                    [-w NUM_WARMUP_ITERATIONS] [-q] [--fp16] [--device DEVICE]
                    model_filepath

Benchmark sparsified YOLOv3 models

positional arguments:
  model_filepath        The full filepath of the ONNX model file or SparseZoo
                        stub to the model for deepsparse and onnxruntime
                        benchmarks. Path to a .pt loadable PyTorch Module for
                        torch benchmarks - the Module can be the top-level
                        object loaded or loaded into 'model' in a state dict

optional arguments:
  -h, --help            show this help message and exit
  -e {deepsparse,onnxruntime,torch}, --engine {deepsparse,onnxruntime,torch}
                        Inference engine backend to run benchmark on. Choices
                        are 'deepsparse', 'onnxruntime', and 'torch'. Default
                        is 'deepsparse'
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
  --fp16                Set flag to execute torch benchmark in half precision
                        (fp16)
  --device DEVICE       Device to benchmark the model with. Default is cpu.
                        Non cpu benchmarking only supported for torch
                        benchmarking. Default is 'cpu'

##########
Example command for running a benchmark on a pruned quantized YOLOv3:
python benchmark.py \
    zoo:cv/detection/yolo_v3-spp/pytorch/ultralytics/coco/pruned_quant-aggressive_90 \
    --data_path /PATH/TO/COCO/DATASET/val2017 \
    --batch-size 32 \

##########
Example for benchmarking on a local YOLOv3 PyTorch model on GPU with half precision:
python benchmark.py \
    /PATH/TO/yolov3-spp.pt \
    --data_path /PATH/TO/COCO/DATASET/val2017 \
    --batch-size 32 \
    --device cuda \
    --half-precision
"""


import argparse
import glob
import os
import time
from pathlib import Path
from typing import Any, Iterable, List, Tuple, Union

import numpy
import onnxruntime
import torch
from tqdm.auto import tqdm

import cv2
from deepsparse import compile_model, cpu
from deepsparse.benchmark import BenchmarkResults
from deepsparse_utils import load_image, postprocess_nms, pre_nms_postprocess


CORES_PER_SOCKET, AVX_TYPE, _ = cpu.cpu_details()

DEEPSPARSE_ENGINE = "deepsparse"
ORT_ENGINE = "onnxruntime"
TORCH_ENGINE = "torch"


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark sparsified YOLOv3 models")

    parser.add_argument(
        "model_filepath",
        type=str,
        help=(
            "The full filepath of the ONNX model file or SparseZoo stub to the model "
            "for deepsparse and onnxruntime benchmarks. Path to a .pt loadable PyTorch "
            "Module for torch benchmarks - the Module can be the top-level object "
            "loaded or loaded into 'model' in a state dict"
        ),
    )

    parser.add_argument(
        "-e",
        "--engine",
        type=str,
        default=DEEPSPARSE_ENGINE,
        choices=[DEEPSPARSE_ENGINE, ORT_ENGINE, TORCH_ENGINE],
        help=(
            "Inference engine backend to run benchmark on. Choices are 'deepsparse', "
            "'onnxruntime', and 'torch'. Default is 'deepsparse'"
        ),
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
        help=("Set flag to execute benchmark with int8 inputs instead of float32"),
        action="store_true",
    )
    parser.add_argument(
        "--fp16",
        help=("Set flag to execute torch benchmark in half precision (fp16)"),
        action="store_true",
    )
    parser.add_argument(
        "--device",
        type=_parse_device,
        default="cpu",
        help=(
            "Device to benchmark the model with. Default is cpu. Non cpu benchmarking "
            "only supported for torch benchmarking. Default is 'cpu'"
        ),
    )

    return parser.parse_args()


def _parse_device(device: Union[str, int]) -> Union[str, int]:
    try:
        return int(device)
    except:
        return device


def _load_images(dataset_path: str, image_size: Tuple[int]) -> List[str]:
    # from yolov5/utils/datasets.py
    path = str(Path(dataset_path).absolute())  # os-agnostic absolute path
    if "*" in path:
        files = sorted(glob.glob(path, recursive=True))  # glob
    elif os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.*")))  # dir
    elif os.path.isfile(path):
        files = [path]  # files
    else:
        raise Exception(f"ERROR: {path} does not exist")

    numpy.random.shuffle(files)
    return [load_image(file, image_size) for file in files]


def _load_model(args) -> Any:
    # validation
    if args.device != "cpu" and args.engine != TORCH_ENGINE:
        raise ValueError(f"device {args.device} is not supported for {args.engine}")
    if args.fp16 and args.engine != TORCH_ENGINE:
        raise ValueError(f"half precision is not supported for {args.engine}")
    if args.quantized_inputs and args.engine == TORCH_ENGINE:
        raise ValueError(f"quantized inputs not supported for {args.engine}")
    if args.num_cores != CORES_PER_SOCKET and args.engine != DEEPSPARSE_ENGINE:
        raise ValueError(
            f"overriding default num_cores not supported for {args.engine}"
        )

    # load model
    if args.engine == DEEPSPARSE_ENGINE:
        print(f"Compiling deepsparse model for {args.model_filepath}")
        model = compile_model(args.model_filepath, args.batch_size, args.num_cores)
    elif args.engine == ORT_ENGINE:
        print(f"loading onnxruntime model for {args.model_filepath}")
        model = onnxruntime.InferenceSession(args.model_filepath)
    elif args.engine == TORCH_ENGINE:
        print(f"loading torch model for {args.model_filepath}")
        model = torch.load(args.model_filepath)
        if isinstance(model, dict):
            model = model["model"]
        model.to(args.device)
        model.eval()
        if args.fp16:
            model.half()
    return model


def _iter_batches(
    dataset: List[Any],
    batch_size: int,
    iterations: int,
) -> Iterable[Any]:
    iteration = 0
    batch = []
    batch_template = numpy.ascontiguousarray(
        numpy.zeros((batch_size, *dataset[0].shape), dtype=dataset[0].dtype)
    )
    while iteration < iterations:
        for item in dataset:
            batch.append(item)

            if len(batch) == batch_size:
                yield numpy.stack(batch, out=batch_template)

                batch = []
                iteration += 1

                if iteration >= iterations:
                    break


def _preprocess_batch(args, batch: numpy.ndarray) -> Union[numpy.ndarray, torch.Tensor]:
    if args.engine == TORCH_ENGINE:
        batch = torch.from_numpy(batch)
        batch = batch.to(args.device)
        batch = batch.half() if args.fp16 else batch.float()
        batch /= 255.0
    elif not args.quantized_inputs:
        batch = batch.astype(numpy.float32) / 255.0
    return batch


def _run_model(
    args, model: Any, batch: Union[numpy.ndarray, torch.Tensor]
) -> List[Union[numpy.ndarray, torch.Tensor]]:
    outputs = None
    if args.engine == TORCH_ENGINE:
        outputs = model(batch)[0]
    elif args.engine == ORT_ENGINE:
        outputs = model.run(
            [out.name for out in model.get_outputs()],  # outputs
            {model.get_inputs()[0].name: batch},  # inputs dict
        )
    else:  # deepsparse
        outputs = model.run([batch])
    return outputs


def benchmark_yolo(args):
    model = _load_model(args)
    print("Loading dataset")
    dataset = _load_images(args.data_path, args.image_shape)
    total_iterations = args.num_iterations + args.num_warmup_iterations
    data_loader = _iter_batches(dataset, args.batch_size, total_iterations)

    print(
        (
            f"Running for {args.num_warmup_iterations} warmup iterations "
            f"and {args.num_iterations} benchmarking iterations"
        ),
        flush=True,
    )

    results = BenchmarkResults()
    progress_bar = tqdm(total=args.num_iterations)

    for iteration, batch in enumerate(data_loader):
        if args.device != "cpu":
            torch.cuda.synchronize()
        iter_start = time.time()

        # pre-processing
        batch = _preprocess_batch(args, batch)

        # inference
        outputs = _run_model(args, model, batch)

        # post-processing
        if args.engine != TORCH_ENGINE:
            outputs = pre_nms_postprocess(outputs)

        # NMS
        outputs = postprocess_nms(outputs)

        if args.device != "cpu":
            torch.cuda.synchronize()
        iter_end = time.time()

        if iteration >= args.num_warmup_iterations:
            results.append_batch(
                time_start=iter_start,
                time_end=iter_end,
                batch_size=args.batch_size,
            )
            progress_bar.update(1)

    progress_bar.close()

    print(f"Benchmarking complete. End-to-end results:\n{results}")

    print(f"End-to-end per image time: {results.ms_per_batch / args.batch_size}ms")


def main():
    args = parse_args()
    assert len(args.image_shape) == 2

    benchmark_yolo(args)


if __name__ == "__main__":
    main()
