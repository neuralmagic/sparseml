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
Example script for hosting a Yolo ONNX model as a Flask server
using the DeepSparse Engine as the inference backend

##########
Command help:
usage: server.py [-h] [-b BATCH_SIZE] [-c NUM_CORES] [-a ADDRESS] [-p PORT]
                 onnx-filepath

Host a Yolo ONNX model as a server, using the DeepSparse Engine and Flask

positional arguments:
  onnx-filepath         The full filepath of the ONNX model file or SparseZoo
                        stub to the model

optional arguments:
  -h, --help            show this help message and exit
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        The batch size to run the analysis for
  -c NUM_CORES, --num-cores NUM_CORES
                        The number of physical cores to run the analysis on,
                        defaults to all physical cores available on the system
  -a ADDRESS, --address ADDRESS
                        The IP address of the hosted model
  -p PORT, --port PORT  The port that the model is hosted on

##########
Example command for running:
python server.py \
    ~/models/yolo-v3-pruned_quant.onnx
"""

import argparse
import time
from typing import List, Tuple

import numpy
import torch

import cv2
import flask
from deepsparse import compile_model
from deepsparse.utils import arrays_to_bytes, bytes_to_arrays
from flask_cors import CORS

# ultralytics/yolov5 imports
from utils.general import non_max_suppression


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Host a Yolo ONNX model as a server, using the DeepSparse Engine and Flask"
        )
    )

    parser.add_argument(
        "onnx-filepath",
        type=str,
        help="The full filepath of the ONNX model file or SparseZoo stub to the model",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        help="The batch size to run the analysis for",
    )
    parser.add_argument(
        "-c",
        "--num-cores",
        type=int,
        default=0,
        help=(
            "The number of physical cores to run the analysis on, "
            "defaults to all physical cores available on the system"
        ),
    )
    parser.add_argument(
        "-a",
        "--address",
        type=str,
        default="0.0.0.0",
        help="The IP address of the hosted model",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=str,
        default="5543",
        help="The port that the model is hosted on",
    )

    return parser.parse_args()


def _get_grid(size: int) -> torch.Tensor:
    # adapted from yolov5.yolo.Detect._make_grid
    coords_y, coords_x = torch.meshgrid([torch.arange(size), torch.arange(size)])
    grid = torch.stack((coords_x, coords_y), 2)
    return grid.view(1, 1, size, size, 2).float()


# Yolo V3 specific variables
_YOLO_V3_ANCHORS = [
    torch.Tensor([[10, 13], [16, 30], [33, 23]]),
    torch.Tensor([[30, 61], [62, 45], [59, 119]]),
    torch.Tensor([[116, 90], [156, 198], [373, 326]]),
]
_YOLO_V3_ANCHOR_GRIDS = [t.clone().view(1, -1, 1, 1, 2) for t in _YOLO_V3_ANCHORS]
_YOLO_V3_OUTPUT_SHAPES = [80, 40, 20]
_YOLO_V3_GRIDS = [_get_grid(grid_size) for grid_size in _YOLO_V3_OUTPUT_SHAPES]


def _preprocess_image(
    img: numpy.ndarray, image_size: Tuple[int] = (640, 640)
) -> numpy.ndarray:
    # raw numpy image from cv2.imread -> preprocessed floats w/ shape (3, (image_size))
    img = cv2.resize(img, image_size)
    img = img[:, :, ::-1].transpose(2, 0, 1)
    return img.astype(numpy.float32) / 255.0


def _preprocess_images(
    images: List[numpy.ndarray], image_size: Tuple[int] = (640, 640)
) -> numpy.ndarray:
    # list of raw numpy images -> preprocessed batch of images
    images = [_preprocess_image(img, image_size) for img in images]

    batch = numpy.stack(images, image_size)  # shape: (batch_size, 3, (image_size))
    return numpy.ascontiguousarray(batch)


def _pre_nms_postprocess(outputs: List[numpy.ndarray]) -> torch.Tensor:
    # postprocess and transform raw outputs into single torch tensor
    processed_outputs = []
    for idx, pred in enumerate(outputs):
        pred = torch.from_numpy(pred)
        pred = pred.sigmoid()

        # get grid and stride
        grid = _YOLO_V3_GRIDS[idx]
        anchor_grid = _YOLO_V3_ANCHOR_GRIDS[idx]
        stride = 640 / _YOLO_V3_OUTPUT_SHAPES[idx]

        # decode xywh box values
        pred[..., 0:2] = (pred[..., 0:2] * 2.0 - 0.5 + grid) * stride
        pred[..., 2:4] = (pred[..., 2:4] * 2) ** 2 * anchor_grid
        # flatten anchor and grid dimensions -> (bs, num_predictions, num_classes + 5)
        processed_outputs.append(pred.view(pred.size(0), -1, pred.size(-1)))
    return torch.cat(processed_outputs, 1)


def _postprocess_nms(outputs: torch.Tensor) -> List[numpy.ndarray]:
    # run nms in PyTorch, only post-process first output
    nms_outputs = non_max_suppression(outputs)
    return [output.numpy() for output in nms_outputs]


def create_and_run_model_server(
    model_path: str, batch_size: int, num_cores: int, address: str, port: str
) -> flask.Flask:
    print(f"Compiling model at {model_path}")
    engine = compile_model(model_path, batch_size, num_cores)
    print(engine)

    app = flask.Flask(__name__)
    CORS(app)

    @app.route("/predict", methods=["POST"])
    def predict():
        # load raw images
        raw_data = flask.request.get_data()
        images_array = bytes_to_arrays(raw_data)
        print(f"Received {len(images_array)} images from client")

        # pre-processing
        preprocess_start_time = time.time()
        inputs = [_preprocess_images(images_array)]
        preprocess_time = time.time() - preprocess_start_time
        print(f"Pre-processing time: {preprocess_time * 1000.0:.4f}ms")

        # inference
        print("Executing model")
        outputs, elapsed_time = engine.timed_run(inputs)
        print(f"Inference time: {elapsed_time * 1000.0:.4f}ms")

        # post-processing
        postprocess_start_time = time.time()
        outputs = _pre_nms_postprocess(outputs)
        postprocess_time = time.time() - postprocess_start_time
        print(f"Post-processing, pre-nms time: {postprocess_time * 1000.0:.4f}ms")

        # NMS
        nms_start_time = time.time()
        outputs = _postprocess_nms(outputs)
        nms_time = time.time() - nms_start_time
        print(f"nms time: {nms_time * 1000.0:.4f}ms")

        return arrays_to_bytes(outputs)

    @app.route("/info", methods=["GET"])
    def info():
        return flask.jsonify({"model_path": model_path, "engine": repr(engine)})

    print("Starting Flask app")
    app.run(host=address, port=port, debug=False, threaded=True)


def main():
    args = parse_args()
    onnx_filepath = args.onnx_filepath
    batch_size = args.batch_size
    num_cores = args.num_cores
    address = args.address
    port = args.port

    create_and_run_model_server(onnx_filepath, batch_size, num_cores, address, port)


if __name__ == "__main__":
    main()
