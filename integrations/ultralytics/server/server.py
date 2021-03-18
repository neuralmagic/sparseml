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
usage: server.py [-h] [-s BATCH_SIZE] [-j NUM_CORES] [-a ADDRESS] [-p PORT]
 onnx_filepath

Host a Yolo ONNX model as a server, using the DeepSparse Engine and Flask

positional arguments:
  onnx_filepath         The full filepath of the ONNX model file or SparseZoo stub
                        to the model

optional arguments:
  -h, --help            show this help message and exit
  -s BATCH_SIZE, --batch_size BATCH_SIZE
                        The batch size to run the analysis for
  -j NUM_CORES, --num_cores NUM_CORES
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
        "onnx_filepath",
        type=str,
        help="The full filepath of the ONNX model file or SparseZoo stub to the model",
    )

    parser.add_argument(
        "-s",
        "--batch_size",
        type=int,
        default=1,
        help="The batch size to run the analysis for",
    )
    parser.add_argument(
        "-j",
        "--num_cores",
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


def arrays_to_bytes(arrays: List[numpy.array]) -> str:
    to_return = bytearray()
    for arr in arrays:
        arr_dtype = bytearray(str(arr.dtype), "utf-8")
        arr_shape = bytearray(",".join([str(a) for a in arr.shape]), "utf-8")
        sep = bytearray("|", "utf-8")
        arr_bytes = arr.ravel().tobytes()
        to_return += arr_dtype + sep + arr_shape + sep + arr_bytes
    return to_return


def bytes_to_arrays(serialized_arr: str) -> List[numpy.array]:
    sep = "|".encode("utf-8")
    arrays = []
    i_start = 0
    while i_start < len(serialized_arr) - 1:
        i_0 = serialized_arr.find(sep, i_start)
        i_1 = serialized_arr.find(sep, i_0 + 1)
        arr_dtype = numpy.dtype(serialized_arr[i_start:i_0].decode("utf-8"))
        arr_shape = tuple(
            [int(a) for a in serialized_arr[i_0 + 1 : i_1].decode("utf-8").split(",")]
        )
        arr_num_bytes = numpy.prod(arr_shape) * arr_dtype.itemsize
        arr_str = serialized_arr[i_1 + 1 : arr_num_bytes + (i_1 + 1)]
        arr = numpy.frombuffer(arr_str, dtype=arr_dtype).reshape(arr_shape)
        arrays.append(arr.copy())

        i_start = i_1 + arr_num_bytes + 1
    return arrays


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
    batch = numpy.stack(images)  # shape: (batch_size, 3, (image_size))
    return numpy.ascontiguousarray(batch)


def _postprocess_outputs(outputs: List[numpy.ndarray]) -> List[numpy.ndarray]:
    # run nms in PyTorch, only post-process first output
    output = torch.from_numpy(outputs[0])
    nms_outputs = non_max_suppression(output)
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
        outputs = _postprocess_outputs(outputs)
        postprocess_time = time.time() - postprocess_start_time
        print(f"Post-processing time: {postprocess_time * 1000.0:.4f}ms")

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
