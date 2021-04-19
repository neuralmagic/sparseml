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
Utilities for Yolo V3 pre and post processing for DeepSparse pipelines

Postprocessing is currently tied to yolov3-spp, modify anchor and output
variables if using a different model.
"""


from tempfile import NamedTemporaryFile
from typing import List, Optional, Tuple, Union

import cv2
import numpy
import onnx
import torch

from sparseml.onnx.utils import get_tensor_dim_shape, set_tensor_dim_shape
from sparsezoo import Zoo

# ultralytics/yolov5 imports
from utils.general import non_max_suppression


__all__ = [
    "load_image",
    "YoloPostprocessor",
    "postprocess_nms",
]


# Yolo V3 specific variables
_YOLO_V3_ANCHORS = [
    torch.Tensor([[10, 13], [16, 30], [33, 23]]),
    torch.Tensor([[30, 61], [62, 45], [59, 119]]),
    torch.Tensor([[116, 90], [156, 198], [373, 326]]),
]
_YOLO_V3_ANCHOR_GRIDS = [t.clone().view(1, -1, 1, 1, 2) for t in _YOLO_V3_ANCHORS]


def load_image(
    img: Union[str, numpy.ndarray], image_size: Tuple[int] = (640, 640)
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    :param img: file path to image or raw image array
    :param image_size: target shape for image
    :return: Image loaded into numpy and reshaped to the given shape and the original
        image
    """
    img = cv2.imread(img) if isinstance(img, str) else img
    img_resized = cv2.resize(img, image_size)
    img_transposed = img_resized[:, :, ::-1].transpose(2, 0, 1)

    return img_transposed, img


class YoloPostprocessor:
    """
    Class for performing postprocessing of YOLOv3 model predictions

    :param image_size: size of input image to model. used to calculate stride based on
        output shapes
    """

    def __init__(self, image_size: Tuple[int] = (640, 640)):
        self._image_size = image_size
        self._grids = {}  # Dict[Tuple[int], torch.Tensor]

    def pre_nms_postprocess(self, outputs: List[numpy.ndarray]) -> torch.Tensor:
        """
        :param outputs: raw outputs of a YOLOv3 model before anchor grid processing
        :return: post-processed model outputs without NMS.
        """
        # postprocess and transform raw outputs into single torch tensor
        processed_outputs = []
        for idx, pred in enumerate(outputs):
            pred = torch.from_numpy(pred)
            pred = pred.sigmoid()

            # get grid and stride
            grid_shape = pred.shape[2:4]
            grid = self._get_grid(grid_shape)
            anchor_grid = _YOLO_V3_ANCHOR_GRIDS[idx]
            stride = self._image_size[0] / grid_shape[0]

            # decode xywh box values
            pred[..., 0:2] = (pred[..., 0:2] * 2.0 - 0.5 + grid) * stride
            pred[..., 2:4] = (pred[..., 2:4] * 2) ** 2 * anchor_grid
            # flatten anchor and grid dimensions ->
            #       (bs, num_predictions, num_classes + 5)
            processed_outputs.append(pred.view(pred.size(0), -1, pred.size(-1)))
        return torch.cat(processed_outputs, 1)

    def _get_grid(self, grid_shape: Tuple[int]) -> torch.Tensor:
        if grid_shape not in self._grids:
            # adapted from yolov5.yolo.Detect._make_grid
            coords_y, coords_x = torch.meshgrid(
                [torch.arange(grid_shape[0]), torch.arange(grid_shape[1])]
            )
            grid = torch.stack((coords_x, coords_y), 2)
            self._grids[grid_shape] = grid.view(
                1, 1, grid_shape[0], grid_shape[1], 2
            ).float()
        return self._grids[grid_shape]


def postprocess_nms(outputs: torch.Tensor) -> List[numpy.ndarray]:
    """
    :param outputs: Tensor of post-processed model outputs
    :return: List of numpy arrays of NMS predictions for each image in the batch
    """
    # run nms in PyTorch, only post-process first output
    nms_outputs = non_max_suppression(outputs)
    return [output.cpu().numpy() for output in nms_outputs]


def modify_yolo_onnx_input_shape(
    model_path: str, image_shape: Tuple[int]
) -> Tuple[str, Optional[NamedTemporaryFile]]:
    """
    Creates a new YOLOv3 ONNX model from the given path that accepts the given input
    shape. If the given model already has the given input shape no modifications are
    made. Uses a tempfile to store the modified model file.

    :param model_path: file path to YOLOv3 ONNX model or SparseZoo stub of the model
        to be loaded
    :param image_shape: 2-tuple of the image shape to resize this yolo model to
    :return: filepath to an onnx model reshaped to the given input shape will be the
        original path if the shape is the same.  Additionally returns the
        NamedTemporaryFile for managing the scope of the object for file deletion
    """
    original_model_path = model_path
    if model_path.startswith("zoo:"):
        # load SparseZoo Model from stub
        model = Zoo.load_model_from_stub(model_path)
        model_path = model.onnx_file.downloaded_path()
        print(f"Downloaded {original_model_path} to {model_path}")

    model = onnx.load(model_path)
    model_input = model.graph.input[0]

    initial_x = get_tensor_dim_shape(model_input, 2)
    initial_y = get_tensor_dim_shape(model_input, 3)

    if not (isinstance(initial_x, int) and isinstance(initial_y, int)):
        return model_path, None  # model graph does not have static integer input shape

    if (initial_x, initial_y) == tuple(image_shape):
        return model_path, None  # no shape modification needed

    scale_x = initial_x / image_shape[0]
    scale_y = initial_y / image_shape[1]
    set_tensor_dim_shape(model_input, 2, image_shape[0])
    set_tensor_dim_shape(model_input, 3, image_shape[1])

    for model_output in model.graph.output:
        output_x = get_tensor_dim_shape(model_output, 2)
        output_y = get_tensor_dim_shape(model_output, 3)
        set_tensor_dim_shape(model_output, 2, int(output_x / scale_x))
        set_tensor_dim_shape(model_output, 3, int(output_y / scale_y))

    tmp_file = NamedTemporaryFile()  # file will be deleted after program exit
    onnx.save(model, tmp_file.name)

    print(
        f"Overwriting original model shape {(initial_x, initial_y)} to {image_shape}\n"
        f"Original model path: {original_model_path}, new temporary model saved to "
        f"{tmp_file.name}"
    )

    return tmp_file.name, tmp_file


_YOLO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def annotate_image(
    img: numpy.ndarray,
    outputs: numpy.ndarray,
    score_threshold: float = 0.35,
) -> numpy.ndarray:
    """
    Draws bounding boxes on predictions of a detection model

    :param img: Original image to annotate (no pre-processing needed)
    :param outputs: numpy array of nms outputs for the image from postprocess_nms
    :param score_threshold: minimum score a detection should have to be annotated
        on the image. Default is 0.35
    :return: the original image annotated with the given bounding boxes
    """
    img_res = numpy.copy(img)

    boxes = outputs[:, 0:4]
    scores = outputs[:, 4]
    labels = outputs[:, 5].astype(int)

    for idx in range(boxes.shape[0]):
        label = labels[idx].item()
        if scores[idx] > score_threshold:
            annotation_text = (
                f"{_YOLO_CLASSES[label]}: {scores[idx]:.0%}"
                if 0 <= label < len(_YOLO_CLASSES)
                else f"{scores[idx]:.0%}"
            )

            # bounding box points
            left = boxes[idx][0]
            top = boxes[idx][1]
            right = boxes[idx][2]
            bottom = boxes[idx][3]

            cv2.putText(
                img_res,
                annotation_text,
                (int(left), int(top) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,  # font scale
                (255, 0, 0),  # color
                2,  # thickness
                cv2.LINE_AA,
            )
            cv2.rectangle(
                img_res,
                (int(left), int(top)),
                (int(right), int(bottom)),
                (125, 255, 51),
                thickness=2,
            )
    return img_res
