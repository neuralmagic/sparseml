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


import glob
import os
import shutil
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Iterator, List, Optional, Tuple, Union

import cv2
import numpy
import onnx
import torch

from sparseml.onnx.utils import get_tensor_dim_shape, set_tensor_dim_shape
from sparseml.utils import create_dirs
from sparsezoo import Zoo

# ultralytics/yolov5 imports
from utils.general import non_max_suppression


__all__ = [
    "get_yolo_loader_and_saver",
    "load_image",
    "YoloPostprocessor",
    "postprocess_nms",
    "modify_yolo_onnx_input_shape",
    "annotate_image",
]


# Yolo V3 specific variables
_YOLO_V3_ANCHORS = [
    torch.Tensor([[10, 13], [16, 30], [33, 23]]),
    torch.Tensor([[30, 61], [62, 45], [59, 119]]),
    torch.Tensor([[116, 90], [156, 198], [373, 326]]),
]
_YOLO_V3_ANCHOR_GRIDS = [t.clone().view(1, -1, 1, 1, 2) for t in _YOLO_V3_ANCHORS]


def get_yolo_loader_and_saver(
    path: str, save_dir: str, image_size: Tuple[int] = (640, 640), args: Any = None
) -> Union[Iterable, Any, bool]:
    """

    :param path: file path to image or directory of .jpg files, a .mp4 video,
        or an integer (i.e. 0) for web-cam
    :param save_dir: path of directory to save to
    :param image_size: size of input images to model
    :param args: optional arguments from annotate script ArgParser
    :return: image loader iterable, result saver objects
        images, video, or web-cam based on path given, and a boolean value
        that is True is the returned objects load videos
    """
    # video
    if path.endswith(".mp4"):
        loader = YoloVideoLoader(path, image_size)
        saver = VideoSaver(
            save_dir,
            loader.original_fps,
            loader.original_frame_size,
            args.target_fps if args else None,
        )
        return loader, saver, True
    # webcam
    if path.isnumeric():
        loader = YoloWebcamLoader(int(path), image_size)
        saver = (
            VideoSaver(save_dir, 30, loader.original_frame_size, None)
            if not args.no_save
            else None
        )
        return loader, saver, True
    # image file(s)
    return YoloImageLoader(path, image_size), ImagesSaver(save_dir), False


class YoloImageLoader:
    """
    Class for pre-processing and iterating over images to be used as input for YOLO
    models

    :param path: Filepath to single image file or directory of image files to load,
        glob paths also valid
    :param image_size: size of input images to model
    """

    def __init__(self, path: str, image_size: Tuple[int] = (640, 640)):
        self._path = path
        self._image_size = image_size

        if os.path.isdir(path):
            self._image_file_paths = [
                os.path.join(path, file_name) for file_name in os.listdir(path)
            ]
        elif "*" in path:
            self._image_file_paths = glob.glob(path)
        elif os.path.isfile(path):
            # single file
            self._image_file_paths = [path]
        else:
            raise ValueError(f"{path} is not a file, glob, or directory")

    def __iter__(self) -> Iterator[Tuple[numpy.ndarray, numpy.ndarray]]:
        for image_path in self._image_file_paths:
            yield load_image(image_path, image_size=self._image_size)


class YoloVideoLoader:
    """
    Class for pre-processing and iterating over video frames to be used as input for
    YOLO models

    :param path: Filepath to single video file
    :param image_size: size of input images to model
    """

    def __init__(self, path: str, image_size: Tuple[int] = (640, 640)):
        self._path = path
        self._image_size = image_size
        self._vid = cv2.VideoCapture(self._path)
        self._total_frames = int(self._vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self._vid.get(cv2.CAP_PROP_FPS)

    def __iter__(self) -> Iterator[Tuple[numpy.ndarray, numpy.ndarray]]:
        for _ in range(self._total_frames):
            loaded, frame = self._vid.read()
            if not loaded:
                break
            yield load_image(frame, image_size=self._image_size)
        self._vid.release()

    @property
    def original_fps(self) -> float:
        """
        :return: the frames per second of the video this object reads
        """
        return self._fps

    @property
    def original_frame_size(self) -> Tuple[int, int]:
        """
        :return: the original size of frames in the video this object reads
        """
        return (
            int(self._vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._vid.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def total_frames(self) -> int:
        """
        :return: the total number of frames this object may laod from the video
        """
        return self._total_frames


class YoloWebcamLoader:
    """
    Class for pre-processing and iterating over webcam frames to be used as input for
    YOLO models.

    Adapted from: https://github.com/ultralytics/yolov5/blob/master/utils/datasets.py

    :param camera: Webcam index
    :param image_size: size of input images to model
    """

    def __init__(self, camera: int, image_size: Tuple[int] = (640, 640)):

        self._camera = camera
        self._image_size = image_size
        self._stream = cv2.VideoCapture(self._camera)
        self._stream.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    def __iter__(self) -> Iterator[Tuple[numpy.ndarray, numpy.ndarray]]:
        while True:
            if cv2.waitKey(1) == ord("q"):  # q to quit
                self._stream.release()
                cv2.destroyAllWindows()
                break
            loaded, frame = self._stream.read()

            assert loaded, f"Could not load image from webcam {self._camera}"

            frame = cv2.flip(frame, 1)  # flip left-right
            yield load_image(frame, image_size=self._image_size)

    @property
    def original_frame_size(self) -> Tuple[int, int]:
        """
        :return: the original size of frames in the stream this object reads
        """
        return (
            int(self._stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._stream.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )


class ImagesSaver:
    """
    Base class for saving YOLO model outputs. Saves each image as an individual file in
    the given directory

    :param save_dir: path to directory to write to
    """

    def __init__(self, save_dir: str):
        self._save_dir = save_dir
        self._idx = 0

        create_dirs(save_dir)

    def save_frame(self, image: numpy.ndarray):
        """
        :param image: numpy array of image to save
        """
        output_path = os.path.join(self._save_dir, f"result-{self._idx}.jpg")
        cv2.imwrite(output_path, image)
        self._idx += 1

    def close(self):
        """
        perform any clean-up tasks
        """
        pass


class VideoSaver(ImagesSaver):
    """
    Class for saving YOLO model outputs as a VideoFile

    :param save_dir: path to directory to write to
    :param original_fps: frames per second to save video with
    :param output_frame_size: size of frames to write
    :param target_fps: fps target for output video. if present, video
        will be written with a certain number of the original frames
        evenly dropped to match the target FPS.
    """

    def __init__(
        self,
        save_dir: str,
        original_fps: float,
        output_frame_size: Tuple[int, int],
        target_fps: Optional[float] = None,
    ):
        super().__init__(save_dir)

        self._output_frame_size = output_frame_size
        self._original_fps = original_fps

        if target_fps is not None and target_fps >= original_fps:
            print(
                f"target_fps {target_fps} is greater than source_fps "
                f"{original_fps}. additional target fps file will not be produced"
            )
        self._target_fps = target_fps

        self._file_path = os.path.join(self._save_dir, "results.mp4")
        self._writer = cv2.VideoWriter(
            self._file_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            original_fps,
            self._output_frame_size,
        )
        self._n_frames = 0

    def save_frame(self, image: numpy.ndarray):
        """
        :param image: numpy array of image to save
        """
        self._writer.write(image)
        self._n_frames += 1

    def close(self):
        """
        perform any clean-up tasks
        """
        self._writer.release()
        if self._target_fps is not None:
            self._write_target_fps_video()

    def _write_target_fps_video(self):
        assert self._target_fps is not None
        num_frames_to_keep = int(
            self._n_frames * (self._target_fps / self._original_fps)
        )
        # adjust target fps so we can keep the same video duration
        adjusted_target_fps = num_frames_to_keep * (self._original_fps / self._n_frames)

        # select num_frames_to_keep evenly spaced frame idxs
        frame_idxs_to_keep = set(
            numpy.round(numpy.linspace(0, self._n_frames, num_frames_to_keep))
            .astype(int)
            .tolist()
        )

        # create new video writer for adjusted video
        vid_path = os.path.join(
            self._save_dir, f"_results-{adjusted_target_fps:.2f}fps.mp4"
        )
        fps_writer = cv2.VideoWriter(
            vid_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            adjusted_target_fps,
            self._output_frame_size,
        )

        # read from original video and write to FPS adjusted video
        saved_vid = cv2.VideoCapture(self._file_path)
        for idx in range(self._n_frames):
            _, frame = saved_vid.read()
            if idx in frame_idxs_to_keep:
                fps_writer.write(frame)

        saved_vid.release()
        fps_writer.release()
        shutil.move(vid_path, self._file_path)  # overwrite original file


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
    model_input_size: Tuple[int, int] = None,
    images_per_sec: Optional[float] = None,
) -> numpy.ndarray:
    """
    Draws bounding boxes on predictions of a detection model

    :param img: Original image to annotate (no pre-processing needed)
    :param outputs: numpy array of nms outputs for the image from postprocess_nms
    :param score_threshold: minimum score a detection should have to be annotated
        on the image. Default is 0.35
    :param model_input_size: 2-tuple of expected input size for the given model to
        be used for bounding box scaling with original image. Scaling will not
        be applied if model_input_size is None. Default is None
    :param images_per_sec: optional images per second to annotate the left corner
        of the image with
    :return: the original image annotated with the given bounding boxes
    """
    img_res = numpy.copy(img)

    boxes = outputs[:, 0:4]
    scores = outputs[:, 4]
    labels = outputs[:, 5].astype(int)

    scale_y = img.shape[0] / (1.0 * model_input_size[0]) if model_input_size else 1.0
    scale_x = img.shape[1] / (1.0 * model_input_size[1]) if model_input_size else 1.0

    for idx in range(boxes.shape[0]):
        label = labels[idx].item()
        if scores[idx] > score_threshold:
            annotation_text = (
                f"{_YOLO_CLASSES[label]}: {scores[idx]:.0%}"
                if 0 <= label < len(_YOLO_CLASSES)
                else f"{scores[idx]:.0%}"
            )

            # bounding box points
            left = boxes[idx][0] * scale_x
            top = boxes[idx][1] * scale_y
            right = boxes[idx][2] * scale_x
            bottom = boxes[idx][3] * scale_y

            cv2.putText(
                img_res,
                annotation_text,
                (int(left), int(top) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,  # font scale
                (245, 46, 6),  # color
                2,  # thickness
                cv2.LINE_AA,
            )
            cv2.rectangle(
                img_res,
                (int(left), int(top)),
                (int(right), int(bottom)),
                (245, 46, 6),
                thickness=2,
            )

    if images_per_sec is not None:
        cv2.putText(
            img_res,
            f"images_per_sec: {int(images_per_sec)}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,  # font scale
            (245, 46, 6),  # color
            2,  # thickness
            cv2.LINE_AA,
        )
    return img_res
