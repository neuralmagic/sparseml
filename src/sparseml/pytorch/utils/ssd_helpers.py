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
Helper functions and classes for creating and training PyTorch SSD models
"""

import itertools
import math
import random
from collections import defaultdict
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Union

import numpy
import torch
from PIL import Image
from torch import Tensor


try:
    from torchvision.ops.boxes import batched_nms, box_iou
except Exception:
    box_iou = None
    batched_nms = None


__all__ = [
    "DefaultBoxes",
    "get_default_boxes_300",
    "ssd_random_crop",
    "MeanAveragePrecision",
]


class DefaultBoxes(object):
    """
    Convenience class for creating, representing, encoding, and decoding default boxes

    :param image_size: input image size
    :param feature_maps: list of feature map sizes
    :param steps: steps to use between boxes in a feature map
    :param scales: list of ranges of size scales to use for each feature map
    :param aspect_ratios: list of aspect ratios to construct boxes with
    :param scale_xy: parameter to scale box center by when encoding
    :param scale_wh: parameter to scale box dimensions by when encoding
    """

    def __init__(
        self,
        image_size: int,
        feature_maps: List[int],
        steps: List[int],
        scales: List[int],
        aspect_ratios: List[List[int]],
        scale_xy: float = 0.1,
        scale_wh: float = 0.2,
    ):
        self._feature_maps = feature_maps
        self._image_size = image_size

        self._scale_xy = scale_xy
        self._scale_wh = scale_wh

        # According to https://github.com/weiliu89/caffe
        # Calculation method slightly different from paper
        self._steps = steps
        self._scales = scales
        self._aspect_ratios = aspect_ratios

        self._default_boxes = self._get_default_boxes()
        self._default_boxes_ltrb = self._get_default_boxes_ltrb()

    def _get_default_boxes(self) -> Tensor:
        default_boxes = []
        feature_steps = self._image_size / numpy.array(self._steps)

        # size of feature and number of feature
        for idx, feature_map_size in enumerate(self._feature_maps):
            # unpack scales
            min_scale, max_scale = self._scales[idx]
            # set scales to range based on image size
            min_scale = min_scale / self._image_size
            max_scale = max_scale / self._image_size
            mid_scale = math.sqrt(min_scale * max_scale)
            all_sizes = [(min_scale, min_scale), (mid_scale, mid_scale)]

            for alpha in self._aspect_ratios[idx]:
                w = min_scale * math.sqrt(alpha)
                h = min_scale / math.sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            for w, h in all_sizes:
                for i, j in itertools.product(range(feature_map_size), repeat=2):
                    cx = (j + 0.5) / feature_steps[idx]
                    cy = (i + 0.5) / feature_steps[idx]
                    default_boxes.append((cx, cy, w, h))
        default_boxes = torch.tensor(default_boxes, dtype=torch.float)
        default_boxes.clamp_(min=0, max=1)
        return default_boxes

    def _get_default_boxes_ltrb(self) -> Tensor:
        # For IoU calculation
        default_boxes_ltrb = self._default_boxes.clone()
        default_boxes_ltrb[:, 0] = (
            self._default_boxes[:, 0] - 0.5 * self._default_boxes[:, 2]
        )
        default_boxes_ltrb[:, 1] = (
            self._default_boxes[:, 1] - 0.5 * self._default_boxes[:, 3]
        )
        default_boxes_ltrb[:, 2] = (
            self._default_boxes[:, 0] + 0.5 * self._default_boxes[:, 2]
        )
        default_boxes_ltrb[:, 3] = (
            self._default_boxes[:, 1] + 0.5 * self._default_boxes[:, 3]
        )
        return default_boxes_ltrb

    @property
    def scale_xy(self) -> float:
        """
        :return: parameter to scale box center by when encoding
        """
        return self._scale_xy

    @property
    def scale_wh(self) -> float:
        """
        :return: parameter to scale box dimensions by when encoding
        """
        return self._scale_wh

    @property
    def num_default_boxes(self) -> int:
        """
        :return: the number of default boxes this object defines
        """
        return self._default_boxes.size(0)

    def as_ltrb(self) -> Tensor:
        """
        :return: The default boxes represented by this object in
            top left, top right pixel representation
        """
        return self._default_boxes_ltrb

    def as_xywh(self) -> Tensor:
        """
        :return: The default boxes represented by this object in
            center pixel, width, height representation
        """
        return self._default_boxes

    def encode_image_box_labels(
        self, boxes: Tensor, labels: Tensor, threshold: float = 0.5
    ) -> Tuple[Tensor, Tensor]:
        """
        Given the bounding box and image annotations for a single image with N objects
        will encode the box annotations as offsets to the default boxes and labels
        to the associated default boxes based on the annotation boxes and default
        boxes with an intersection over union (IoU) greater than the given threshold.

        :param boxes: Bounding box annotations for objects in an image. Should have
            shape N,4 and be represented in ltrb format
        :param labels: Label annotations for N objects in an image.
        :param threshold: The minimum IoU bounding boxes and default boxes should share
            to be encoded
        :return: A tuple of the offset encoded bounding boxes and default box encoded
            labels
        """
        if box_iou is None:
            raise RuntimeError(
                "Unable to import box_iou from torchvision.ops try upgrading your"
                " torch and torchvision versions"
            )
        if labels.numel() == 0:
            # return encoded box offsets as zeros
            boxes_encoded = torch.zeros(4, self.num_default_boxes).float()
            labels_encoded = torch.zeros(self.num_default_boxes, dtype=torch.long)
            return boxes_encoded, labels_encoded

        ious = box_iou(boxes, self._default_boxes_ltrb)  # N,num_default_box

        # Ensure that at least one box is encoded for each annotation
        best_dbox_ious, best_dbox_idx = ious.max(dim=0)  # best IoU for each default box
        best_bbox_ious, best_bbox_idx = ious.max(dim=1)  # best IoU for each label box

        best_dbox_ious.index_fill_(0, best_bbox_idx, 2.0)
        idx = torch.arange(0, best_bbox_idx.size(0))
        best_dbox_idx[best_bbox_idx[idx]] = idx

        # filter default boxes by IoU threshold
        labels_encoded = torch.zeros(self.num_default_boxes, dtype=torch.long)
        boxes_masked = self._default_boxes_ltrb.clone()

        threshold_mask = best_dbox_ious > threshold

        labels_encoded[threshold_mask] = labels[best_dbox_idx[threshold_mask]].long()
        boxes_masked[threshold_mask, :] = boxes[best_dbox_idx[threshold_mask], :]

        _ltrb_to_xywh(boxes_masked)  # convert to lrtb format

        # encode masked boxes as offset tensor
        xy_encoded = (
            (1.0 / self.scale_xy)
            * (boxes_masked[:, :2] - self._default_boxes[:, :2])
            / self._default_boxes[:, :2]
        )
        wh_encoded = (1.0 / self.scale_wh) * (
            boxes_masked[:, 2:] / self._default_boxes[:, 2:]
        ).log()

        boxes_encoded = torch.cat(
            (xy_encoded, wh_encoded), dim=1
        )  # shape: num_default_boxes, 4
        boxes_encoded = boxes_encoded.transpose(
            0, 1
        ).contiguous()  # final shape: 4, num_default_boxes

        return boxes_encoded.float(), labels_encoded.long()

    def decode_output_batch(
        self,
        boxes: Tensor,
        scores: Tensor,
        score_threhsold: float = 0.01,
        iou_threshold: float = 0.45,
        max_detections: int = 200,
    ) -> List[Tuple[Tensor, Tensor, Tensor]]:
        """
        Decodes a batch detection model outputs from default box offsets and class
        scores to ltrb formatted bounding boxes, predicted labels, and scores
        for each image of the batch using non maximum suppression.

        :param boxes: Encoded default-box offsets. Expected shape:
            batch_size,4,num_default_boxes
        :param scores: Class scores for each image, class, box combination.
            Expected shape: batch_size,num_classes,num_default_boxes
        :param score_threhsold: minimum softmax score to be considered a positive
            prediction. Default is 0.01 following the SSD paper
        :param iou_threshold: The minimum IoU between two boxes to be considered the
            same object in non maximum suppression
        :param max_detections: the maximum number of detections to keep per image.
            Default is 200
        :return: Detected object boudning boxes, predicted labels, and class score for
            each image in this batch
        """
        if batched_nms is None:
            raise RuntimeError(
                "Unable to import batched_nms from torchvision.ops try upgrading your"
                " torch and torchvision versions"
            )
        # Re-order so that dimensions are batch_size,num_default_boxes,{4,num_classes}
        boxes = boxes.permute(0, 2, 1)
        scores = scores.permute(0, 2, 1)

        # convert box offsets to bounding boxes and convert to ltrb form
        default_boxes = self._default_boxes.unsqueeze(0)  # extra dimension for math ops
        boxes[:, :, :2] = (
            self.scale_xy * boxes[:, :, :2] * default_boxes[:, :, :2]
            + default_boxes[:, :, :2]
        )
        boxes[:, :, 2:] = (self._scale_wh * boxes[:, :, 2:]).exp() * default_boxes[
            :, :, 2:
        ]
        _xywh_to_ltrb_batch(boxes)

        # take softmax of class scores
        scores = torch.nn.functional.softmax(scores, dim=-1)  # class dimension

        # run non max suppression for each image in the batch and store outputs
        detection_outputs = []
        for image_boxes, box_class_scores in zip(boxes.split(1, 0), scores.split(1, 0)):
            # strip batch dimension
            image_boxes = image_boxes.squeeze(0)
            box_class_scores = box_class_scores.squeeze(0)

            # get highest score per box and filter out background class
            box_class_scores[:, 0] = 0
            box_scores, box_labels = box_class_scores.max(dim=1)
            # background_filter = torch.nonzero(box_labels, as_tuple=False).squeeze()
            background_filter = box_scores > score_threhsold
            image_boxes = image_boxes[background_filter]
            box_scores = box_scores[background_filter]
            box_labels = box_labels[background_filter]

            if image_boxes.dim() == 0:
                # nothing predicted, add empty result and continue
                detection_outputs.append(
                    (torch.zeros(1, 4), torch.zeros(1), torch.zeros(1))
                )
                continue
            if image_boxes.dim() == 1:
                image_boxes = image_boxes.unsqueeze(0)
                box_scores = box_scores.unsqueeze(0)
                box_labels = box_labels.unsqueeze(0)

            # filter boxes, classes, and scores by nms results
            nms_filter = batched_nms(image_boxes, box_scores, box_labels, iou_threshold)
            if nms_filter.size(0) > max_detections:
                # update nms_filter to keep the boxes with top max_detections scores
                box_scores_nms = box_scores[nms_filter]
                sorted_scores_nms_idx = torch.argsort(box_scores_nms, descending=True)
                nms_filter = nms_filter[sorted_scores_nms_idx[:max_detections]]
            detection_outputs.append(
                (
                    image_boxes[nms_filter],
                    box_labels[nms_filter],
                    box_scores[nms_filter],
                )
            )

        return detection_outputs


def get_default_boxes_300(voc: bool = False) -> DefaultBoxes:
    """
    Convenience function for generating DefaultBoxes object for standard SSD300 model

    :param voc: set True if default boxes should be made for VOC dataset.
        Will set scales to be slightly larger than for the default
        COCO dataset configuration
    :return: DefaultBoxes object implemented for standard SSD300 models
    """
    image_size = 300
    feature_maps = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    # use the scales here:
    # https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    if voc:
        scales = [[30, 60], [60, 111], [111, 162], [162, 213], [213, 264], [264, 315]]
    else:
        scales = [[21, 45], [45, 99], [99, 153], [153, 207], [207, 261], [261, 315]]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    return DefaultBoxes(image_size, feature_maps, steps, scales, aspect_ratios)


def _ltrb_to_xywh(boxes):
    # in-place conversion from ltrb to cx,cy,w,h format
    # expected input shape N,4
    cx = 0.5 * (boxes[:, 0] + boxes[:, 2])
    cy = 0.5 * (boxes[:, 1] + boxes[:, 3])
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    boxes[:, 0] = cx
    boxes[:, 1] = cy
    boxes[:, 2] = w
    boxes[:, 3] = h


def _xywh_to_ltrb_batch(boxes):
    # in-place conversion from cx, cy, w, h format to ltrb
    # expected input shape M,N,4
    lt_x = boxes[:, :, 0] - 0.5 * boxes[:, :, 2]
    lt_y = boxes[:, :, 1] - 0.5 * boxes[:, :, 3]
    rb_x = boxes[:, :, 0] + 0.5 * boxes[:, :, 2]
    rb_y = boxes[:, :, 1] + 0.5 * boxes[:, :, 3]

    boxes[:, :, 0] = lt_x
    boxes[:, :, 1] = lt_y
    boxes[:, :, 2] = rb_x
    boxes[:, :, 3] = rb_y


# potential IoU thresholds for selecting cropping boundaries for SSD model input
_SSD_RANDOM_CROP_OPTIONS = (
    # return original
    None,
    # random crop (all IoUs are valid)
    -1,
    # crop with minimum box IoU
    0.1,
    0.3,
    0.5,
    0.7,
    0.9,
)


def ssd_random_crop(
    image: Image.Image, boxes: Tensor, labels: Tensor
) -> Tuple[Image.Image, Tensor, Tensor]:
    """
    Performs one of the random SSD crops on a given image, bounding boxes,
    and labels as implemented in the original paper.

    | Chooses between following 3 conditions:
    |     1. Preserve the original image
    |     2. Random crop minimum IoU is among 0.1, 0.3, 0.5, 0.7, 0.9
    |     3. Random crop

    Adapted from: https://github.com/chauhan-utk/ssd.DomainAdaptation

    :param image: the image to potentially crop
    :param boxes: a tensor of bounding boxes in ltrb format with shape n_boxes,4
    :param labels: a tensor of labels for each of the bounding boxes
    :return: the cropped image, boxes, and labels
    """
    if box_iou is None:
        raise RuntimeError(
            "Unable to import box_iou from torchvision.ops try upgrading your"
            " torch and torchvision versions"
        )
    # Loop will always return something when None or 0 is selected
    while True:
        min_iou = random.choice(_SSD_RANDOM_CROP_OPTIONS)

        # do nothing
        if min_iou is None:
            return image, boxes, labels

        w_orig, h_orig = image.size

        # search for 50 random crops before trying a different threshold
        for _ in range(50):
            # crops to [.1,1.0] of image area since 0.3 * 0.3 ~= 0.1
            w_crop = random.uniform(0.3, 1.0)
            h_crop = random.uniform(0.3, 1.0)

            if w_crop / h_crop < 0.5 or w_crop / h_crop > 2:
                continue  # keep crop ratio between 1:2 / 2:1

            # generate bounding box of size w_crop,h_crop
            left = random.uniform(0, 1.0 - w_crop)
            top = random.uniform(0, 1.0 - h_crop)
            right = left + w_crop
            bottom = top + h_crop

            # get IoUs between given bounding boxes and cropped box
            ious = box_iou(boxes, torch.tensor([[left, top, right, bottom]]))

            if not (ious > min_iou).all():
                continue  # do not use this crop if all boxes do not pass threshold

            # discard any boxes whose center is not in the cropped image
            x_centers = 0.5 * (boxes[:, 0] + boxes[:, 2])
            y_centers = 0.5 * (boxes[:, 1] + boxes[:, 3])

            center_in_crop_mask = (
                (x_centers > left)
                & (x_centers < right)
                & (y_centers > top)
                & (y_centers < bottom)
            )

            if not center_in_crop_mask.any():
                continue  # do not use crop if no boxes are centered in it

            # clip bounding boxes to the cropped boundaries
            boxes[boxes[:, 0] < left, 0] = left
            boxes[boxes[:, 1] < top, 1] = top
            boxes[boxes[:, 2] > right, 2] = right
            boxes[boxes[:, 3] > bottom, 3] = bottom

            # drop bounding boxes whose centers are not in the copped region
            boxes = boxes[center_in_crop_mask, :]
            labels = labels[center_in_crop_mask]

            # expand the cropped region to map to pixels in the image and crop
            image_crop_box = (
                int(left * w_orig),
                int(top * h_orig),
                int(right * w_orig),
                int(bottom * h_orig),
            )
            image = image.crop(image_crop_box)

            # shift and crop bounding boxes
            boxes[:, 0] = (boxes[:, 0] - left) / w_crop
            boxes[:, 1] = (boxes[:, 1] - top) / h_crop
            boxes[:, 2] = (boxes[:, 2] - left) / w_crop
            boxes[:, 3] = (boxes[:, 3] - top) / h_crop

            return image, boxes, labels


"""
named tuple for storing detection model score and truth
"""
DetectionResult = NamedTuple(
    "DetectionResult",
    [("score", float), ("is_true_positive", bool)],
)


class MeanAveragePrecision(object):
    """
    Class for computing the mean average precision of an object detection model output.
    Inputs will be decoded by the provided post-processing function.
    Each batch update tracks the cumulative ground truth objects of each class, and the
    scores the model gives each class.

    calculate_map object uses the aggregated results to find the mAP at the given
    threshold(s)

    :param postprocessing_fn: function that takes in detection model output and returns
        post-processed tuple of predicted bounding boxes, classification labels, and
        scores
    :param iou_threshold: IoU thresholds to match predicted objects to ground truth
        objects. Can provide a single IoU or a tuple of two representing a range.
        mAP will be averaged over the range of values at each IoU
    :param iou_steps: the amount of IoU to shift between measurements between
        iou_threshold values
    """

    def __init__(
        self,
        postprocessing_fn: Callable[[Any], Tuple[Tensor, Tensor, Tensor]],
        iou_threshold: Union[float, Tuple[float, float]] = 0.5,
        iou_steps: float = 0.05,
    ):
        self._postprocessing_fn = postprocessing_fn
        if isinstance(iou_threshold, float):
            self._iou_thresholds = [iou_threshold]
        else:
            min_threshold, max_threshold = iou_threshold
            steps = abs(round((max_threshold - min_threshold) / iou_steps))
            self._iou_thresholds = [
                min_threshold + (iou_steps * step) for step in range(steps)
            ]
            if self._iou_thresholds[-1] < max_threshold:
                self._iou_thresholds.append(max_threshold)

        # dictionaries used to store model results for mAP calculation
        self._ground_truth_classes_count = defaultdict(int)  # class -> num_expected
        self._detection_results_by_class = defaultdict(
            lambda: defaultdict(list)
        )  # iou_threshold -> class -> results

    def __str__(self):
        iou_thresh = (
            str(self._iou_thresholds[0])
            if len(self._iou_thresholds) == 1
            else "[{}:{}]".format(self._iou_thresholds[0], self._iou_thresholds[-1])
        )
        return "mAP@{}".format(iou_thresh)

    def clear(self):
        """
        Resets the ground truth class count and results dictionaries
        """
        self._ground_truth_classes_count = defaultdict(int)  # class -> num_expected
        self._detection_results_by_class = defaultdict(lambda: defaultdict(list))

    def _update_class_counts(self, actual_labels: Tensor):
        for label in actual_labels.reshape(-1):
            self._ground_truth_classes_count[label.item()] += 1

    def _update_model_results(
        self,
        prediction_is_true_positive: Tensor,
        pred_labels: Tensor,
        pred_scores: Tensor,
        iou_threshold: float,
    ):
        for idx in range(pred_labels.size(0)):
            self._detection_results_by_class[iou_threshold][
                pred_labels[idx].item()
            ].append(
                DetectionResult(
                    score=pred_scores[idx].item(),
                    is_true_positive=prediction_is_true_positive[idx].item() != 0,
                )
            )

    def batch_forward(
        self,
        model_output: Tuple[Tensor, Tensor],
        ground_truth_annotations: List[Tuple[Tensor, Tensor]],
    ):
        """
        Decodes the model outputs using non maximum suppression, then stores the
        number of ground truth objects per class, true positives, and true negatives
        that can be used to calculate the overall mAP in the calculate_map function

        :param model_output: the predictions tuple containing [predicted_boxes,
            predicted_labels] batch size should match length of ground_truth_annotations
        :param ground_truth_annotations: annotations from data loader to compare the
            batch results to, should be
        """
        if box_iou is None:
            raise RuntimeError(
                "Unable to import box_iou from torchvision.ops try upgrading your"
                " torch and torchvision versions"
            )

        # run postprocessing / nms
        nms_results = self._postprocessing_fn(model_output)

        # match nms results to ground truth objects for each batch image and store
        for prediction, annotations in zip(nms_results, ground_truth_annotations):
            actual_boxes, actual_labels = annotations

            self._update_class_counts(actual_labels)

            if prediction is None or len(prediction) == 0:
                continue
            pred_boxes, pred_labels, pred_scores = prediction

            if pred_boxes.size(0) == 0:
                continue
            if actual_boxes.size(0) == 0:  # no GTs, all results will be False negative
                prediction_is_true_positive = torch.zeros(pred_labels.shape)
                for threshold in self._iou_thresholds:
                    self._update_model_results(
                        prediction_is_true_positive, pred_labels, pred_scores, threshold
                    )
                continue

            # order predictions by scores
            pred_ranks = torch.argsort(pred_scores, descending=True)
            pred_boxes = pred_boxes[pred_ranks]
            pred_labels = pred_labels[pred_ranks]
            pred_scores = pred_scores[pred_ranks]
            ious = box_iou(pred_boxes, actual_boxes)  # ordered by score on dim 0

            for threshold in self._iou_thresholds:
                prediction_is_true_positive = MeanAveragePrecision._get_true_positives(
                    pred_labels, actual_labels, ious, threshold
                )
                self._update_model_results(
                    prediction_is_true_positive, pred_labels, pred_scores, threshold
                )

    def calculate_map(
        self, num_recall_levels: int = 11
    ) -> Tuple[float, Dict[float, Dict[int, float]]]:
        """
        Calculates mAP at the given threshold values based on the results stored in
        forward passes

        :param num_recall_levels: the number of recall levels to use between
            0 and 1 inclusive. Defaults to 11, the VOC dataset standard
        :return: tuple of the overall mAP, and a dictionary that maps threshold level
            to class to average precision for that class
        """
        threshold_maps = []
        threshold_aps_by_class = []

        recall_levels = MeanAveragePrecision.get_recall_levels(num_recall_levels)

        for threshold in self._iou_thresholds:
            aps_by_class = {}
            for label, results in self._detection_results_by_class[threshold].items():
                if self._ground_truth_classes_count[label] == 0:
                    continue
                results = sorted(results, key=lambda r: r.score, reverse=True)
                prediction_is_true_positive = [
                    result.is_true_positive for result in results
                ]
                aps_by_class[label] = MeanAveragePrecision._in_class_average_precision(
                    prediction_is_true_positive,
                    self._ground_truth_classes_count[label],
                    recall_levels,
                )
            threshold_aps_by_class.append(aps_by_class)
            aps = list(aps_by_class.values())
            threshold_maps.append(float(sum(aps)) / float(len(aps)))

        mean_average_precision = float(sum(threshold_maps)) / float(len(threshold_maps))
        threshold_aps_by_class = dict(zip(self._iou_thresholds, threshold_aps_by_class))

        return mean_average_precision, threshold_aps_by_class

    @staticmethod
    def _get_true_positives(
        pred_labels: Tensor,
        actual_labels: Tensor,
        ious: Tensor,
        iou_threshold: float,
    ) -> Tensor:
        same_label_mask = pred_labels.unsqueeze(1).expand(
            ious.shape
        ) == actual_labels.unsqueeze(0).expand(
            ious.shape
        )  # same_label_mask.shape == ious.shape

        true_positive_mask = same_label_mask & (ious > iou_threshold)
        # deduplicate any matches
        for idx in range(pred_labels.size(0)):
            matches = torch.nonzero(true_positive_mask[idx], as_tuple=False)
            if matches.size(0) > 0:
                # clear first matched label from all following predictions
                true_positive_mask[(idx + 1) :, matches[0]] = 0
            if matches.size(0) > 1:
                # clear extra matches from current prediction
                true_positive_mask[idx, matches[1:]] = 0

        # return 1 for labels with a match 0 otherwise
        return true_positive_mask.sum(dim=1)

    @staticmethod
    def _interpolated_precision(
        prediction_is_true_positive: List[bool],
        num_ground_truth_objects: int,
    ) -> List[Tuple[float, float]]:
        num_true_positives = 0.0
        interpolated_precisions = defaultdict(float)
        num_ground_truth_objects = float(num_ground_truth_objects)
        for idx, is_true_positive in enumerate(prediction_is_true_positive):
            if is_true_positive:
                num_true_positives += 1.0
            precision = num_true_positives / (idx + 1)  # denominator == TP + FP
            recall = num_true_positives / num_ground_truth_objects
            interpolated_precisions[recall] = max(
                interpolated_precisions[recall], precision
            )  # p_inter(r) = max(r(p))
        sorted_recalls = sorted(interpolated_precisions)  # sort by recall level
        return [(recall, interpolated_precisions[recall]) for recall in sorted_recalls]

    @staticmethod
    def _in_class_average_precision(
        prediction_is_true_positive: List[bool],
        num_ground_truth_class_objects: int,
        recall_levels: List[float],
    ) -> float:
        interpolated_precisions = MeanAveragePrecision._interpolated_precision(
            prediction_is_true_positive, num_ground_truth_class_objects
        )
        if not interpolated_precisions:  # occurs if there are no true positives
            return 0.0

        # get interpolated precision associated with each recall level
        interpolated_precisions_idx = 0
        precision_levels = []

        for recall in recall_levels:
            # updated interpolated_precisions_idx for current recall level
            while (
                recall >= interpolated_precisions[interpolated_precisions_idx][0]
                and interpolated_precisions_idx < len(interpolated_precisions) - 1
            ):
                interpolated_precisions_idx += 1
            # add interpolated precision at current recall level
            precision_levels.append(
                interpolated_precisions[interpolated_precisions_idx][1]
            )

        return sum(precision_levels) / len(precision_levels)

    @staticmethod
    def get_recall_levels(num_recall_levels: int = 11):
        """
        :param num_recall_levels: the number of recall levels to use between
            0 and 1 inclusive. Defaults to 11, the VOC dataset standard
        :return: list of evenly spaced recall levels between 0 and 1 inclusive with
            num_recall_levels elements
        """
        levels = list(range(num_recall_levels))
        num_recall_levels = float(num_recall_levels) - 1
        return [float(level) / num_recall_levels for level in levels]
