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
Code related to convenience functions for controlling the calculation of losses and
metrics.
Additionally adds in support for knowledge distillation
"""

from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as TF
from torch import Tensor
from torch.nn import Module

from sparseml.pytorch.utils.helpers import tensors_module_forward
from sparseml.pytorch.utils.yolo_helpers import (
    box_giou,
    build_targets,
    get_output_grid_shapes,
    yolo_v3_anchor_groups,
)


__all__ = [
    "TEACHER_LOSS_KEY",
    "DEFAULT_LOSS_KEY",
    "LossWrapper",
    "BinaryCrossEntropyLossWrapper",
    "CrossEntropyLossWrapper",
    "InceptionCrossEntropyLossWrapper",
    "KDSettings",
    "KDLossWrapper",
    "SSDLossWrapper",
    "YoloLossWrapper",
    "Accuracy",
    "TopKAccuracy",
]


TEACHER_LOSS_KEY = "__teacher_loss__"
DEFAULT_LOSS_KEY = "__loss__"


class LossWrapper(object):
    """
    Generic loss class for controlling how to feed inputs and compare
    with predictions for standard loss functions and metrics.

    :param loss_fn: the loss function to calculate on forward call of this object,
        accessible in the returned Dict at DEFAULT_LOSS_KEY
    :param extras: extras representing other metrics that should be calculated
        in addition to the loss
    :param deconstruct_tensors: True to break the tensors up into expected
        predictions and labels, False to pass the tensors as is to loss and extras
    """

    def __init__(
        self,
        loss_fn: Callable[[Any, Any], Tensor],
        extras: Union[None, Dict[str, Callable]] = None,
        deconstruct_tensors: bool = True,
    ):
        super(LossWrapper, self).__init__()
        self._loss_fn = loss_fn
        self._extras = extras
        self._deconstruct_tensors = deconstruct_tensors

    def __call__(self, data: Any, pred: Any) -> Dict[str, Tensor]:
        return self.forward(data, pred)

    def __repr__(self):
        def _create_repr(_obj: Any) -> str:
            if hasattr(_obj, "__name__"):
                return _obj.__name__

            if hasattr(_obj, "__class__"):
                return _obj.__class__.__name__

            return str(_obj)

        extras = (
            [_create_repr(extra) for extra in self._extras.values()]
            if self._extras is not None
            else []
        )

        return "{}(Loss: {}; Extras: {})".format(
            self.__class__.__name__, _create_repr(self._loss_fn), ",".join(extras)
        )

    @property
    def available_losses(self) -> Tuple[str, ...]:
        """
        :return: a collection of all the loss and metrics keys available
            for this instance
        """
        return (DEFAULT_LOSS_KEY, *list(self._extras.keys()))

    def forward(self, data: Any, pred: Any) -> Dict[str, Tensor]:
        """
        :param data: the input data to the model, expected to contain the labels
        :param pred: the predicted output from the model
        :return: a dictionary containing all calculated losses and metrics with
            the loss from the loss_fn at DEFAULT_LOSS_KEY
        """
        calculated = {
            DEFAULT_LOSS_KEY: self._loss_fn(
                self.get_preds(data, pred, DEFAULT_LOSS_KEY),
                self.get_labels(data, pred, DEFAULT_LOSS_KEY),
            )
        }

        if self._extras:
            for extra, func in self._extras.items():
                calculated[extra] = func(
                    self.get_preds(data, pred, extra),
                    self.get_labels(data, pred, extra),
                )

        return calculated

    def get_preds(self, data: Any, pred: Any, name: str) -> Any:
        """
        overridable function that is responsible for extracting the predictions
        from a model's output

        :param data: data from a data loader
        :param pred: the prediction from the model, if it is a tensor returns this,
            if it is an iterable returns first
        :param name: the name of the loss function that is asking for the
            information for calculation
        :return: the predictions from the model for the loss function
        """
        if isinstance(pred, Tensor) or not self._deconstruct_tensors:
            return pred

        # assume that the desired prediction for loss is in the first instance
        if isinstance(pred, Iterable):
            for tens in pred:
                return tens

        raise TypeError(
            "unsupported type of pred given of {}".format(pred.__class__.__name__)
        )

    def get_labels(self, data: Any, pred: Any, name: str) -> Any:
        """
        overridable function that is responsible for extracting the labels
        for the loss calculation from the input data to the model

        :param data: data from a data loader, expected to contain a tuple of
            (features, labels)
        :param pred: the predicted output from a model
        :param name: the name of the loss function that is asking for the
            information for calculation
        :return: the label for the data
        """
        if isinstance(data, Iterable) and not isinstance(data, Tensor):
            labels = None

            for tens in data:
                labels = tens

            if labels is not None:
                return labels

        raise TypeError(
            "unsupported type of data given of {}".format(data.__class__.__name__)
        )


class BinaryCrossEntropyLossWrapper(LossWrapper):
    """
    Convenience class for doing binary cross entropy loss calculations,
    ie the default loss function is TF.binary_cross_entropy_with_logits.

    :param extras: extras representing other metrics that should be calculated
        in addition to the loss
    """

    def __init__(
        self,
        extras: Union[None, Dict] = None,
    ):
        super().__init__(
            TF.binary_cross_entropy_with_logits,
            extras,
        )


class CrossEntropyLossWrapper(LossWrapper):
    """
    Convenience class for doing cross entropy loss calculations,
    ie the default loss function is TF.cross_entropy.

    :param extras: extras representing other metrics that should be calculated
        in addition to the loss
    """

    def __init__(
        self,
        extras: Union[None, Dict] = None,
    ):
        super().__init__(TF.cross_entropy, extras)


class InceptionCrossEntropyLossWrapper(LossWrapper):
    """
    Loss wrapper for training an inception model that has an aux output
    with cross entropy.

    Defines the loss in the following way:
    aux_weight * cross_entropy(aux_pred, lab) + cross_entropy(pred, lab)

    Additionally adds cross_entropy into the extras.

    :param extras: extras representing other metrics that should be calculated
        in addition to the loss
    :param aux_weight: the weight to use for the cross_entropy value
        calculated from the aux output
    """

    def __init__(
        self,
        extras: Union[None, Dict] = None,
        aux_weight: float = 0.4,
    ):
        if extras is None:
            extras = {}

        extras["cross_entropy"] = TF.cross_entropy
        self._aux_weight = aux_weight

        super().__init__(self.loss, extras)

    def loss(self, preds: Tuple[Tensor, Tensor], labels: Tensor):
        """
        Loss function for inception to combine the overall outputs from the model
        along with the the auxiliary loss from an earlier point in the model

        :param preds: the predictions tuple containing [aux output, output]
        :param labels: the labels to compare to
        :return: the combined cross entropy value
        """

        aux_loss = TF.cross_entropy(preds[0], labels)
        loss = TF.cross_entropy(preds[1], labels)

        return loss + self._aux_weight * aux_loss

    def get_preds(
        self, data: Any, pred: Tuple[Tensor, Tensor, Tensor], name: str
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Override get_preds for the inception training output.
        Specifically expects the pred from the model to be a three tensor tuple:
        (aux logits, logits, classes)

        For the loss function returns a tuple containing (aux logits, logits),
        for all other extras returns the logits tensor

        :param data: data from a data loader
        :param pred: the prediction from an inception model,
            expected to be a tuple containing (aux logits, logits, classes)
        :param name: the name of the loss function that is asking for the
            information for calculation
        :return: the predictions from the model for the loss function;
            a tuple containing (aux logits, logits),
            for all other extras returns the logits tensor
        """
        if name == DEFAULT_LOSS_KEY:
            return pred[0], pred[1]  # return aux, logits for loss function

        return pred[1]  # return logits for other calculations


class KDSettings(object):
    """
    properties class for settings for applying knowledge distillation as
    part of the loss calculation.

    :param teacher: the teacher that provides targets for the student to learn from
    :param temp_student: temperature coefficient for the student
    :param temp_teacher: temperature coefficient for the teacher
    :param weight: the weight for how much of the kd loss to use in proportion
        with the original loss
    :param contradict_hinton: in hinton's original paper they included T^2
        as a scaling factor some implementations dropped this factor
        so contradicting hinton does not scale by T^2
    """

    def __init__(
        self,
        teacher: Module,
        temp_student: float = 5.0,
        temp_teacher: float = 5.0,
        weight: float = 0.5,
        contradict_hinton: bool = False,
    ):
        self._teacher = teacher
        self._temp_student = temp_student
        self._temp_teacher = temp_teacher
        self._weight = weight
        self._contradict_hinton = contradict_hinton

    @property
    def teacher(self) -> Module:
        """
        :return: the teacher that provides targets for the student to learn from
        """
        return self._teacher

    @property
    def temp_student(self) -> float:
        """
        :return: temperature coefficient for the student
        """
        return self._temp_student

    @property
    def temp_teacher(self) -> float:
        """
        :return: temperature coefficient for the teacher
        """
        return self._temp_teacher

    @property
    def weight(self) -> float:
        """
        :return: the weight for how much of the kd loss to use in proportion
            with the original loss
        """
        return self._weight

    @property
    def contradict_hinton(self) -> bool:
        """
        :return: in hinton's original paper they included T^2 as a scaling factor
            some implementations dropped this factor so contradicting hinton
            does not scale by T^2
        """
        return self._contradict_hinton


class KDLossWrapper(LossWrapper):
    """
    Special case of the loss wrapper that allows knowledge distillation.
    Makes some assumptions specifically for image classification tasks,
    so may not work out of the box for everything.

    :param loss_fn: the loss function to calculate on forward call of this object,
        accessible in the returned Dict at DEFAULT_LOSS_KEY
    :param extras: extras representing other metrics that should be
        calculated in addition to the loss
    :param deconstruct_tensors: True to break the tensors up into expected
        predictions and labels, False to pass the tensors as is to loss and extras
    :param kd_settings: the knowledge distillation settings that guide
        how to calculate the total loss
    """

    def __init__(
        self,
        loss_fn: Callable[[Any, Any], Tensor],
        extras: Union[None, Dict[str, Callable]] = None,
        deconstruct_tensors: bool = True,
        kd_settings: Union[None, KDSettings] = None,
    ):
        super().__init__(loss_fn, extras, deconstruct_tensors)
        self._kd_settings = kd_settings  # type: KDSettings

    def get_inputs(self, data: Any, pred: Any, name: str) -> Any:
        """
        overridable function that is responsible for extracting the inputs to the model
        from the input data to the model and the output from the model

        :param data: data from a data loader, expected to contain a tuple of
            (features, labels)
        :param pred: the predicted output from a model
        :param name: the name of the loss function that is asking for the information
            for calculation
        :return: the input data for the model
        """
        if isinstance(data, Tensor):
            return data

        if isinstance(data, Iterable):
            for tens in data:
                return tens

        raise TypeError(
            "unsupported type of data given of {}".format(data.__class__.__name__)
        )

    def forward(self, data: Any, pred: Any) -> Dict[str, Tensor]:
        """
        override to calculate the knowledge distillation loss if kd_settings
        is supplied and not None

        :param data: the input data to the model, expected to contain the labels
        :param pred: the predicted output from the model
        :return: a dictionary containing all calculated losses and metrics with
            the loss from the loss_fn at DEFAULT_LOSS_KEY
        """
        losses = super().forward(data, pred)

        if self._kd_settings is not None:
            with torch.no_grad():
                teacher = self._kd_settings.teacher  # type: Module
                preds_teacher = tensors_module_forward(
                    self.get_inputs(data, pred, TEACHER_LOSS_KEY), teacher.eval()
                )

            preds_teacher = self.get_preds(data, preds_teacher, TEACHER_LOSS_KEY)

            soft_log_probs = TF.log_softmax(
                self.get_preds(data, pred, DEFAULT_LOSS_KEY)
                / self._kd_settings.temp_student,
                dim=1,
            )
            soft_targets = TF.softmax(
                preds_teacher / self._kd_settings.temp_teacher, dim=1
            )
            distill_loss = (
                TF.kl_div(soft_log_probs, soft_targets, size_average=False)
                / soft_targets.shape[0]
            )

            if not self._kd_settings.contradict_hinton:
                # in hinton's original paper they included T^2 as a scaling factor
                # some implementations dropped this factor
                # so contradicting hinton does not scale by T^2
                distill_loss = (
                    (self._kd_settings.temp_student + self._kd_settings.temp_teacher)
                    / 2
                ) ** 2 * distill_loss

            losses[DEFAULT_LOSS_KEY] = (
                self._kd_settings.weight * distill_loss
                + (1 - self._kd_settings.weight) * losses[DEFAULT_LOSS_KEY]
            )

        return losses


class SSDLossWrapper(LossWrapper):
    """
    Loss wrapper for SSD models.  Implements the loss as the sum of:

    1. Confidence Loss: All labels, with hard negative mining
    2. Localization Loss: Only on positive labels

    :param extras: extras representing other metrics that should be
        calculated in addition to the loss
    """

    def __init__(
        self,
        extras: Union[None, Dict] = None,
    ):
        if extras is None:
            extras = {}

        self._localization_loss = nn.SmoothL1Loss(reduction="none")
        self._confidence_loss = nn.CrossEntropyLoss(reduction="none")

        super().__init__(self.loss, extras)

    def loss(self, preds: Tuple[Tensor, Tensor], labels: Tuple[Tensor, Tensor, Tensor]):
        """
        Calculates the loss for a multibox SSD output as the sum of the confidence
        and localization loss for the positive samples in the predictor with hard
        negative mining.

        :param preds: the predictions tuple containing [predicted_boxes,
            predicted_lables].
        :param labels: the labels to compare to
        :return: the combined location and confidence losses
        """
        # extract predicted and ground truth boxes / labels
        predicted_boxes, predicted_scores = preds
        ground_boxes, ground_labels, _ = labels

        # create positive label mask and count positive samples
        positive_mask = ground_labels > 0
        num_pos_labels = positive_mask.sum(dim=1)  # shape: BATCH_SIZE,1

        # sum loss on localization values, and mask out negative results
        loc_loss = self._localization_loss(predicted_boxes, ground_boxes).sum(dim=1)
        loc_loss = (positive_mask.float() * loc_loss).sum(dim=1)

        # confidence loss with hard negative mining
        con_loss_init = self._confidence_loss(predicted_scores, ground_labels)

        # create mask to select 3 negative samples for every positive sample per image
        con_loss_neg_vals = con_loss_init.clone()
        con_loss_neg_vals[positive_mask] = 0  # clear positive sample values

        _, neg_sample_sorted_idx = con_loss_neg_vals.sort(dim=1, descending=True)
        _, neg_sample_rank = neg_sample_sorted_idx.sort(dim=1)  # ascending value rank
        neg_threshold = torch.clamp(  # set threshold to 3x number of positive samples
            3 * num_pos_labels, max=positive_mask.size(1)
        ).unsqueeze(-1)
        neg_mask = neg_sample_rank < neg_threshold  # select samples with highest loss

        con_loss = (con_loss_init * (positive_mask.float() + neg_mask.float())).sum(
            dim=1
        )

        # take average total loss over number of positive samples
        # sets loss to 0 for images with no objects
        total_loss = loc_loss + con_loss
        pos_label_mask = (num_pos_labels > 0).float()
        num_pos_labels = num_pos_labels.float().clamp(min=1e-6)

        return (total_loss * pos_label_mask / num_pos_labels).mean(dim=0)

    def get_preds(
        self, data: Any, pred: Tuple[Tensor, Tensor], name: str
    ) -> Tuple[Tensor, Tensor]:
        """
        Override get_preds for SSD model output.

        :param data: data from a data loader
        :param pred: the prediction from an ssd model: two tensors
            representing object location and object label respectively
        :param name: the name of the loss function that is asking for the
            information for calculation
        :return: the predictions from the model without any changes
        """
        return pred[0], pred[1]  # predicted locations, predicted labels


class YoloLossWrapper(LossWrapper):
    """
    Loss wrapper for Yolo models.  Implements the loss as a sum of class loss,
    objectness loss, and GIoU

    :param extras: extras representing other metrics that should be
        calculated in addition to the loss
    :param anchor_groups: List of n,2 tensors of the Yolo model's anchor points
        for each output group
    """

    def __init__(
        self,
        extras: Union[None, Dict] = None,
        anchor_groups: List[Tensor] = None,
    ):
        if extras is None:
            extras = {}
        self.anchor_groups = anchor_groups or yolo_v3_anchor_groups()

        self.class_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1.0]))
        self.obj_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1.0]))

        super().__init__(self.loss, extras)

    def loss(self, preds: List[Tensor], labels: Tuple[Tensor, Tensor]):
        """
        Calculates the loss for a Yolo model output as the sum of the box, object,
        and class losses

        :param preds: the predictions list containing objectness, class, and location
            values for each detector in the Yolo model.
        :param labels: the labels to compare to
        :return: the combined box, object, and class losses
        """
        targets, _ = labels

        grid_shapes = get_output_grid_shapes(preds)
        target_classes, target_boxes, target_indices, anchors = build_targets(
            targets, self.anchor_groups, grid_shapes
        )

        device = targets.device
        self.class_loss_fn = self.class_loss_fn.to(device)
        self.obj_loss_fn = self.obj_loss_fn.to(device)

        class_loss = torch.zeros(1, device=device)
        box_loss = torch.zeros(1, device=device)
        object_loss = torch.zeros(1, device=device)

        num_targets = 0
        object_loss_balance = [4.0, 1.0, 0.4, 0.1]  # usually only first 3 used

        for i, pred in enumerate(preds):
            image, anchor, grid_x, grid_y = target_indices[i]
            target_object = torch.zeros_like(pred[..., 0], device=device)

            if image.shape[0]:
                num_targets += image.shape[0]

                # filter for predictions on actual objects
                predictions = pred[image, anchor, grid_x, grid_y]

                # box loss
                predictions_xy = predictions[:, :2].sigmoid() * 2.0 - 0.5
                predictions_wh = (predictions[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                predictions_box = torch.cat((predictions_xy, predictions_wh), 1).to(
                    device
                )
                giou = box_giou(predictions_box.T, target_boxes[i].T)

                box_loss += (1.0 - giou).mean()

                # giou
                target_object[image, anchor, grid_x, grid_y] = (
                    giou.detach().clamp(0).type(target_object.dtype)
                )

                # class loss
                target_class_mask = torch.zeros_like(predictions[:, 5:], device=device)
                target_class_mask[range(image.shape[0]), target_classes[i]] = 1.0
                class_loss += self.class_loss_fn(predictions[:, 5:], target_class_mask)

            object_loss += (
                self.obj_loss_fn(pred[..., 4], target_object) * object_loss_balance[i]
            )
        # scale losses
        box_loss *= 0.05
        class_loss *= 0.5
        batch_size = preds[0].shape[0]

        loss = batch_size * (box_loss + object_loss + class_loss)
        return loss, box_loss, object_loss, class_loss

    def forward(self, data: Any, pred: Any) -> Dict[str, Tensor]:
        """
        :param data: the input data to the model, expected to contain the labels
        :param pred: the predicted output from the model
        :return: a dictionary containing all calculated losses (default, giou,
            object, and class) and metrics with the loss from the loss_fn at
            DEFAULT_LOSS_KEY
        """
        loss, box_loss, object_loss, class_loss = self._loss_fn(
            self.get_preds(data, pred, DEFAULT_LOSS_KEY),
            self.get_labels(data, pred, DEFAULT_LOSS_KEY),
        )
        calculated = {
            DEFAULT_LOSS_KEY: loss,
            "giou": box_loss,
            "object": object_loss,
            "classification": class_loss,
        }

        if self._extras:
            for extra, func in self._extras.items():
                calculated[extra] = func(
                    self.get_preds(data, pred, extra),
                    self.get_labels(data, pred, extra),
                )

        return calculated

    def get_preds(self, data: Any, pred: List[Tensor], name: str) -> List[Tensor]:
        """
        Override get_preds for SSD model output.

        :param data: data from a data loader
        :param pred: the prediction from an ssd model: two tensors
            representing object location and object label respectively
        :param name: the name of the loss function that is asking for the
            information for calculation
        :return: the predictions from the model without any changes
        """
        return pred


class Accuracy(Module):
    """
    Class for calculating the accuracy for a given prediction and the labels
    for comparison.
    Expects the inputs to be from a range of 0 to 1 and sets a crossing threshold at 0.5
    the labels are similarly rounded.
    """

    def forward(self, pred: Tensor, lab: Tensor) -> Tensor:
        """
        :param pred: the models prediction to compare with
        :param lab: the labels for the data to compare to
        :return: the calculated accuracy
        """
        return Accuracy.calculate(pred, lab)

    @staticmethod
    def calculate(pred: Tensor, lab: Tensor):
        """
        :param pred: the models prediction to compare with
        :param lab: the labels for the data to compare to
        :return: the calculated accuracy
        """
        pred = pred >= 0.5
        lab = lab >= 0.5
        correct = (pred == lab).sum()
        total = lab.numel()
        acc = correct.float() / total * 100.0

        return acc


class TopKAccuracy(Module):
    """
    Class for calculating the top k accuracy for a given prediction and the labels for
    comparison; ie the top1 or top5 accuracy. top1 is equivalent to the Accuracy class

    :param topk: the numbers of buckets the model is considered to be correct within
    """

    def __init__(self, topk: int = 1):
        super(TopKAccuracy, self).__init__()
        self._topk = topk

    def forward(self, pred: Tensor, lab: Tensor) -> Tensor:
        """
        :param pred: the models prediction to compare with
        :param lab: the labels for the data to compare to
        :return: the calculated topk accuracy
        """
        return TopKAccuracy.calculate(pred, lab, self._topk)

    @staticmethod
    def calculate(pred: Tensor, lab: Tensor, topk: int):
        """
        :param pred: the models prediction to compare with
        :param lab: the labels for the data to compare to
        :param topk: the number of bins to be within for the correct label
        :return: the calculated topk accuracy
        """
        with torch.no_grad():
            batch_size = lab.size(0)

            _, pred = pred.topk(topk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(lab.view(1, -1).expand_as(pred))

            correct_k = (
                correct[:topk].contiguous().view(-1).float().sum(0, keepdim=True)
            )
            correct_k = correct_k.mul_(100.0 / batch_size)

            return correct_k
