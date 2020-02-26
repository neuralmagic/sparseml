"""
Code related to convenience functions for controlling the calculation of losses and metrics
Additionally adds in support for knowledge distillation
"""

from typing import Dict, Union, Callable, Tuple, Any, Iterable
import torch
import torch.nn.functional as TF
from torch import Tensor
from torch.nn import Module

from neuralmagicML.pytorch.utils.helpers import tensors_module_forward


__all__ = [
    "TEACHER_LOSS_KEY",
    "DEFAULT_LOSS_KEY",
    "LossWrapper",
    "KDSettings",
    "KDLossWrapper",
    "BinaryCrossEntropyLossWrapper",
    "CrossEntropyLossWrapper",
    "Accuracy",
    "TopKAccuracy",
]


TEACHER_LOSS_KEY = "__teacher_loss__"
DEFAULT_LOSS_KEY = "__loss__"


class LossWrapper(object):
    """
    Generic loss class for controlling how to feed inputs and compare with predictions for
    standard loss functions and metrics
    """

    def __init__(
        self,
        loss_fn: Callable[[Any, Any], Tensor],
        extras: Union[None, Dict[str, Callable]] = None,
    ):
        """
        :param loss_fn: the loss function to calculate on forward call of this object,
                        accessible in the returned Dict at DEFAULT_LOSS_KEY
        :param extras: extras representing other metrics that should be calculated in addition to the loss
        """
        super(LossWrapper, self).__init__()
        self._loss_fn = loss_fn
        self._extras = extras

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
        :return: a collection of all the loss and metrics keys available for this instance
        """
        return (DEFAULT_LOSS_KEY, *list(self._extras.keys()))

    def forward(self, data: Any, pred: Any) -> Dict[str, Tensor]:
        """
        :param data: the input data to the model, expected to contain the labels
        :param pred: the predicted output from the model
        :return: a dictionary containing all calculated losses and metrics with the loss from the loss_fn at
                 DEFAULT_LOSS_KEY
        """
        calculated = {
            DEFAULT_LOSS_KEY: self.calc_loss(
                self.get_inputs(data, pred, DEFAULT_LOSS_KEY),
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

    def get_inputs(self, data: Any, pred: Any, name: str) -> Any:
        """
        overridable function that is responsible for extracting the inputs to the model from the
        input data to the model and the output from the model

        :param data: data from a data loader, expected to contain a tuple of (features, labels)
        :param pred: the predicted output from a model
        :param name: the name of the loss function that is asking for the information for calculation
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

    def get_preds(self, data: Any, pred: Any, name: str) -> Any:
        """
        overridable function that is responsible for extracting the predictions from a model's output

        :param data: data from a data loader
        :param pred: the prediction from the model, if it is a tensor returns this, if it is an iterable returns first
        :param name: the name of the loss function that is asking for the information for calculation
        :return: the predictions from the model for the loss function
        """
        if isinstance(pred, Tensor):
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
        overridable function that is responsible for extracting the labels for the loss calculation from the
        input data to the model

        :param data: data from a data loader, expected to contain a tuple of (features, labels)
        :param pred: the predicted output from a model
        :param name: the name of the loss function that is asking for the information for calculation
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

    def calc_loss(self, inputs: Any, preds: Any, labels: Any) -> Tensor:
        """
        overridable function to calculate the default loss function for training the model

        :param inputs: the inputs to the model, taken from get_inputs
        :param preds: the predictions from the model, taken from get_preds
        :param labels: the labels for the data, taken from get_labels
        :return: the resulting calculated loss from the default loss_fn
        """
        return self._loss_fn(preds, labels)


class KDSettings(object):
    """
    properties class for settings for applying knowledge distillation as part of the loss calculation
    """

    def __init__(
        self,
        teacher: Module,
        temp_student: float = 5.0,
        temp_teacher: float = 5.0,
        weight: float = 0.5,
        contradict_hinton: bool = False,
    ):
        """
        :param teacher: the teacher that provides targets for the student to learn from
        :param temp_student: temperature coefficient for the student
        :param temp_teacher: temperature coefficient for the teacher
        :param weight: the weight for how much of the kd loss to use in proportion with the original loss
        :param contradict_hinton: in hinton's original paper they included T^2 as a scaling factor
                                  some implementations dropped this factor so contradicting hinton does not scale by T^2
        """
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
        :return: the weight for how much of the kd loss to use in proportion with the original loss
        """
        return self._weight

    @property
    def contradict_hinton(self) -> bool:
        """
        :return: in hinton's original paper they included T^2 as a scaling factor
                 some implementations dropped this factor so contradicting hinton does not scale by T^2
        """
        return self._contradict_hinton


class KDLossWrapper(LossWrapper):
    """
    Special case of the loss wrapper that allows knowledge distillation
    Makes some assumptions specifically for image classification tasks, so may not work out of the box for everything
    """

    def __init__(
        self,
        loss_fn: Callable[[Any, Any], Tensor],
        extras: Union[None, Dict[str, Callable]] = None,
        kd_settings: Union[None, KDSettings] = None,
    ):
        """
        :param loss_fn: the loss function to calculate on forward call of this object,
                        accessible in the returned Dict at DEFAULT_LOSS_KEY
        :param extras: extras representing other metrics that should be calculated in addition to the loss
        :param kd_settings: the knowledge distillation settings that guide how to calculate the total loss
        """
        super(KDLossWrapper, self).__init__(loss_fn, extras)
        self._kd_settings = kd_settings  # type: KDSettings

    def calc_loss(self, inputs: Any, preds: Any, labels: Any) -> Tensor:
        """
        override to calculate the knowledge distillation loss if kd_settings is supplied and not None

        :param inputs: the inputs to the model, taken from get_inputs
        :param preds: the predictions from the model, taken from get_preds
        :param labels: the labels for the data, taken from get_labels
        :return: the resulting calculated loss from the default loss_fn combined with the distillation loss if
                 kd_settings is not None
        """
        loss = super().calc_loss(inputs, preds, labels)

        if self._kd_settings is None:
            return loss

        with torch.no_grad():
            teacher = self._kd_settings.teacher  # type: Module
            preds_teacher = tensors_module_forward(inputs, teacher.eval())

        preds_teacher = self.get_preds(None, preds_teacher, TEACHER_LOSS_KEY)

        soft_log_probs = TF.log_softmax(preds / self._kd_settings.temp_student, dim=1)
        soft_targets = TF.softmax(preds_teacher / self._kd_settings.temp_teacher, dim=1)
        distill_loss = (
            TF.kl_div(soft_log_probs, soft_targets, size_average=False)
            / soft_targets.shape[0]
        )

        if not self._kd_settings.contradict_hinton:
            # in hinton's original paper they included T^2 as a scaling factor
            # some implementations dropped this factor
            # so contradicting hinton does not scale by T^2
            distill_loss = (
                (self._kd_settings.temp_student + self._kd_settings.temp_teacher) / 2
            ) ** 2 * distill_loss

        return (
            self._kd_settings.weight * distill_loss
            + (1 - self._kd_settings.weight) * loss
        )


class BinaryCrossEntropyLossWrapper(KDLossWrapper):
    """
    Convenience class for doing binary cross entropy loss calculations,
    ie the default loss function is TF.binary_cross_entropy_with_logits
    """

    def __init__(
        self,
        extras: Union[None, Dict] = None,
        kd_settings: Union[None, KDSettings] = None,
    ):
        """
        :param extras: extras representing other metrics that should be calculated in addition to the loss
        :param kd_settings: the knowledge distillation settings that guide how to calculate the total loss if
                            knowledge distillation is desired to be used
        """
        super(BinaryCrossEntropyLossWrapper, self).__init__(
            TF.binary_cross_entropy_with_logits, extras, kd_settings
        )


class CrossEntropyLossWrapper(KDLossWrapper):
    """
    Convenience class for doing cross entropy loss calculations,
    ie the default loss function is TF.cross_entropy
    """

    def __init__(
        self,
        extras: Union[None, Dict] = None,
        kd_settings: Union[None, KDSettings] = None,
    ):
        """
        :param extras: extras representing other metrics that should be calculated in addition to the loss
        :param kd_settings: the knowledge distillation settings that guide how to calculate the total loss if
                            knowledge distillation is desired to be used
        """
        super(CrossEntropyLossWrapper, self).__init__(
            TF.cross_entropy, extras, kd_settings
        )


class Accuracy(Module):
    """
    Class for calculating the accuracy for a given prediction and the labels for comparison
    Expects the inputs to be from a range of 0 to 1 and sets a crossing threshold at 0.5
    the labels are similarly rounded
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
    Class for calculating the top k accuracy for a given prediction and the labels for comparison;
    ie the top1 or top5 accuracy. top1 is equivalent to the Accuracy class
    """

    def __init__(self, topk: int = 1):
        """
        :param topk: the numbers of buckets the model is considered to be correct within
        """
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

            correct_k = correct[:topk].view(-1).float().sum(0, keepdim=True)
            correct_k = correct_k.mul_(100.0 / batch_size)

            return correct_k
