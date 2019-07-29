from typing import Dict, Union, Callable, List, Tuple
import torch
import torch.nn.functional as TF
from torch import Tensor
from torch.nn import Module


class KnowledgeDistillationSettings(object):
    def __init__(self, teacher: Module, temp_student: float = 5.0, temp_teacher: float = 5.0, weight: float = 0.5,
                 contradict_hinton: bool = False):
        self._teacher = teacher.eval()
        self._temp_student = temp_student
        self._temp_teacher = temp_teacher
        self._weight = weight
        self._contradict_hinton = contradict_hinton

    @property
    def teacher(self) -> Module:
        return self._teacher

    @property
    def temp_student(self) -> float:
        return self._temp_student

    @property
    def temp_teacher(self) -> float:
        return self._temp_teacher

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def contradict_hinton(self) -> bool:
        return self._contradict_hinton


class LossCalc(object):
    def __init__(self, loss_fn: Callable, extras: Union[None, Dict[str, Callable]] = None,
                 kd_settings: Union[None, KnowledgeDistillationSettings] = None):
        super(LossCalc, self).__init__()
        self._loss_fn = loss_fn
        self._extras = extras
        self._kd_settings = kd_settings  # type: KnowledgeDistillationSettings

    def __call__(self,  x_feature: Tensor, y_lab: Tensor,
                 y_pred: Union[Tensor, List[Tensor], Tuple[Tensor, Tensor]]) -> Dict[str, Tensor]:
        return self.forward(x_feature, y_lab, y_pred)

    def forward(self, x_feature: Tensor, y_lab: Tensor,
                y_pred: Union[Tensor, List[Tensor], Tuple[Tensor, Tensor]]) -> Dict[str, Tensor]:
        calculated = {
            'loss': self._calc_loss(x_feature, y_lab, y_pred)
        }

        if self._extras:
            for extra, func in self._extras.items():
                calculated[extra] = func(y_pred, y_lab)

        return calculated

    def _calc_loss(self, x_feature: Tensor, y_lab: Tensor,
                   y_pred: Union[Tensor, List[Tensor], Tuple[Tensor, Tensor]]) -> Tensor:
        if not isinstance(y_pred, Tensor):
            # returning multiple outputs (like logits and classes)
            # assume first index is supposed to be the logits
            y_pred = y_pred[0]

        loss = self._loss_fn(y_pred, y_lab)

        if self._kd_settings is None:
            return loss

        with torch.no_grad():
            y_pred_teacher = self._kd_settings.teacher(x_feature)

            if not isinstance(y_pred_teacher, Tensor):
                # returning multiple outputs (like logits and classes)
                # assume first index is supposed to be the logits
                y_pred_teacher = y_pred_teacher[0]

        soft_log_probs = TF.log_softmax(y_pred / self._kd_settings.temp_student, dim=1)
        soft_targets = TF.softmax(y_pred_teacher / self._kd_settings.temp_teacher, dim=1)
        distill_loss = TF.kl_div(soft_log_probs, soft_targets, size_average=False) / soft_targets.shape[0]

        if not self._kd_settings.contradict_hinton:
            # in hinton's original paper they included T^2 as a scaling factor
            # distller and other implementations dropped this factor
            # so contradicting hinton does not scale by T^2
            distill_loss = ((self._kd_settings.temp_student + self._kd_settings.temp_teacher) / 2) ** 2 * distill_loss

        return self._kd_settings.weight * distill_loss + (1 - self._kd_settings.weight) * loss


class BinaryCrossEntropyLossCalc(LossCalc):
    def __init__(self, extras: Union[None, Dict] = None,
                 kd_settings: Union[None, KnowledgeDistillationSettings] = None):
        super(BinaryCrossEntropyLossCalc, self).__init__(TF.binary_cross_entropy_with_logits, extras, kd_settings)


class CrossEntropyLossCalc(LossCalc):
    def __init__(self, extras: Union[None, Dict] = None,
                 kd_settings: Union[None, KnowledgeDistillationSettings] = None):
        super(CrossEntropyLossCalc, self).__init__(TF.cross_entropy, extras, kd_settings)
