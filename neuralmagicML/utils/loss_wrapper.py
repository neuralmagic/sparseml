from typing import Dict, Union, Callable, List, Tuple, Any, Iterable
import torch
import torch.nn.functional as TF
from torch import Tensor
from torch.nn import Module


from .module import DEFAULT_LOSS_KEY


__all__ = ['KnowledgeDistillationSettings', 'LossWrapper', 'BinaryCrossEntropyLossWrapper', 'CrossEntropyLossWrapper']


TEACHER_LOSS_KEY = '__teacher_loss__'


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


class LossWrapper(object):
    def __init__(self, loss_fn: Callable[[Any, Any], Tensor], extras: Union[None, Dict[str, Callable]] = None,
                 kd_settings: Union[None, KnowledgeDistillationSettings] = None):
        super(LossWrapper, self).__init__()
        self._loss_fn = loss_fn
        self._extras = extras
        self._kd_settings = kd_settings  # type: KnowledgeDistillationSettings

    def __call__(self,  data: Any, pred: Any) -> Dict[str, Tensor]:
        return self.forward(data, pred)

    def __repr__(self):
        def _create_repr(_obj: Any) -> str:
            if hasattr(_obj, '__name__'):
                return _obj.__name__

            if hasattr(_obj, '__class__'):
                return _obj.__class__.__name__

            return str(_obj)

        extras = [_create_repr(extra) for extra in self._extras.values()] if self._extras is not None else []

        return '{}(Loss: {}; Extras: {})'.format(self.__class__.__name__, _create_repr(self._loss_fn), ','.join(extras))

    @property
    def available_losses(self) -> Tuple[str, ...]:
        return (DEFAULT_LOSS_KEY, *list(self._extras.keys()))

    def forward(self, data: Any, pred: Any) -> Dict[str, Tensor]:
        calculated = {
            DEFAULT_LOSS_KEY: self.calc_loss(
                self.get_inputs(data, pred, DEFAULT_LOSS_KEY),
                self.get_preds(data, pred, DEFAULT_LOSS_KEY),
                self.get_labels(data, pred, DEFAULT_LOSS_KEY)
            )
        }

        if self._extras:
            for extra, func in self._extras.items():
                calculated[extra] = func(
                    self.get_preds(data, pred, extra),
                    self.get_labels(data, pred, extra)
                )

        return calculated

    def get_inputs(self, data: Any, pred: Any, name: str) -> Any:
        if isinstance(data, Tensor):
            return data

        if isinstance(data, Iterable):
            for tens in data:
                return tens

        raise TypeError('unsupported type of data given of {}'.format(data.__class__.__name__))

    def get_preds(self, data: Any, pred: Any, name: str) -> Any:
        if isinstance(pred, Tensor):
            return pred

        # assume that the desired prediction for loss is in the first instance
        if isinstance(pred, Iterable):
            for tens in pred:
                return tens

        raise TypeError('unsupported type of pred given of {}'.format(pred.__class__.__name__))

    def get_labels(self, data: Any, pred: Any, name: str) -> Any:
        if isinstance(data, Iterable):
            labels = None

            for tens in data:
                labels = tens

            if labels is not None:
                return labels

        raise TypeError('unsupported type of data given of {}'.format(data.__class__.__name__))

    def calc_loss(self, inputs: Any, preds: Any, labels: Any) -> Tensor:
        loss = self._loss_fn(preds, labels)

        if self._kd_settings is None:
            return loss

        with torch.no_grad():
            if isinstance(inputs, Tensor):
                preds_teacher = self._kd_settings.teacher(inputs)
            elif isinstance(inputs, Dict):
                preds_teacher = self._kd_settings.teacher(**inputs)
            elif isinstance(inputs, Iterable):
                preds_teacher = self._kd_settings.teacher(*inputs)
            else:
                raise TypeError('unsupported type of inputs given of {}'.format(inputs.__class__.__name__))

        preds_teacher = self.get_preds(None, preds_teacher, TEACHER_LOSS_KEY)

        soft_log_probs = TF.log_softmax(preds / self._kd_settings.temp_student, dim=1)
        soft_targets = TF.softmax(preds_teacher / self._kd_settings.temp_teacher, dim=1)
        distill_loss = TF.kl_div(soft_log_probs, soft_targets, size_average=False) / soft_targets.shape[0]

        if not self._kd_settings.contradict_hinton:
            # in hinton's original paper they included T^2 as a scaling factor
            # some implementations dropped this factor
            # so contradicting hinton does not scale by T^2
            distill_loss = ((self._kd_settings.temp_student + self._kd_settings.temp_teacher) / 2) ** 2 * distill_loss

        return self._kd_settings.weight * distill_loss + (1 - self._kd_settings.weight) * loss


class BinaryCrossEntropyLossWrapper(LossWrapper):
    def __init__(self, extras: Union[None, Dict] = None,
                 kd_settings: Union[None, KnowledgeDistillationSettings] = None):
        super(BinaryCrossEntropyLossWrapper, self).__init__(TF.binary_cross_entropy_with_logits, extras, kd_settings)


class CrossEntropyLossWrapper(LossWrapper):
    def __init__(self, extras: Union[None, Dict] = None,
                 kd_settings: Union[None, KnowledgeDistillationSettings] = None):
        super(CrossEntropyLossWrapper, self).__init__(TF.cross_entropy, extras, kd_settings)
