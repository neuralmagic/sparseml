from typing import Union, List, Tuple
from torch import Tensor
from torch.nn import Module


class TopKAccuracy(Module):
    def __init__(self, topk: int = 1):
        super(TopKAccuracy, self).__init__()
        self._topk = topk

    def forward(self, pred: Union[Tensor, List[Tensor], Tuple[Tensor, Tensor]], lab: Tensor) -> Tensor:
        if not isinstance(pred, Tensor):
            # returning multiple outputs (like logits and classes)
            # grab first index to check against
            pred = pred[0]

        return topk_accuracy(pred, lab, self._topk)


def topk_accuracy(pred: Tensor, lab: Tensor, topk: int):
    batch_size = lab.size(0)

    _, pred = pred.topk(topk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(lab.view(1, -1).expand_as(pred))
    correct_k = correct[:topk].view(-1).float().sum(0)
    correct_k = correct_k.mul_(100.0 / batch_size)

    return correct_k


class Accuracy(Module):
    def forward(self, pred: Union[Tensor, List[Tensor], Tuple[Tensor, Tensor]], lab: Tensor) -> Tensor:
        if not isinstance(pred, Tensor):
            # returning multiple outputs (like logits and classes)
            # assume second index is supposed to be the classes
            pred = pred[1]

        return accuracy(pred, lab)


def accuracy(pred: Tensor, lab: Tensor):
    pred = pred >= 0.5
    lab = lab >= 0.5
    correct = (pred == lab).sum()
    total = lab.numel()
    acc = correct.float() / total * 100.0

    return acc
