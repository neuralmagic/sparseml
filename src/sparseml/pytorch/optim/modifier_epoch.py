"""
Modifiers related to controlling the training epochs while training a model
"""

from sparseml.pytorch.optim.modifier import ScheduledModifier, PyTorchModifierYAML


__all__ = ["EpochRangeModifier"]


@PyTorchModifierYAML()
class EpochRangeModifier(ScheduledModifier):
    """
    Simple modifier to set the range of epochs for running in a scheduled optimizer
    (ie to set min and max epochs within a range without hacking other modifiers).

    Note, that if other modifiers exceed the range of this one for min or max epochs,
    this modifier will not have an effect.

    | Sample yaml:
    |   !EpochRangeModifier:
    |       start_epoch: 0
    |       end_epoch: 90

    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    """

    def __init__(
        self, start_epoch: float, end_epoch: float,
    ):
        super().__init__(
            start_epoch=start_epoch, end_epoch=end_epoch, end_comparator=-1
        )
