"""
Contains code for epoch modifiers in Keras
"""

from sparseml.keras.optim.modifier import (
    ScheduledModifier,
    KerasModifierYAML,
)


__all__ = ["EpochRangeModifier"]


@KerasModifierYAML()
class EpochRangeModifier(ScheduledModifier):
    """
    Simple modifier to set the range of epochs to train over for
    the recalibration process.
    Note, that if other modifiers exceed the range of this one for min or max epochs,
    this modifier will not have an effect.

    | Sample yaml:
    |   !EpochRangeModifier:
    |       start_epoch: 0
    |       end_epoch: 90
    """

    def __init__(
        self,
        start_epoch: float,
        end_epoch: float,
    ):
        """
        :param start_epoch: The epoch to start the modifier at
        :param end_epoch: The epoch to end the modifier at
        """
        super().__init__(
            start_epoch=start_epoch, end_epoch=end_epoch, end_comparator=-1
        )
