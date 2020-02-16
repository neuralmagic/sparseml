"""
Contains code for epoch modifiers
"""

import yaml

from .modifier import ScheduledModifier


__all__ = ["EpochRangeModifier"]


class EpochRangeModifier(ScheduledModifier):
    """
    Simple modifier to set the range of epochs for running in a scheduled optimizer
    (ie to set min and max epochs within a range without hacking other modifiers)

    Note, that if other modifiers exceed the range of this one for min or max epochs,
    this modifier will not have an effect

    Sample yaml:
        !EpochRangeModifier:
            start_epoch: 0
            end_epoch: 90
    """

    YAML_KEY = u"!EpochRangeModifier"

    @staticmethod
    def yaml_constructor(loader, node):
        instance = EpochRangeModifier.__new__(EpochRangeModifier)
        yield instance
        state = loader.construct_mapping(node, deep=True)
        instance.__init__(**state)

    def __init__(
        self, start_epoch: float, end_epoch: float,
    ):
        """
        :param start_epoch: The epoch to start the modifier at
        :param end_epoch: The epoch to end the modifier at
        """
        super().__init__(start_epoch, end_epoch)

    def __repr__(self):
        return "{}(start_epoch={}, end_epoch={})".format(
            self.__class__.__name__, self._start_epoch, self._end_epoch,
        )


yaml.add_constructor(EpochRangeModifier.YAML_KEY, EpochRangeModifier.yaml_constructor)
yaml.add_constructor(
    EpochRangeModifier.YAML_KEY, EpochRangeModifier.yaml_constructor, yaml.SafeLoader,
)
