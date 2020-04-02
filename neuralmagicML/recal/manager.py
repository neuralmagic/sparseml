"""
Code related to managers that is shared across frameworks.
Managers control groups of modifiers to allow modifying the training process of a model;
ex to perform model pruning.
"""

from typing import List, Dict
import math

from neuralmagicML.recal.modifier import ModifierProp, BaseScheduled, BaseObject


__all__ = ["BaseManager"]


class BaseManager(BaseObject):
    """
    Parent class meant to be used for all managers.
    Handles base implementations for properties and methods.

    :param modifiers: the modifiers to wrap
    """

    def __init__(self, modifiers: List[BaseScheduled], **kwargs):
        super().__init__(**kwargs)
        self._modifiers = modifiers

    def __del__(self):
        self._modifiers.clear()

    @ModifierProp()
    def modifiers(self) -> Dict[str, List[BaseScheduled]]:
        return {"modifiers": self._modifiers}

    @ModifierProp(serializable=False)
    def modifiers(self) -> List[BaseScheduled]:
        return self._modifiers

    @ModifierProp(serializable=False)
    def min_epochs(self) -> int:
        """
        :return: the minimum epochs required by any of the modifiers under the manager
        """
        vals = []
        vals.extend(
            [
                math.floor(mod.start_epoch)
                for mod in self._modifiers
                if mod.start_epoch > -1
            ]
        )
        vals.extend(
            [math.floor(mod.end_epoch) for mod in self._modifiers if mod.end_epoch > -1]
        )

        return min(vals) if len(vals) > 0 else -1

    @ModifierProp(serializable=False)
    def max_epochs(self) -> int:
        """
        :return: the maximum number of epochs required by any of the modifiers
            under the manager
        """
        vals = []
        vals.extend(
            [
                math.ceil(mod.start_epoch)
                for mod in self._modifiers
                if mod.start_epoch > -1
            ]
        )
        vals.extend(
            [math.ceil(mod.end_epoch) for mod in self._modifiers if mod.end_epoch > -1]
        )

        return max(vals) if len(vals) > 0 else -1
