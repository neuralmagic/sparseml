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
        # sort the modifiers so they are iterated in order of their start epoch
        # if start epoch is the same, end epoch is used to break ties
        # with ending first running first
        self._modifiers = sorted(
            modifiers, key=lambda m: m.start_epoch + m.end_epoch * 1e-6
        )

    def __del__(self):
        for mod in self._modifiers:
            del mod

        self._modifiers.clear()

    def __str__(self) -> str:
        man_str = "\nversion: 1.1.0\n\n"
        man_str += super().__str__()

        for mod in self._modifiers:
            mod_lines = str(mod).splitlines()

            for line in mod_lines:
                man_str += "\t{}\n".format(line)

            man_str += "\n"

        return man_str

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
