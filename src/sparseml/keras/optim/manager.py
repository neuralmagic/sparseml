"""
Contains base code related to modifier managers: modifier managers handle
grouping modifiers and running them together.
Also handles loading modifiers from yaml files
"""

from typing import List, Union

import tensorflow as tf
from sparseml.keras.optim.modifier import Modifier, ScheduledModifier
from sparseml.optim import BaseManager


__all__ = ["ScheduledModifierManager"]


class ScheduledModifierManager(BaseManager, Modifier):
    """
    The base modifier manager, handles managing multiple ScheduledModifier.
    """

    @staticmethod
    def from_yaml(file_path: str, add_modifiers: List[Modifier] = None):
        """
        Convenience function used to create the manager of multiple modifiers
        from a yaml file.

        :param file_path: the path to the yaml file to load the modifier from
        :param add_modifiers: additional modifiers that should be added to the
            returned manager alongside the ones loaded from the yaml file
        :return: ScheduledModifierManager() created from the yaml file
        """
        with open(file_path, "r") as yaml_file:
            yaml_str = yaml_file.read()

        modifiers = Modifier.load_list(yaml_str)
        if add_modifiers:
            modifiers.extend(add_modifiers)

        manager = ScheduledModifierManager(modifiers)

        return manager

    def __init__(self, modifiers: List[ScheduledModifier]):
        super().__init__(modifiers=modifiers)

    def modify(
        self,
        model: Union[tf.keras.Model, tf.keras.Sequential],
        optimizer: tf.keras.optimizers.Optimizer,
        steps_per_epoch: int,
        input_tensors: tf.Tensor = None,
    ):
        callbacks = []
        for mod in self._modifiers:
            model, optimizer, callback = mod.modify(
                model, optimizer, steps_per_epoch, input_tensors=input_tensors
            )
            if callback is None:
                continue
            if isinstance(callback, list):
                callbacks = callbacks + callback
            elif isinstance(callback, tf.keras.callbacks.Callback):
                callbacks.append(callback)
            else:
                raise RuntimeError("Invalid callback type")
        return model, optimizer, callbacks
