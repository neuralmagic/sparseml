import math
from typing import Optional, Union, Dict, Any

from sparseml.pytorch.models import ModelRegistry
from sparseml.pytorch.optim import ScheduledModifierManager, ScheduledOptimizer
import torch
import logging

from torch import Module

_LOGGER = logging.getLogger(__name__)


class ImageClassificationTrainer:

    def __init__(
            self,
            checkpoint_path: str,
            recipe_path: str,
    ):
        self._checkpoint_path = checkpoint_path
        self._checkpoint = self._load_checkpoint()

        self.arch_key = self._load_arch_key_from_checkpoint()
        self.recipe_path = recipe_path

        self.model = self._load_model()

    def _load_checkpoint(self) -> Dict[str, Any]:
        # pre-condition: self._checkpoint_path is loaded

        _LOGGER.info(f"Loading checkpoint from {self._checkpoint_path}"
                     f" This might take a while...")
        return torch.load(self._checkpoint_path)

    def _load_arch_key_from_checkpoint(self) -> str:
        # pre-condition: self._checkpoint is loaded

        if 'arch_key' in self._checkpoint:
            return self._checkpoint['arch_key']

        if 'arch-key' in self._checkpoint:
            return self._checkpoint['arch-key']

        raise ValueError(
            f"Could not find `arch_key` in checkpoint {self._checkpoint_path}"
        )

    def _load_model(self) -> Module:
        # pre-condition: self.arch_key is loaded

        model_registry = ModelRegistry()
