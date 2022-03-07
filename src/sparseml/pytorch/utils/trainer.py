import math
from typing import Optional, Union, Dict, Any

from sparseml.pytorch.models import ModelRegistry
from sparseml.pytorch.optim import ScheduledModifierManager, ScheduledOptimizer
import torch
import logging

from torch import Module

_LOGGER = logging.getLogger(__name__)


class TrainingWrapper:
    """
    Helper class for training a model.
    """

    def __init__(
            self,
            checkpoint_path,
            recipe,
            recipe_args: Optional[Union[Dict[str, Any], str]] = None,
            train_loader=None,
            val_loader=None,
            optimizer_name="adam",
            optimizer_args: Optional[Union[Dict[str, Any], str]] = None,
            batch_size=None,
            **kwargs,
    ):
        self.checkpoint_path = checkpoint_path
        self.recipe_args = recipe_args
        self.recipe = recipe

        self._loaded_checkpoint = self._load()
        self.arch_key = self._get_arch_key_from_checkpoint()
        
        self.model = self.load_model()
        
        self.current_epoch = 0
        self._override_start_epoch_from_checkpoint()

        self.manager = self._setup_managers(kwargs=kwargs)


        self.optimizer_name = optimizer_name
        self.optimizer_args = optimizer_args

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = batch_size

        self.optimizer = self.get_optimizer()

        self._manager_applied = False
        self._manager_initialized = False
        self._manager_finalized = False
        self._manager_steps_per_epoch = 0

    def _load(self):
        return torch.load(self.checkpoint_path)

    def _get_arch_key_from_checkpoint(self):
        # pre-condition: self._loaded_checkpoint must be loaded

        if 'arch_key' in self._loaded_checkpoint:
            return self._loaded_checkpoint['arch_key']
        if 'arch-key' in self._loaded_checkpoint:
            return self._loaded_checkpoint['arch-key']
        raise ValueError('Could not find `arch_key` in checkpoint')

    def _override_start_epoch_from_checkpoint(self):
        # pre-condition: self._loaded_checkpoint must be loaded

        if 'epoch' in self._loaded_checkpoint:
            self.current_epoch = self._loaded_checkpoint['epoch']
            _LOGGER.info(
                'Overriding start epoch from checkpoint'
                f' to {self.current_epoch}.'
            )
        else:
            self.current_epoch = 0
            _LOGGER.info(
                'Could not find `epoch` in checkpoint. '
                f'Will start training from epoch {self.current_epoch}.'
            )

    def _setup_managers(self, kwargs):
        manager = None

        if self.recipe is not None:
            manager = ScheduledModifierManager.from_yaml(
                self.recipe, recipe_variables=self.recipe_args
            )
            _LOGGER.info(
                "Loaded SparseML recipe variable into manager for recipe: "
                f"{self.recipe} and recipe_variables: {self.recipe_args}"
            )

        if (
                manager is not None
                and manager.max_epochs
                and "args" in kwargs
                and (hasattr(kwargs["args"], "num_train_epochs"))
        ):
            _LOGGER.warning(
                f"Overriding num_train_epochs from Recipe to {manager.max_epochs}"
            )
            kwargs["args"].num_train_epochs = manager.max_epochs

        return manager

    def finalize_manager(self) -> bool:
        """
        Finalize the current recipes to wrap up any held state.

        :return: True if recipes were finalized, False otherwise
        """
        if (
                self.manager is None
                or not self._manager_initialized
                or self._manager_finalized
        ):
            return False

        self.manager.finalize(self.model)
        self._manager_finalized = True
        _LOGGER.info("Finalized SparseML recipe argument applied to the model")
        return True

    def get_optimizer(self):

        optim_constructor = torch.optim.__dict__[self.optim_name]
        optim = optim_constructor(
            self.model.parameters(),  **self.optim_args
        )
        optim = ScheduledOptimizer(
            optim,
            self.model,
            self.manager,
            steps_per_epoch=len(self.train_loader),
        )
        
        return optim

    def load_model(self):
        model = ModelRegistry.create(
            self.arch_key,
        )
        pass



