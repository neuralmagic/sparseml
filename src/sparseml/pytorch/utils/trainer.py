import os
from typing import Optional, Union, Dict, Any, Callable, Tuple

from sparseml.pytorch.datasets import DatasetRegistry, ssd_collate_fn, \
    yolo_collate_fn
from sparseml.pytorch.models import ModelRegistry
from sparseml.pytorch.utils import torch_distributed_zero_first, \
    set_deterministic_seeds
from sparsezoo import Zoo
import torch
import logging

from torch import Module
from torch.utils.data import DataLoader, Dataset

_LOGGER = logging.getLogger(__name__)


class ImageClassificationTrainer:
    "Utility class for training image classification models"

    def __init__(
            self,
            checkpoint_path: str,
            recipe_path: str,
            dataset: str,
            dataset_path: str,
            train_batch_size: int,
            test_batch_size: int,
            eval_mode: bool = False,
            local_rank: int = -1,
            loader_num_workers: int = 4,
            loader_pin_memory: bool = True,
            pretrained: Union[str, bool] = True,
            pretrained_dataset: Optional[str] = None,

            dataset_kwargs: Optional[Dict[str, Any]] = None,
            model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self._checkpoint_path = checkpoint_path
        self._checkpoint = self._load_checkpoint()

        self.arch_key = self._load_arch_key_from_checkpoint()
        self.recipe_path = recipe_path

        self.input_shape = ModelRegistry.input_shape(self.arch_key)
        self.input_image_size = self.input_shape[1]  # assume shape [C, S, S]

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.loader_num_workers = loader_num_workers
        self.loader_pin_memory = loader_pin_memory
        self.pretrained = pretrained
        self.pretrained_dataset = pretrained_dataset

        self.dataset = dataset
        self.dataset_path = dataset_path
        self.local_rank = local_rank
        self.eval_mode = eval_mode
        self.dataset_kwargs = dataset_kwargs or {}

        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.env_rank = int(os.environ.get("RANK", -1))

        # non DDP execution or 0th DDP process
        self.is_main_process = self.rank in [-1, 0]
        self._validate_and_fix_train_batch_size()
        self._setup_dataset_preprocessing()
        self._setup_ddp_seeds()

        self.approximate = False

        self.train_dataset, self.train_loader = None, None
        self.val_dataset, self.val_loader = None, None
        self._setup_dataloaders()

        self.model_kwargs = model_kwargs or {}
        self.num_classes = self._infer_num_classes()

        self.model = self.create_and_load_model()

    def _setup_dataloaders(self):
        if not self.eval_mode:
            self.train_dataset, self.train_loader = self._get_dataset_and_dataloader(
                validation=False,
            )
        if self.is_main_process and self.dataset != 'imagefolder':
            self.val_dataset, self.val_loader = self._get_dataset_and_dataloader(
                validation=True,
            )

    def _validate_and_fix_train_batch_size(self):
        # modify training batch size for give world size
        if self.train_batch_size % self.world_size != 0:
            raise ValueError(
                f"Invalid training batch size for world size {self.world_size} "
                f"given batch size {self.train_batch_size}. "
                f"world size must divide training batch size evenly."
            )
        self.train_batch_size = self.train_batch_size // self.world_size

    def _setup_ddp_seeds(self):
        if self.local_rank != -1:
            torch.distributed.init_process_group(backend="nccl",
                                                 init_method="env://")
            set_deterministic_seeds(0)

    def _setup_dataset_preprocessing(self):
        if "preprocessing_type" not in self.dataset_kwargs and (
                "coco" in self.dataset.lower() or "voc" in self.dataset.lower()
        ):
            if self.arch_is_ssd:
                self.dataset_kwargs["preprocessing_type"] = "ssd"
            elif self.arch_is_yolo:
                self.dataset_kwargs["preprocessing_type"] = "yolo"

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

    @property
    def arch_is_ssd(self):
        return 'ssd' in self.arch_key.lower()

    @property
    def arch_is_yolo(self):
        return 'yolo' in self.arch_key.lower()

    def _infer_num_classes(self) -> int:
        if "num_classes" in self.model_kwargs:
            # handle manually overriden num classes
            num_classes = self.model_kwargs["num_classes"]
            del self.model_kwargs["num_classes"]

        elif self.dataset == "imagefolder":
            dataset = self.val_dataset or self.train_dataset  # get non None dataset
            num_classes = dataset.num_classes
        else:
            dataset_attributes = DatasetRegistry.attributes(self.dataset)
            num_classes = dataset_attributes["num_classes"]
        return num_classes

    def _load_model(self) -> Module:
        # pre-condition: self.arch_key is loaded
        pass

    def _get_dataset_and_dataloader(self, validation: bool = False) -> Tuple[
        Dataset, DataLoader]:

        with torch_distributed_zero_first(
                self.local_rank,
        ):  # only download once locally
            _dataset = DatasetRegistry.create(
                self.dataset,
                root=self.dataset_path,
                train=not validation,
                rand_trans=not validation,
                image_size=self.image_size,
                **self.dataset_kwargs,
            )
        sampler = (
            torch.utils.data.distributed.DistributedSampler(_dataset)
            if self.env_rank != -1
            else None
        )
        shuffle = False if sampler is not None else True

        _batch_size = (
            self.test_batch_size
            if validation
            else self.train_batch_size
        )
        dataloader = DataLoader(
            _dataset,
            batch_size=_batch_size,
            shuffle=False if validation else shuffle,
            num_workers=self.loader_num_workers,
            pin_memory=self.loader_pin_memory,
            sampler=sampler,
            collate_fn=self._get_collate_fn(),
        )
        _dataset_type = 'validation' if validation else 'train'
        print(f"created {_dataset_type} dataset and dataloader: {_dataset}")

        return _dataset, dataloader

    def _get_collate_fn(self) -> Optional[Callable]:
        if self.arch_is_ssd:
            return ssd_collate_fn
        if self.arch_is_yolo:
            return yolo_collate_fn
        return None

    def create_and_load_model(self) -> Module:
        """
        Create and load model from checkpoint.

        Note: currently the implementation relies on Module.create
        which reloads the checkpoint
        """
        # only download once locally
        with torch_distributed_zero_first(self.local_rank):
            if self._checkpoint_path == "zoo":
                if self.recipe_path and self.recipe_path.startswith("zoo:"):
                    self._download_checkpoint_from_zoo()
                else:
                    raise ValueError(
                        "'zoo' provided as checkpoint_path but a SparseZoo "
                        "stub prefixed by 'zoo:' not provided as recipe_path"
                    )

            model = ModelRegistry.create(
                self.arch_key,
                self.pretrained,
                self._checkpoint_path,
                self.pretrained_dataset,
                num_classes=self.num_classes,
                **self.model_kwargs,
            )
        print(f"created model: {model}")
        return model

    def _download_checkpoint_from_zoo(self):
        _framework_files = Zoo.download_recipe_base_framework_files(
            self.recipe_path,
            extensions=[".pth"]
        )
        self._checkpoint_path = _framework_files[0]
