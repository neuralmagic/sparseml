import os
from typing import Optional, Union, Dict, Any, Callable, Tuple, List
import warnings

from sparseml.pytorch.datasets import DatasetRegistry, ssd_collate_fn, \
    yolo_collate_fn
from sparseml.pytorch.image_classification.utils import save_recipe, \
    save_model_training
from sparseml.pytorch.models import ModelRegistry
from sparseml.pytorch.optim import ScheduledModifierManager, ScheduledOptimizer
from sparseml.pytorch.sparsification import ConstantPruningModifier
from sparseml.pytorch.utils import (
    torch_distributed_zero_first,
    set_deterministic_seeds,
    LossWrapper,
    SSDLossWrapper,
    YoloLossWrapper,
    TopKAccuracy,
    CrossEntropyLossWrapper,
    InceptionCrossEntropyLossWrapper,
    default_device,
    TensorBoardLogger,
    PythonLogger,
    model_to_device,
    ModuleTrainer,
    ModuleDeviceContext,
    ModuleTester,
    DEFAULT_LOSS_KEY,
    get_prunable_layers,
    tensor_sparsity,
)
from sparseml.utils import create_dirs
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
            init_lr: float = 1e-9,
            optimizer_name: str = 'adam',
            sparse_transfer_learn: bool = False,
            use_mixed_precision: bool = False,
            device: str = default_device(),
            debug_steps: int = -1,
            save_best_after: int = -1,
            save_epochs: List[int] = [],
            optim_kwargs: Optional[Dict[str, Any]] = None,
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
        self.optimizer_name = optimizer_name
        self.init_lr = init_lr
        self.sparse_transfer_learn = sparse_transfer_learn
        self.use_mixed_precision = use_mixed_precision
        self.device = device
        self.debug_steps = debug_steps
        self.save_best_after = save_best_after
        self.save_epochs = save_epochs

        self.dataset_kwargs = dataset_kwargs or {}
        self.optim_kwargs = optim_kwargs or {}

        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.env_rank = int(os.environ.get("RANK", -1))

        # non DDP execution or 0th DDP process
        self.is_main_process = self.rank in [-1, 0]
        self.approximate = False
        self.train_dataset, self.train_loader = None, None
        self.val_dataset, self.val_loader = None, None
        self.val_loss, self.trainings_loss = None, None
        self.epoch, self.optimizer, self.manager = 0, None, None
        self.model_kwargs = model_kwargs or {}
        self.num_classes, self.model = None, None
        self.loggers = None
        self.ddp = False
        self.save_dir, self.loggers = None, None
        self.trainer, self.tester = None, None

        self.__setup()

    def __setup(self):
        self._validate_and_fix_train_batch_size()
        self._setup_dataset_preprocessing()
        self._setup_ddp_seeds()
        self._setup_dataloaders()
        self._setup_loss_wrapper()
        self._setup_loggers_and_save_dirs()
        self.num_classes = self._infer_num_classes()
        self.model = self.create_and_load_model()
        self._setup_epoch_optimizer_and_manager()
        self._setup_device()
        self._setup_training()

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
            torch.distributed.init_process_group(
                backend="nccl",
                init_method="env://",
            )
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
        """
        return: True if the model is an SSD model, False otherwise
        """
        return 'ssd' in self.arch_key.lower()

    @property
    def arch_is_yolo(self):
        """
        return: True if arch_key is yolo, False otherwise
        """
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

    def _setup_loss_wrappers(self):
        self.val_loss = self._setup_loss_wrapper()
        _LOGGER.info(f"created loss for validation: {self.val_loss}")

        self.train_loss = None if self.eval_mode else self._setup_loss_wrapper()
        _LOGGER.info(f"created loss for training: {self.train_loss}")

    def _setup_loss_wrapper(self) -> LossWrapper:
        """
        """
        if self.arch_is_ssd:
            return SSDLossWrapper()
        if self.arch_is_yolo:
            return YoloLossWrapper()

        extras = {"top1acc": TopKAccuracy(1), "top5acc": TopKAccuracy(5)}
        arch_is_inception = "inception" in self.arch_key.lower()
        return (
            CrossEntropyLossWrapper(extras=extras)
            if not arch_is_inception
            else InceptionCrossEntropyLossWrapper(extras=extras)
        )

    def _setup_epoch_optimizer_and_manager(self):
        # # optimizer setup
        optimizer_constructor = torch.optim.__dict__[self.optimizer_name]

        _optim = optimizer_constructor(
            self.model.parameters(), lr=self.init_lr,
            **self.optim_kwargs,
        )
        _LOGGER.info(f"created optimizer: {_optim}")
        _LOGGER.info(
            "note, the lr for the optimizer may not reflect the manager yet "
            "until the recipe config is created and run"
        )

        # restore from previous check point
        if self.checkpoint_path:
            # currently optimizer restoring is unsupported
            # mapping of the restored params to the correct device is not working
            # load_optimizer(args.checkpoint_path, optimizer)
            self.epoch = 0  # load_epoch(args.checkpoint_path) + 1
            print(
                f"restored checkpoint from {self.checkpoint_path} for "
                f"epoch {self.epoch - 1}"
            )
        else:
            self.epoch = 0

        # modifier setup
        additional_modifiers = (
            ConstantPruningModifier.from_sparse_model(model=self.model)
            if self.sparse_transfer_learn
            else None
        )
        self.manager = ScheduledModifierManager.from_yaml(
            file_path=self.recipe_path, add_modifiers=additional_modifiers
        )
        self.optimizer = ScheduledOptimizer(
            optimizer=_optim,
            module=self.model,
            manager=self.manager,
            steps_per_epoch=len(self.train_loader),
            loggers=self.loggers,
        )
        print(f"created manager: {self.manager}")

    def _setup_device(self):
        # device setup
        if not self.env_rank == -1:
            torch.cuda.set_device(self.local_rank)
            self.device = self.local_rank
            self.ddp = True

    def _setup_loggers_and_save_dirs(self):
        if self.is_main_process:
            save_dir = os.path.abspath(os.path.expanduser(self.save_dir))
            logs_dir = (
                os.path.abspath(os.path.expanduser(os.path.join(self.logs_dir)))
            )

            if not self.model_tag:
                dataset_name = (
                    f"{self.dataset}-{self.dataset_kwargs['year']}"
                    if "year" in self.dataset_kwargs
                    else self.dataset
                )
                _model_tag = f"{self.arch_key.replace('/', '.')}_{dataset_name}"
                _model_id = _model_tag
                model_inc = 0
                # set location to check for models with same name
                model_main_dir = logs_dir or save_dir

                while os.path.exists(os.path.join(model_main_dir, _model_id)):
                    model_inc += 1
                    _model_id = f"{_model_tag}__{model_inc:02d}"
            else:
                _model_id = self.model_tag

            save_dir = os.path.join(save_dir, _model_id)
            create_dirs(save_dir)

            # loggers setup
            loggers = [PythonLogger()]

            logs_dir = os.path.join(logs_dir, _model_id)
            create_dirs(logs_dir)

            try:
                loggers.append(TensorBoardLogger(log_path=logs_dir))
            except AttributeError:
                warnings.warn(
                    "Failed to initialize TensorBoard logger, "
                    "it will not be used for logging",
                )

            _LOGGER.info(f"Model id is set to {_model_id}")
        else:
            # do not log for non main processes
            save_dir = None
            loggers = []

        self.save_dir, self.loggers = save_dir, loggers

    def _setup_training(self):
        self.model, self.device, device_ids = model_to_device(
            model=self.model,
            device=self.device,
            ddp=self.ddp,
        )
        _LOGGER.info(f"running on device {self.device} for ids {device_ids}")

        if not self.eval_mode:
            self.trainer = ModuleTrainer(
                module=self.model,
                device=self.device,
                loss=self.train_loss,
                optimizer=self.optimizer,
                loggers=self.loggers,
                device_context=ModuleDeviceContext(
                    use_mixed_precision=self.use_mixed_precision,
                    world_size=self.world_size,
                ),
            )

        if self.is_main_process:  # only test on one DDP process if using DDP
            tester = ModuleTester(
                module=self.model,
                device=self.device,
                loss=self.val_loss,
                loggers=self.loggers,
                log_steps=-1,
            )

            # initial baseline eval run
            tester.run_epoch(
                data_loader=self.val_loader,
                epoch=self.epoch - 1,
                max_steps=self.debug_steps
            )

        if not self.eval_mode:
            save_recipe(
                recipe_manager=self.manager,
                save_dir=self.save_dir,
            )
            _LOGGER.info(f"starting training from epoch {self.epoch}")
            if self.epoch > 0:
                _LOGGER.info("adjusting ScheduledOptimizer to restore point")
                self.optimizer.adjust_current_step(self.epoch, 0)

    def run_training(self):
        if not self.eval_mode:
            target_metric = (
                "top1acc" if "top1acc" in self.tester.loss.available_losses else DEFAULT_LOSS_KEY
            )
            best_metric = None
            val_res = None

            while self.epoch < self.manager.max_epochs:

                if self.debug_steps > 0:
                    # correct since all optimizer steps are not
                    # taken in the epochs for debug mode
                    self.optimizer.adjust_current_step(self.epoch, 0)

                if self.env_rank != -1:  # sync DDP dataloaders
                    self.train_loader.sampler.set_epoch(self.epoch)

                self.trainer.run_epoch(
                    data_loader=self.train_loader,
                    epoch=self.epoch,
                    max_steps=self.debug_steps,
                    show_progress=self.is_main_process,
                )

                # testing steps
                if self.is_main_process:
                    # only test and save on main process
                    val_res = self.tester.run_epoch(
                        data_loader=self.val_loader,
                        epoch=self.epoch,
                        max_steps=self.debug_steps
                    )
                    val_metric = val_res.result_mean(target_metric).item()

                    if self.epoch >= self.save_best_after and (
                            best_metric is None
                            or (
                                    val_metric <= best_metric
                                    if target_metric != "top1acc"
                                    else val_metric >= best_metric
                            )
                    ):
                        save_model_training(
                            self.model,
                            self.optimizer,
                            "checkpoint-best",
                            self.save_dir,
                            self.epoch,
                            self.val_res,
                        )
                        best_metric = val_metric
                # save checkpoints
                _save_epoch = (
                        self.is_main_process
                        and self.save_epochs
                        and self.epoch in self.save_epochs
                )
                if _save_epoch:
                    save_model_training(
                        self.model,
                        self.optimizer,
                        f"checkpoint-{self.epoch:04d}-{val_metric:.04f}",
                        self.save_dir,
                        self.epoch,
                        val_res,
                    )

                self.epoch += 1

                # export the final model
            _LOGGER.info("completed...")
            if self.is_main_process:
                # only convert qat -> quantized ONNX graph for finalized model
                # TODO: change this to all checkpoints when conversion times improve
                save_model_training(
                    self.model, self.optimizer, "model", self.save_dir,
                    self.epoch - 1, val_res
                )

                _LOGGER.info("layer sparsities:")
                for (name, layer) in get_prunable_layers(self.model):
                    _LOGGER.info(
                        f"{name}.weight: {tensor_sparsity(layer.weight).item():.4f}"
                    )

                # close DDP
                if self.env_rank != -1:
                    torch.distributed.destroy_process_group()
