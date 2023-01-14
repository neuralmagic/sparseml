# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/pytorch/vision

import datetime
import logging
import math
import os
import sys
import time
import warnings
from functools import update_wrapper
from types import SimpleNamespace
from typing import Callable, Optional

import torch
import torch.utils.data
import torchvision
from torch import nn
from torch.utils.data.dataloader import DataLoader, default_collate
from torchvision.transforms.functional import InterpolationMode

import click
from sparseml.pytorch.models.registry import ModelRegistry
from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.torchvision import presets, transforms, utils
from sparseml.pytorch.torchvision.sampler import RASampler
from sparseml.pytorch.utils.helpers import (
    download_framework_model_by_recipe_type,
    torch_distributed_zero_first,
)
from sparseml.pytorch.utils.logger import (
    LoggerManager,
    PythonLogger,
    TensorBoardLogger,
    WANDBLogger,
)
from sparseml.pytorch.utils.model import load_model
from sparsezoo import Model


_LOGGER = logging.getLogger(__name__)


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    data_loader_test: DataLoader,
    device: torch.device,
    epoch: int,
    args,
    log_metrics_fn: Callable[[str, utils.MetricLogger, int, int], None],
    manager=None,
    model_ema=None,
    scaler=None,
) -> utils.MetricLogger:
    accum_steps = args.gradient_accum_steps

    model.train()
    metric_logger = utils.MetricLogger(_LOGGER, delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter(
        "imgs_per_sec", utils.SmoothedValue(window_size=10, fmt="{value}")
    )
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=accum_steps))
    metric_logger.add_meter("acc1", utils.SmoothedValue(window_size=accum_steps))
    metric_logger.add_meter("acc5", utils.SmoothedValue(window_size=accum_steps))

    steps_accumulated = 0
    num_optim_steps = 0

    # initial zero grad for gradient accumulation
    optimizer.zero_grad()

    header = f"Epoch: [{epoch}]"
    for (image, target) in metric_logger.log_every(
        data_loader, args.logging_steps * accum_steps, header
    ):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = output = model(image)
            if isinstance(output, tuple):
                # NOTE: sparseml models return two things (logits & probs)
                output = output[0]
            loss = criterion(output, target)

        if steps_accumulated % accum_steps == 0:
            if manager is not None:
                loss = manager.loss_update(
                    loss=loss,
                    module=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    steps_per_epoch=len(data_loader) / accum_steps,
                    student_outputs=outputs,
                    student_inputs=image,
                )

            # first: do training to consume gradients
            if scaler is not None:
                scaler.scale(loss).backward()
                if args.clip_grad_norm is not None:
                    # we should unscale the gradients of optimizer's assigned params
                    # if do gradient clipping
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()

            # zero grad here to start accumulating next set of gradients
            optimizer.zero_grad()
            num_optim_steps += 1
        steps_accumulated += 1

        if model_ema and num_optim_steps % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["imgs_per_sec"].update(
            batch_size / (time.time() - start_time)
        )

        if args.eval_steps is not None and num_optim_steps % args.eval_steps == 0:
            eval_metrics = evaluate(model, criterion, data_loader_test, device)
            model.train()
            log_metrics_fn("Test", eval_metrics, epoch, num_optim_steps)

        if num_optim_steps % args.logging_steps == 0:
            log_metrics_fn("Train", metric_logger, epoch, num_optim_steps)

    return metric_logger


def evaluate(
    model,
    criterion,
    data_loader,
    device,
    print_freq=100,
    log_suffix="",
) -> utils.MetricLogger:
    model.eval()
    metric_logger = utils.MetricLogger(_LOGGER, delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            if isinstance(output, tuple):
                output = output[0]
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, "
            f"but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    _LOGGER.info(
        header
        + f"Acc@1 {metric_logger.acc1.global_avg:.3f}"
        + f"Acc@5 {metric_logger.acc5.global_avg:.3f}"
    )
    return metric_logger


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join(
        "~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt"
    )
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, args):
    # Data loading code
    _LOGGER.info("Loading data")
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    _LOGGER.info("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        _LOGGER.info(f"Loading dataset_train from {cache_path}")
        dataset, _ = torch.load(cache_path)
    else:
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        ra_magnitude = args.ra_magnitude
        augmix_severity = args.augmix_severity
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            presets.ClassificationPresetTrain(
                crop_size=train_crop_size,
                interpolation=interpolation,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob,
                ra_magnitude=ra_magnitude,
                augmix_severity=augmix_severity,
            ),
        )
        if args.cache_dataset:
            _LOGGER.info(f"Saving dataset_train to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    _LOGGER.info(f"Took {time.time() - st}")

    _LOGGER.info("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        _LOGGER.info(f"Loading dataset_test from {cache_path}")
        dataset_test, _ = torch.load(cache_path)
    else:
        preprocessing = presets.ClassificationPresetEval(
            crop_size=val_crop_size,
            resize_size=val_resize_size,
            interpolation=interpolation,
        )

        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            preprocessing,
        )
        if args.cache_dataset:
            _LOGGER.info(f"Saving dataset_test to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    _LOGGER.info("Creating data loaders")
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_test, shuffle=False
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def main(args):
    if args.resume is not None and args.checkpoint_path is not None:
        raise ValueError(
            "Only one of --resume or --checkpoint-path can be specified, not both."
        )

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    if not utils.is_main_process():
        _LOGGER.disabled = True

    _LOGGER.info(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.dataset_path, "train")
    val_dir = os.path.join(args.dataset_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(
        train_dir, val_dir, args
    )

    collate_fn = None
    num_classes = len(dataset.classes)
    mixup_transforms = []
    if args.mixup_alpha > 0.0:
        mixup_transforms.append(
            transforms.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha)
        )
    if args.cutmix_alpha > 0.0:
        mixup_transforms.append(
            transforms.RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha)
        )
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )

    _LOGGER.info("Creating model")
    local_rank = args.rank if args.distributed else None
    model = _create_model(
        arch_key=args.arch_key,
        local_rank=local_rank,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint_path,
        pretrained_dataset=args.pretrained_dataset,
        device=device,
        num_classes=num_classes,
    )

    if args.distill_teacher not in ["self", "disable", None]:
        _LOGGER.info("Instantiating teacher")
        args.distill_teacher = _create_model(
            arch_key=args.teacher_arch_key,
            local_rank=local_rank,
            pretrained=True,  # teacher is always pretrained
            pretrained_dataset=args.pretrained_teacher_dataset,
            checkpoint_path=args.distill_teacher,
            device=device,
            num_classes=num_classes,
        )

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in [
            "class_token",
            "position_embedding",
            "relative_position_bias_table",
        ]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay
        if len(custom_keys_weight_decay) > 0
        else None,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            eps=0.0316,
            alpha=0.9,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            parameters, lr=args.lr, weight_decay=args.weight_decay
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(
            parameters, lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        raise RuntimeError(
            f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported."
        )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent from
        # other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates =
        #   (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and ommit it.
        # Thus:
        # adjust = 1 / total_ema_updates ~=
        #   n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(
            model, device=device, decay=1.0 - alpha
        )

    manager = checkpoint_manager = None
    if args.checkpoint_path:
        checkpoint = _load_checkpoint(args.checkpoint_path)

        # restore state from prior recipe
        manager = (
            ScheduledModifierManager.from_yaml(
                args.recipe, recipe_variables=args.recipe_args
            )
            if args.recipe is not None
            else None
        )
        checkpoint_manager = (
            ScheduledModifierManager.from_yaml(checkpoint["recipe"])
            if "recipe" in checkpoint and checkpoint["recipe"] is not None
            else None
        )
    elif args.resume:
        checkpoint = _load_checkpoint(args.resume)

        if "recipe" in checkpoint:
            # NOTE: override manager with the checkpoint's manager
            manager = ScheduledModifierManager.from_yaml(checkpoint["recipe"])
            checkpoint_manager = None
        else:
            raise ValueError("Flag --resume is set but checkpoint does not have recipe")

        # NOTE: override start epoch
        args.start_epoch = checkpoint["epoch"] + 1
    else:
        checkpoint = None
        manager = (
            ScheduledModifierManager.from_yaml(
                args.recipe, recipe_variables=args.recipe_args
            )
            if args.recipe is not None
            else None
        )
        checkpoint_manager = None

    # load params
    if checkpoint is not None:
        if "optimizer" in checkpoint and not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if model_ema and "model_ema" in checkpoint:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can
        # noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(
                model_ema,
                criterion,
                data_loader_test,
                device,
                print_freq=args.logging_steps,
                log_suffix="EMA",
            )
        else:
            evaluate(
                model,
                criterion,
                data_loader_test,
                device,
                print_freq=args.logging_steps,
            )
        return

    if utils.is_main_process():
        loggers = [
            PythonLogger(logger=_LOGGER),
            TensorBoardLogger(log_path=args.output_dir),
        ]
        try:
            config = vars(args)
            if manager is not None:
                config["manager"] = str(manager)
            loggers.append(WANDBLogger(init_kwargs=dict(config=config)))
        except ImportError:
            warnings.warn("Unable to import wandb for logging")
        logger = LoggerManager(loggers)
    else:
        logger = LoggerManager(log_python=False)

    if args.recipe is not None:
        logger.save(args.recipe)

    steps_per_epoch = len(data_loader) / args.gradient_accum_steps

    def log_metrics(tag: str, metrics: utils.MetricLogger, epoch: int, epoch_step: int):
        step = int(epoch * steps_per_epoch + epoch_step)
        for metric_name, smoothed_value in metrics.meters.items():
            logger.log_scalar(
                f"{tag}/{metric_name}", smoothed_value.global_avg, step=step
            )

    if manager is not None:
        manager.initialize(
            model,
            epoch=args.start_epoch,
            loggers=logger,
            distillation_teacher=args.distill_teacher,
        )
        step_wrapper = manager.modify(
            model,
            optimizer,
            steps_per_epoch=steps_per_epoch,
            epoch=args.start_epoch,
            wrap_optim=scaler,
        )
        if scaler is None:
            optimizer = step_wrapper
        else:
            scaler = step_wrapper

    lr_scheduler = _get_lr_scheduler(
        args, optimizer, checkpoint=checkpoint, manager=manager
    )

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    best_top1_acc = -math.inf

    _LOGGER.info("Start training")

    start_time = time.time()
    max_epochs = manager.max_epochs if manager is not None else args.epochs
    for epoch in range(args.start_epoch, max_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if manager is not None and manager.qat_active(epoch=epoch):
            if scaler is not None:
                scaler._enabled = False
            model_ema = None

        train_metrics = train_one_epoch(
            model,
            criterion,
            optimizer,
            data_loader,
            data_loader_test,
            device,
            epoch,
            args,
            log_metrics,
            manager=manager,
            model_ema=model_ema,
            scaler=scaler,
        )
        log_metrics("Train", train_metrics, epoch, steps_per_epoch)

        if lr_scheduler:
            lr_scheduler.step()

        eval_metrics = evaluate(model, criterion, data_loader_test, device)
        log_metrics("Test", eval_metrics, epoch, steps_per_epoch)

        top1_acc = eval_metrics.acc1.global_avg
        if model_ema:
            ema_eval_metrics = evaluate(
                model_ema,
                criterion,
                data_loader_test,
                device,
                log_suffix="EMA",
            )
            log_metrics("Test/EMA", ema_eval_metrics, epoch, steps_per_epoch)

        is_new_best = epoch >= args.save_best_after and top1_acc > best_top1_acc
        if is_new_best:
            best_top1_acc = top1_acc
        if args.output_dir:
            checkpoint = {
                "state_dict": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": args,
            }
            if lr_scheduler:
                checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()

            if checkpoint_manager is not None:
                checkpoint["epoch"] = (
                    -1
                    if epoch == max_epochs - 1
                    else epoch + checkpoint_manager.max_epochs
                )
                checkpoint["recipe"] = str(
                    ScheduledModifierManager.compose_staged(checkpoint_manager, manager)
                )
            else:
                checkpoint["epoch"] = -1 if epoch == max_epochs - 1 else epoch
                if str(manager) is not None:
                    checkpoint["recipe"] = str(manager)

            file_names = ["checkpoint.pth"]
            if is_new_best:
                file_names.append("checkpoint-best.pth")
            _save_checkpoints(
                epoch,
                args.output_dir,
                file_names,
                checkpoint,
                train_metrics,
                eval_metrics,
            )

    if manager is not None:
        manager.finalize()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    _LOGGER.info(f"Training time {total_time_str}")


def _create_model(
    arch_key: Optional[str] = None,
    local_rank=None,
    pretrained: Optional[bool] = False,
    checkpoint_path: Optional[str] = None,
    pretrained_dataset: Optional[str] = None,
    device=None,
    num_classes=None,
):
    if arch_key in ModelRegistry.available_keys():
        with torch_distributed_zero_first(local_rank):
            model = ModelRegistry.create(
                key=arch_key,
                pretrained=pretrained,
                pretrained_path=checkpoint_path,
                pretrained_dataset=pretrained_dataset,
                num_classes=num_classes,
            )
    elif arch_key in torchvision.models.__dict__:
        # fall back to torchvision
        model = torchvision.models.__dict__[arch_key](
            pretrained=pretrained, num_classes=num_classes
        )
        if checkpoint_path is not None:
            load_model(checkpoint_path, model, strict=True)
    else:
        raise ValueError(
            f"Unable to find {arch_key} in ModelRegistry or in torchvision.models"
        )
    model.to(device)
    return model


def _get_lr_scheduler(args, optimizer, checkpoint=None, manager=None):
    lr_scheduler = None

    if manager is not None and manager.learning_rate_modifiers:
        lr_scheduler = None
    else:
        args.lr_scheduler = args.lr_scheduler.lower()
        if args.lr_scheduler == "steplr":
            main_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
            )
        elif args.lr_scheduler == "cosineannealinglr":
            main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs - args.lr_warmup_epochs,
                eta_min=args.lr_min,
            )
        elif args.lr_scheduler == "exponentiallr":
            main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=args.lr_gamma
            )
        else:
            raise RuntimeError(
                f"Invalid lr scheduler '{args.lr_scheduler}'. "
                "Only StepLR, CosineAnnealingLR and ExponentialLR "
                "are supported."
            )

        if args.lr_warmup_epochs > 0:
            if args.lr_warmup_method == "linear":
                warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=args.lr_warmup_decay,
                    total_iters=args.lr_warmup_epochs,
                )
            elif args.lr_warmup_method == "constant":
                warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                    optimizer,
                    factor=args.lr_warmup_decay,
                    total_iters=args.lr_warmup_epochs,
                )
            else:
                raise RuntimeError(
                    f"Invalid warmup lr method '{args.lr_warmup_method}'. "
                    "Only linear and constant are supported."
                )
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_lr_scheduler, main_lr_scheduler],
                milestones=[args.lr_warmup_epochs],
            )
        else:
            lr_scheduler = main_lr_scheduler

        if args.resume and checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    return lr_scheduler


def _load_checkpoint(path):
    if path.startswith("zoo:"):
        path = download_framework_model_by_recipe_type(Model(path))
    return torch.load(path, map_location="cpu")


def _save_checkpoints(
    epoch, output_dir, file_names, checkpoint, train_metrics, eval_metrics
):
    metrics = "\n".join(
        [
            f"epoch: {epoch}",
            f"__loss__: {train_metrics.loss.global_avg}",
            f"top1acc: {eval_metrics.acc1.global_avg}",
            f"top5acc: {eval_metrics.acc5.global_avg}",
        ]
    )
    for fname in file_names:
        utils.save_on_master(checkpoint, os.path.join(output_dir, fname))
        if utils.is_main_process():
            with open(
                os.path.join(output_dir, fname.replace(".pth", ".txt")), "w"
            ) as fp:
                fp.write(metrics)


_ARGUMENTS_ERROR = (
    "Deprecated arguments found: {}. "
    "Please see --help for new arguments.\n"
    "The old script can be accessed with "
    "`sparseml.pytorch.image_classification.train`"
)


def _deprecate_old_arguments(f):
    def new_func(*args, **kwargs):
        if "--recipe-path" in sys.argv:
            raise ValueError(_ARGUMENTS_ERROR.format("--recipe-path"))
        return f(*args, **kwargs)

    return update_wrapper(new_func, f)


@_deprecate_old_arguments
@click.command(
    context_settings=dict(
        token_normalize_func=lambda x: x.replace("-", "_"),
        show_default=True,
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.option("--recipe", default=None, type=str, help="Path to recipe")
@click.option(
    "--recipe-args",
    default=None,
    type=str,
    help="json parsable dict of recipe variable names to values to overwrite with",
)
@click.option("--dataset-path", required=True, type=str, help="dataset path")
@click.option(
    "--arch-key",
    default=None,
    type=str,
    help=(
        "The architecture key for image classification model; "
        "example: `resnet50`, `mobilenet`. "
        "Note: Will be read from the checkpoint if not specified"
    ),
)
@click.option(
    "--pretrained",
    default="True",
    type=str,
    help=(
        "The type of pretrained weights to use, "
        "loads default pretrained weights for "
        "the model if not specified or set to `True`. "
        "Otherwise, should be set to the desired weights "
        "type: [base, optim, optim-perf]. To not load any weights set"
        " to one of [none, false]"
    ),
)
@click.option(
    "--pretrained-dataset",
    default=None,
    type=str,
    help=(
        "The dataset to load pretrained weights for if pretrained is "
        "set. Load the default dataset for the architecture if set to None. "
        "examples:`imagenet`, `cifar10`, etc..."
    ),
)
@click.option(
    "--device",
    default="cuda",
    type=str,
    help="device (Use cuda or cpu)",
)
@click.option(
    "-b",
    "--batch-size",
    default=32,
    type=int,
    help="images per gpu, the total batch size is $NGPU x batch_size",
)
@click.option(
    "--epochs",
    default=10,
    type=int,
    metavar="N",
    help="number of total epochs to run",
)
@click.option(
    "-j",
    "--workers",
    default=16,
    type=int,
    metavar="N",
    help="number of data loading workers",
)
@click.option("--opt", default="sgd", type=str, help="optimizer")
@click.option("--lr", default=0.1, type=float, help="initial learning rate")
@click.option("--momentum", default=0.9, type=float, metavar="M", help="momentum")
@click.option(
    "-w", "--weight-decay", default=1e-4, type=float, metavar="W", help="weight decay"
)
@click.option(
    "--norm-weight-decay",
    default=None,
    type=float,
    help="weight decay for Normalization layers",
)
@click.option(
    "--bias-weight-decay",
    default=None,
    type=float,
    help="weight decay for bias parameters of all layers",
)
@click.option(
    "--transformer-embedding-decay",
    default=None,
    type=float,
    help="weight decay for embedding parameters for vision transformer models.",
)
@click.option("--label-smoothing", default=0.0, type=float, help="label smoothing")
@click.option("--mixup-alpha", default=0.0, type=float, help="mixup alpha")
@click.option("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha")
@click.option("--lr-scheduler", default="steplr", type=str, help="the lr scheduler")
@click.option(
    "--lr-warmup-epochs",
    default=0,
    type=int,
    help="the number of epochs to warmup",
)
@click.option(
    "--lr-warmup-method",
    default="constant",
    type=str,
    help="the warmup method",
)
@click.option("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
@click.option(
    "--lr-step-size",
    default=30,
    type=int,
    help="decrease lr every step-size epochs",
)
@click.option(
    "--lr-gamma",
    default=0.1,
    type=float,
    help="decrease lr by a factor of lr-gamma",
)
@click.option(
    "--lr-min",
    default=0.0,
    type=float,
    help="minimum lr of lr schedule",
)
@click.option("--print-freq", default=None, type=int, help="DEPRECATED. Does nothing.")
@click.option(
    "--logging-steps",
    default=100,
    type=int,
    help="Frequency in number of batch updates for logging/printing",
)
@click.option(
    "--eval-steps",
    default=None,
    type=int,
    help="Number of steps to evaluate during training",
)
@click.option("--output-dir", default=".", type=str, help="path to save outputs")
@click.option("--resume", default=None, type=str, help="path of checkpoint")
@click.option(
    "--checkpoint-path",
    default=None,
    type=str,
    help=(
        "A path to a previous checkpoint to load the state from "
        "and resume the state for. If provided, pretrained will "
        "be ignored. If using a SparseZoo recipe, can also "
        "provide 'zoo' to load the base weights associated with "
        "that recipe. Additionally, can also provide a SparseZoo model stub "
        "to load model weights from SparseZoo"
    ),
)
@click.option("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
@click.option(
    "--cache-dataset",
    is_flag=True,
    default=False,
    help="Cache the datasets for quicker initialization. "
    "It also serializes the transforms",
)
@click.option("--sync-bn", is_flag=True, default=False, help="Use sync batch norm")
@click.option("--test-only", is_flag=True, default=False, help="Only test the model")
@click.option("--auto-augment", default=None, type=str, help="auto augment policy")
@click.option(
    "--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy"
)
@click.option(
    "--augmix-severity", default=3, type=int, help="severity of augmix policy"
)
@click.option(
    "--random-erase", default=0.0, type=float, help="random erasing probability"
)
@click.option(
    "--amp",
    is_flag=True,
    default=False,
    help="Use torch.cuda.amp for mixed precision training",
)
@click.option(
    "--world-size", default=1, type=int, help="number of distributed processes"
)
@click.option(
    "--dist-url",
    default="env://",
    type=str,
    help="url used to set up distributed training",
)
@click.option(
    "--model-ema",
    is_flag=True,
    default=False,
    help="enable tracking Exponential Moving Average of model parameters",
)
@click.option(
    "--model-ema-steps",
    type=int,
    default=32,
    help="the number of iterations that controls how often to update the EMA model",
)
@click.option(
    "--model-ema-decay",
    type=float,
    default=0.99998,
    help="decay factor for Exponential Moving Average of model parameters",
)
@click.option(
    "--use-deterministic-algorithms",
    is_flag=True,
    default=False,
    help="Forces the use of deterministic algorithms only.",
)
@click.option(
    "--interpolation",
    default="bilinear",
    type=str,
    help="the interpolation method",
)
@click.option(
    "--val-resize-size",
    default=256,
    type=int,
    help="the resize size used for validation",
)
@click.option(
    "--val-crop-size",
    default=224,
    type=int,
    help="the central crop size used for validation",
)
@click.option(
    "--train-crop-size",
    default=224,
    type=int,
    help="the random crop size used for training",
)
@click.option(
    "--clip-grad-norm",
    default=None,
    type=float,
    help="the maximum gradient norm",
)
@click.option(
    "--ra-sampler",
    is_flag=True,
    default=False,
    help="whether to use Repeated Augmentation in training",
)
@click.option(
    "--ra-reps",
    default=3,
    type=int,
    help="number of repetitions for Repeated Augmentation",
)
@click.option(
    "--gradient-accum-steps",
    default=1,
    type=int,
    help="gradient accumulation steps",
)
@click.option(
    "--save-best-after",
    default=1,
    type=int,
    help="Save the best validation result after the given "
    "epoch completes until the end of training",
)
@click.option(
    "--distill-teacher",
    default=None,
    type=str,
    help="Teacher model for distillation (a trained image classification model)"
    " can be set to 'self' for self-distillation and 'disable' to switch-off"
    " distillation, additionally can also take in a SparseZoo stub",
)
@click.option(
    "--pretrained-teacher-dataset",
    default=None,
    type=str,
    help=(
        "The dataset to load pretrained weights for the teacher"
        "Load the default dataset for the architecture if set to None. "
        "examples:`imagenet`, `cifar10`, etc..."
    ),
)
@click.option(
    "--teacher-arch-key",
    default=None,
    type=str,
    help=(
        "The architecture key for teacher image classification model; "
        "example: `resnet50`, `mobilenet`. "
        "Note: Will be read from the checkpoint if not specified"
    ),
)
@click.pass_context
def cli(ctx, **kwargs):
    """
    PyTorch classification training
    """
    if len(ctx.args) > 0:
        raise ValueError(_ARGUMENTS_ERROR.format(ctx.args))
    main(SimpleNamespace(**kwargs))


if __name__ == "__main__":
    cli()
