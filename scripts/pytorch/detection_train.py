"""
Object detection recal script. Setup to support the following use cases:
- training SSD object detection models
- pruning SSD object detection models
- transfer learning SSD object detection models
- evaluating SSD object detection models


##########
Command help:
usage: detection_train.py [-h] [--recal-config-path RECAL_CONFIG_PATH]
                          [--sparse-transfer-learn] [--eval-mode] --arch-key
                          ARCH_KEY [--pretrained PRETRAINED]
                          [--pretrained-dataset PRETRAINED_DATASET]
                          [--checkpoint-path CHECKPOINT_PATH]
                          [--device DEVICE] --dataset DATASET --dataset-path
                          DATASET_PATH [--dataset-year DATASET_YEAR]
                          --train-batch-size TRAIN_BATCH_SIZE
                          --test-batch-size TEST_BATCH_SIZE
                          [--loader-num-workers LOADER_NUM_WORKERS]
                          [--loader-pin-memory LOADER_PIN_MEMORY]
                          [--optim OPTIM] [--init-lr INIT_LR]
                          [--optim-args OPTIM_ARGS] [--model-tag MODEL_TAG]
                          [--save-dir SAVE_DIR] [--logs-dir LOGS_DIR]
                          [--save-best-after SAVE_BEST_AFTER]
                          [--save-epochs SAVE_EPOCHS [SAVE_EPOCHS ...]]
                          [--map-calculation-frequency MAP_CALCULATION_FREQUENCY]
                          [--use-mixed-precision]
                          [--debug-steps DEBUG_STEPS]

Train and/or prune an image detection architecture on a dataset

optional arguments:
  -h, --help            show this help message and exit
  --recal-config-path RECAL_CONFIG_PATH
                        The path to the yaml file containing the modifiers and
                        schedule to apply them with. If set to
                        'transfer_learning', then will create a schedule to
                        enable sparse transfer learning
  --sparse-transfer-learn
                        Enable sparse transfer learning modifiers to enforce
                        the sparsity for already sparse layers. The modifiers
                        are added to the ones to be loaded from the recal-
                        config-path
  --eval-mode           Puts into evaluation mode so that the model can be
                        evaluated on the desired dataset
  --arch-key ARCH_KEY   The type of model to create, ex: resnet50, vgg16,
                        mobilenet put as help to see the full list (will raise
                        an exception with the list)
  --pretrained PRETRAINED
                        The type of pretrained weights to use, default is true
                        to load the default pretrained weights for the model.
                        Otherwise should be set to the desired weights type:
                        [base, recal, recal-perf]. To not load any weights set
                        to one of [none, false]
  --pretrained-dataset PRETRAINED_DATASET
                        The dataset to load pretrained weights for if
                        pretrained is set. Default is None which will load the
                        default dataset for the architecture. Ex can be set to
                        imagenet, cifar10, etc
  --checkpoint-path CHECKPOINT_PATH
                        A path to a previous checkpoint to load the state from
                        and resume the state for. If provided, pretrained will
                        be ignored
  --device DEVICE       The device to run on (can also include ids for data
                        parallel), ex: cpu, cuda, cuda:0,1
  --dataset DATASET     The dataset to use for training, ex: coco, voc-
                        detection, etc. Set to imagefolder for a generic image
                        detection dataset setup
  --dataset-path DATASET_PATH
                        The root path to where the dataset is stored
  --dataset-year DATASET_YEAR
                        the year of the dataset to use. defaults to the
                        default year
  --train-batch-size TRAIN_BATCH_SIZE
                        The batch size to use while training
  --test-batch-size TEST_BATCH_SIZE
                        The batch size to use while testing
  --loader-num-workers LOADER_NUM_WORKERS
                        The number of workers to use for data loading
  --loader-pin-memory LOADER_PIN_MEMORY
                        Use pinned memory for data loading
  --optim OPTIM         The optimizer type to use, one of [SGD, Adam]
  --init-lr INIT_LR     The initial learning rate to use while training, set
                        to a low value because LR should come from the recal
                        config path
  --optim-args OPTIM_ARGS
                        Additional args to be passed to the optimizer passed
                        in as a json dict
  --model-tag MODEL_TAG
                        A tag to use for the model for saving results under
                        save-dir and in tensorboard, defaults to the model
                        arch and dataset used
  --save-dir SAVE_DIR   The path to the directory for saving results
  --logs-dir LOGS_DIR   The path to the directory for saving logs
  --save-best-after SAVE_BEST_AFTER
                        start saving the best validation result after the
                        given epoch completes until the end of training
  --save-epochs SAVE_EPOCHS [SAVE_EPOCHS ...]
                        epochs to save checkpoints at
  --map-calculation-frequency MAP_CALCULATION_FREQUENCY
                        number of epochs between validation mAP calculation.
                        default calculates mAP every 10 epochs
  --use-mixed-precision
                        Trains model using mixed precision. Supported
                        environments are single GPU and multiple GPUs using
                        DistributedDataParallel with one GPU per process
  --debug-steps DEBUG_STEPS
                        Amount of steps to run for training and testing for a
                        debug mode


##########
Example command for training COCO on VOC dataset:
python scripts/pytorch/detection_train.py \
  --recal-config-path scripts/pytorch/configs/ssd_training.yaml \
  --arch-key ssd300_resnet50 \
  --dataset voc_detection \
  --dataset-path ~/datasets/voc-detection-2007/ \
  --dataset-year 2007 \
  --train-batch-size 32 \
  --test-batch-size 32 \
  --optim-args '{"momentum": 0.9, "weight_decay": 0.0005}'
"""

import argparse
from typing import Tuple, Union
import logging
import os
import time
import json

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam

try:
    from torch.optim import RMSprop
except Exception:
    RMSprop = None
    logging.warning("RMSprop not available as an optimizer")

from torch.optim.optimizer import Optimizer

from neuralmagicML import get_main_logger
from neuralmagicML.pytorch.datasets import (
    DatasetRegistry,
    ssd_collate_fn,
    yolo_collate_fn,
)
from neuralmagicML.pytorch.models import ModelRegistry
from neuralmagicML.pytorch.utils import (
    SSDLossWrapper,
    YoloLossWrapper,
    MeanAveragePrecision,
    ModuleTrainer,
    ModuleTester,
    ModuleRunResults,
    TensorBoardLogger,
    PythonLogger,
    model_to_device,
    load_optimizer,
    load_epoch,
    load_model,
    default_device,
    ModuleExporter,
    get_prunable_layers,
    tensor_sparsity,
    DEFAULT_LOSS_KEY,
)
from neuralmagicML.pytorch.recal import (
    ScheduledModifierManager,
    ScheduledOptimizer,
    ConstantKSModifier,
)
from neuralmagicML.utils import create_dirs


LOGGER = get_main_logger()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and/or prune an image detection " "architecture on a dataset"
    )

    # recal
    parser.add_argument(
        "--recal-config-path",
        type=str,
        default=None,
        help="The path to the yaml file containing the modifiers and "
        "schedule to apply them with. If set to 'transfer_learning', "
        "then will create a schedule to enable sparse transfer learning",
    )
    parser.add_argument(
        "--sparse-transfer-learn",
        action="store_true",
        help="Enable sparse transfer learning modifiers to enforce the sparsity "
        "for already sparse layers. The modifiers are added to the "
        "ones to be loaded from the recal-config-path",
    )
    parser.add_argument(
        "--eval-mode",
        action="store_true",
        help="Puts into evaluation mode so that the model can be "
        "evaluated on the desired dataset",
    )

    # model args
    parser.add_argument(
        "--arch-key",
        type=str,
        required=True,
        help="The type of model to create, ex: resnet50, vgg16, mobilenet "
        "put as help to see the full list (will raise an exception with the list)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=True,
        help="The type of pretrained weights to use, "
        "default is true to load the default pretrained weights for the model. "
        "Otherwise should be set to the desired weights type: "
        "[base, recal, recal-perf]. "
        "To not load any weights set to one of [none, false]",
    )
    parser.add_argument(
        "--pretrained-dataset",
        type=str,
        default=None,
        help="The dataset to load pretrained weights for if pretrained is set. "
        "Default is None which will load the default dataset for the architecture."
        " Ex can be set to imagenet, cifar10, etc",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="A path to a previous checkpoint to load the state from and "
        "resume the state for. If provided, pretrained will be ignored",
    )

    # training and dataset args
    parser.add_argument(
        "--device",
        type=str,
        default=default_device(),
        help="The device to run on (can also include ids for data parallel), ex: "
        "cpu, cuda, cuda:0,1",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The dataset to use for training, "
        "ex: coco, voc-detection, etc. "
        "Set to imagefolder for a generic image detection dataset setup",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="The root path to where the dataset is stored",
    )
    parser.add_argument(
        "--dataset-year",
        type=str,
        default=None,
        help="the year of the dataset to use. defaults to the default year",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        required=True,
        help="The batch size to use while training",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        required=True,
        help="The batch size to use while testing",
    )
    parser.add_argument(
        "--loader-num-workers",
        type=int,
        default=4,
        help="The number of workers to use for data loading",
    )
    parser.add_argument(
        "--loader-pin-memory",
        type=bool,
        default=True,
        help="Use pinned memory for data loading",
    )

    # optim args
    parser.add_argument(
        "--optim",
        type=str,
        default="SGD",
        help="The optimizer type to use, one of [SGD, Adam]",
    )
    parser.add_argument(
        "--init-lr",
        type=float,
        default=1e-9,
        help="The initial learning rate to use while training, "
        "set to a low value because LR should come from the recal config path",
    )
    parser.add_argument(
        "--optim-args",
        type=json.loads,
        default={"momentum": 0.9, "nesterov": True, "weight_decay": 0.0001},
        help="Additional args to be passed to the optimizer passed in as a json dict",
    )

    # logging and saving
    parser.add_argument(
        "--model-tag",
        type=str,
        default=None,
        help="A tag to use for the model for saving results under save-dir "
        "and in tensorboard, defaults to the model arch and dataset used",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="pytorch_detection_train",
        help="The path to the directory for saving results",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default=os.path.join("pytorch_detection_train", "tensorboard-logs"),
        help="The path to the directory for saving logs",
    )
    parser.add_argument(
        "--save-best-after",
        type=int,
        default=-1,
        help="start saving the best validation result after the given "
        "epoch completes until the end of training",
    )
    parser.add_argument(
        "--save-epochs",
        type=int,
        default=[],
        nargs="+",
        help="epochs to save checkpoints at",
    )
    parser.add_argument(
        "--map-calculation-frequency",
        type=int,
        default=10,
        help="number of epochs between validation mAP calculation. "
        "default calculates mAP every 10 epochs",
    )
    parser.add_argument(
        "--use-mixed-precision",
        action="store_true",
        help="Trains model using mixed precision. Supported environments are single GPU"
        " and multiple GPUs using DistributedDataParallel with one GPU per process",
    )

    # debug options
    parser.add_argument(
        "--debug-steps",
        type=int,
        default=-1,
        help="Amount of steps to run for training and testing for a debug mode",
    )

    args = parser.parse_args()

    if str(args.pretrained).lower() == "true":
        args.pretrained = True
    elif str(args.pretrained).lower() in ["false", "none"]:
        args.pretrained = False

    return args


def _save_model(
    model: Module,
    optim: Optimizer,
    input_shape: Tuple[int, ...],
    save_name: str,
    save_dir: str,
    epoch: int,
    val_res: Union[ModuleRunResults, None],
):
    LOGGER.info(
        "Saving model for epoch {} and val_loss {} to {} for {}".format(
            epoch, val_res.result_mean(DEFAULT_LOSS_KEY).item(), save_dir, save_name
        )
    )
    exporter = ModuleExporter(model, save_dir)
    exporter.export_pytorch(optim, epoch, "{}.pth".format(save_name))
    exporter.export_onnx(torch.randn(1, *input_shape), "{}.onnx".format(save_name))

    info_path = os.path.join(save_dir, "{}.txt".format(save_name))

    with open(info_path, "w") as info_file:
        info_lines = [
            "epoch: {}".format(epoch),
        ]

        if val_res is not None:
            for loss in val_res.results.keys():
                info_lines.append(
                    "{}: {}".format(loss, val_res.result_mean(loss).item())
                )

        info_file.write("\n".join(info_lines))


def main(args):
    # logging and saving setup
    save_dir = os.path.abspath(os.path.expanduser(args.save_dir))
    logs_dir = os.path.abspath(os.path.expanduser(os.path.join(args.logs_dir)))

    if not args.model_tag:
        dataset_name = (
            args.dataset
            if args.dataset_year is None
            else "{}-{}".format(args.dataset, args.dataset_year)
        )
        model_tag = "{}_{}".format(args.arch_key.replace("/", "."), dataset_name)
        model_id = model_tag
        model_inc = 0

        while os.path.exists(os.path.join(logs_dir, model_id)):
            model_inc += 1
            model_id = "{}__{:02d}".format(model_tag, model_inc)
    else:
        model_id = args.model_tag

    save_dir = os.path.join(save_dir, model_id)
    logs_dir = os.path.join(logs_dir, model_id)
    create_dirs(save_dir)
    create_dirs(logs_dir)

    # loggers setup
    loggers = [TensorBoardLogger(log_path=logs_dir), PythonLogger()]
    LOGGER.info("Model id is set to {}".format(model_id))

    # dataset creation
    input_shape = ModelRegistry.input_shape(args.arch_key)
    image_size = input_shape[1]  # assume shape [C, S, S] where S is the image size

    dataset_kwargs = {} if args.dataset_year is None else {"year": args.dataset_year}
    if "coco" in args.dataset.lower() or "voc" in args.dataset.lower():
        if "ssd" in args.arch_key.lower():
            dataset_kwargs["preprocessing_type"] = "ssd"
        elif "yolo" in args.arch_key.lower():
            dataset_kwargs["preprocessing_type"] = "yolo"

    collate_fn = ssd_collate_fn if "ssd" in args.arch_key.lower() else yolo_collate_fn

    if not args.eval_mode:
        train_dataset = DatasetRegistry.create(
            args.dataset,
            root=args.dataset_path,
            train=True,
            rand_trans=True,
            image_size=image_size,
            **dataset_kwargs,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            num_workers=args.loader_num_workers,
            pin_memory=args.loader_pin_memory,
        )
        LOGGER.info("created train_dataset: {}".format(train_dataset))
    else:
        train_dataset = None
        train_loader = None

    val_dataset = DatasetRegistry.create(
        args.dataset,
        root=args.dataset_path,
        train=False,
        rand_trans=False,
        image_size=image_size,
        **dataset_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.test_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=args.loader_num_workers,
        pin_memory=args.loader_pin_memory,
    )
    LOGGER.info("created val_dataset: {}".format(val_dataset))

    # model creation
    if args.dataset == "imagefolder":
        num_classes = val_dataset.num_classes
    else:
        dataset_attributes = DatasetRegistry.attributes(args.dataset)
        num_classes = dataset_attributes["num_classes"]

    model = ModelRegistry.create(
        args.arch_key,
        args.pretrained,
        args.checkpoint_path,
        args.pretrained_dataset,
        num_classes=num_classes,
    )

    if args.checkpoint_path:
        load_model(args.checkpoint_path, model)
    LOGGER.info("created model: {}".format(model))

    # val loss setup
    val_loss = SSDLossWrapper() if "ssd" in args.arch_key.lower() else YoloLossWrapper()
    LOGGER.info("created loss for validation: {}".format(val_loss))

    if not args.eval_mode:
        # train loss setup, different from val if using inception
        train_loss = (
            SSDLossWrapper() if "ssd" in args.arch_key.lower() else YoloLossWrapper()
        )
        LOGGER.info("created loss for validation: {}".format(val_loss))

        # optimizer setup
        if args.optim == "SGD":
            optim_const = SGD
        elif args.optim == "Adam":
            optim_const = Adam
        elif args.optim == "RMSProp":
            optim_const = RMSprop
        else:
            raise ValueError(
                "unsupported value given for optim_type of {}".format(args.optim_type)
            )

        optim = optim_const(model.parameters(), lr=args.init_lr, **args.optim_args)
        LOGGER.info("created optimizer: {}".format(optim))
        LOGGER.info(
            "note, the lr for the optimizer may not reflect the manager yet until "
            "the recal config is created and run"
        )

        # restore from previous check point
        if args.checkpoint_path:
            # currently optimizer restoring is unsupported
            # mapping of the restored params to the correct device is not working
            # load_optimizer(args.checkpoint_path, optim)
            epoch = load_epoch(args.checkpoint_path) + 1
            LOGGER.info(
                "restored checkpoint from {} for epoch {}".format(
                    args.checkpoint_path, epoch - 1
                )
            )
        else:
            epoch = 0

        # recal setup
        add_mods = (
            ConstantKSModifier.from_sparse_model(model)
            if args.sparse_transfer_learn
            else None
        )
        manager = ScheduledModifierManager.from_yaml(
            file_path=args.recal_config_path, add_modifiers=add_mods
        )
        optim = ScheduledOptimizer(
            optim, model, manager, steps_per_epoch=len(train_loader), loggers=loggers,
        )
        LOGGER.info("created manager: {}".format(manager))
    else:
        epoch = 0
        train_loss = None
        optim = None
        manager = None

    # device setup
    model, device, device_ids = model_to_device(model, args.device)
    LOGGER.info("running on device {} for ids {}".format(device, device_ids))

    trainer = (
        ModuleTrainer(
            model,
            device,
            train_loss,
            optim,
            loggers=loggers,
            use_mixed_precision=args.use_mixed_precision,
        )
        if not args.eval_mode
        else None
    )
    tester = ModuleTester(model, device, val_loss, loggers=loggers, log_steps=-1)

    # initial baseline eval run
    tester.run_epoch(val_loader, epoch=epoch - 1, max_steps=args.debug_steps)

    if not args.eval_mode:
        LOGGER.info("starting training from epoch {}".format(epoch))

        if epoch > 0:
            LOGGER.info("adjusting ScheduledOptimizer to restore point")
            optim.adjust_current_step(epoch, 0)

        best_loss = None
        val_res = None

        while epoch < manager.max_epochs:
            if args.debug_steps > 0:
                # correct since all optimizer steps are not
                # taken in the epochs for debug mode
                optim.adjust_current_step(epoch, 0)

            train_res = trainer.run_epoch(
                train_loader, epoch, max_steps=args.debug_steps
            )
            val_res = tester.run_epoch(val_loader, epoch, max_steps=args.debug_steps)
            epoch_val_loss = val_res.result_mean(DEFAULT_LOSS_KEY).item()

            if epoch >= args.save_best_after and (
                best_loss is None or epoch_val_loss <= best_loss
            ):
                _save_model(
                    model,
                    optim,
                    input_shape,
                    "checkpoint-best",
                    save_dir,
                    epoch,
                    val_res,
                )
                best_loss = epoch_val_loss

            if args.save_epochs and epoch in args.save_epochs:
                _save_model(
                    model,
                    optim,
                    input_shape,
                    "checkpoint-{:04d}-{:.04f}".format(epoch, epoch_val_loss),
                    save_dir,
                    epoch,
                    val_res,
                )

            epoch += 1

        # export the final model
        LOGGER.info("completed...")
        _save_model(model, optim, input_shape, "model", save_dir, epoch, val_res)

    LOGGER.info("layer sparsities:")
    for (name, layer) in get_prunable_layers(model):
        LOGGER.info(
            "{}.weight: {:.4f}".format(name, tensor_sparsity(layer.weight).item())
        )

    # add sleep to make sure all background processes have finished,
    # ex tensorboard writing
    time.sleep(5)


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
