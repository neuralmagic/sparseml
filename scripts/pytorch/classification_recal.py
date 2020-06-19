"""
Image classification recal script. Setup to support the following use cases:
- training image classification architectures
- pruning image classification architectures
- transfer learning image classification architectures
- evaluating image classification architectures


Command help:
usage: classification_recal.py [-h] [--recal-config-path RECAL_CONFIG_PATH]
                               [--sparse-transfer-learn] [--eval-mode]
                               --arch-key ARCH_KEY [--pretrained PRETRAINED]
                               [--pretrained-dataset PRETRAINED_DATASET]
                               [--checkpoint-path CHECKPOINT_PATH]
                               [--class-type CLASS_TYPE] [--device DEVICE]
                               --dataset DATASET --dataset-path DATASET_PATH
                               --train-batch-size TRAIN_BATCH_SIZE
                               --test-batch-size TEST_BATCH_SIZE
                               [--loader-num-workers LOADER_NUM_WORKERS]
                               [--loader-pin-memory LOADER_PIN_MEMORY]
                               [--optim OPTIM] [--init-lr INIT_LR]
                               [--optim-args OPTIM_ARGS]
                               [--model-tag MODEL_TAG] [--save-dir SAVE_DIR]
                               [--logs-dir LOGS_DIR]
                               [--save-epochs SAVE_EPOCHS [SAVE_EPOCHS ...]]
                               [--debug-steps DEBUG_STEPS]

Train and/or prune an image classification architecture on a dataset

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
  --class-type CLASS_TYPE
                        One of [single, multi] where single is for single
                        class training using a softmax and multi is for multi
                        class training using a sigmoid
  --device DEVICE       The device to run on (can also include ids for data
                        parallel), ex: cpu, cuda, cuda:0,1
  --dataset DATASET     The dataset to use for training, ex: imagenet,
                        imagenette, cifar10, etc. Set to imagefolder for a
                        generic image classification dataset setup with an
                        image folder structure setup like imagenet.
  --dataset-path DATASET_PATH
                        The root path to where the dataset is stored
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
  --save-epochs SAVE_EPOCHS [SAVE_EPOCHS ...]
                        epochs to save checkpoints at
  --debug-steps DEBUG_STEPS
                        Amount of steps to run for training and testing for a
                        debug mode


Example command for training mobilenet on imagenet dataset:
python scripts/pytorch/classification_recal.py \
    --recal-config-path scripts/pytorch/configs/imagenet_training.yaml \
    --arch-key mobilenet --dataset imagenet --dataset-path ~/datasets/ILSVRC2012 \
    --train-batch-size 256 --test-batch-size 1024


Example command for pruning resnet50 on imagenet dataset:
python scripts/pytorch/classification_recal.py \
    --recal-config-path scripts/pytorch/configs/pruning_resnet50.yaml \
    --arch-key resnet50 --dataset imagenet --dataset-path ~/datasets/ILSVRC2012 \
    --train-batch-size 256 --test-batch-size 1024


Example command for transfer learning sparse resnet50 on an image folder dataset:
python scripts/pytorch/classification_recal.py \
    --sparse-transfer-learn \
    --recal-config-path scripts/pytorch/configs/pruning_resnet50.yaml \
    --arch-key resnet50 --pretrained recal-perf \
    --dataset imagenet --dataset-path ~/datasets/ILSVRC2012 \
    --train-batch-size 256 --test-batch-size 1024


Example command for evaluating a mobilenetv2 model on imagenet dataset:
python scripts/pytorch/classification_recal.py \
    --eval-mode --arch-key mobilenetv2 --dataset imagenet \
    --dataset-path ~/datasets/ILSVRC2012 --train-batch-size 256 --test-batch-size 1024
"""

import argparse
import os
import time
import json

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
import torch.nn.functional as TF

from neuralmagicML.pytorch.datasets import DatasetRegistry
from neuralmagicML.pytorch.models import ModelRegistry
from neuralmagicML.pytorch.utils import (
    LossWrapper,
    TopKAccuracy,
    ModuleTrainer,
    ModuleTester,
    TensorBoardLogger,
    PythonLogger,
    model_to_device,
    save_model,
    load_optimizer,
    default_device,
    ModuleExporter,
    get_prunable_layers,
    tensor_sparsity,
)
from neuralmagicML.pytorch.recal import (
    ScheduledModifierManager,
    ScheduledOptimizer,
    ConstantKSModifier,
)
from neuralmagicML.utils import create_dirs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and/or prune an image classification "
        "architecture on a dataset"
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
    parser.add_argument(
        "--class-type",
        type=str,
        default="single",
        help="One of [single, multi] where single is for single class training "
        "using a softmax and multi is for multi class training using a sigmoid",
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
        "ex: imagenet, imagenette, cifar10, etc. "
        "Set to imagefolder for a generic image classification dataset setup "
        "with an image folder structure setup like imagenet.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="The root path to where the dataset is stored",
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
        default=10e-9,
        help="The initial learning rate to use while training, "
        "set to a low value because LR should come from the recal config path",
    )
    parser.add_argument(
        "--optim-args",
        type=json.loads,
        default="{}",
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
        default="pytorch_classification_pruning",
        help="The path to the directory for saving results",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default=os.path.join("pytorch_classification_pruning", "tensorboard-logs"),
        help="The path to the directory for saving logs",
    )
    parser.add_argument(
        "--save-epochs",
        type=int,
        default=[],
        nargs="+",
        help="epochs to save checkpoints at",
    )

    # debug options
    parser.add_argument(
        "--debug-steps",
        type=int,
        default=-1,
        help="Amount of steps to run for training and testing for a debug mode",
    )

    return parser.parse_args()


def main(args):
    # logging and saving setup
    save_dir = os.path.abspath(os.path.expanduser(args.save_dir))
    logs_dir = os.path.abspath(os.path.expanduser(os.path.join(args.logs_dir)))

    model_tag = (
        "{}_{}".format(args.arch_key.replace("/", "."), args.dataset)
        if not args.model_tag
        else args.model_tag
    )
    model_id = model_tag
    model_inc = 0

    while os.path.exists(os.path.join(logs_dir, model_id)):
        model_inc += 1
        model_id = "{}__{:02d}".format(model_tag, model_inc)

    save_dir = os.path.join(save_dir, model_id)
    logs_dir = os.path.join(logs_dir, model_id)
    create_dirs(save_dir)
    create_dirs(logs_dir)

    # loggers setup
    tb_logger = TensorBoardLogger(log_path=logs_dir)
    py_logger = PythonLogger()
    loggers = [tb_logger, py_logger]
    py_logger.info("Model id is set to {}".format(model_id))

    # dataset creation
    input_shape = ModelRegistry.input_shape(args.arch_key)
    image_size = input_shape[1]  # assume shape [C, S, S] where S is the image size

    if not args.eval_mode:
        train_dataset = DatasetRegistry.create(
            args.dataset,
            root=args.dataset_path,
            train=True,
            rand_trans=True,
            image_size=image_size,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.loader_num_workers,
            pin_memory=args.loader_pin_memory,
        )
        py_logger.info("created train_dataset: {}".format(train_dataset))
    else:
        train_dataset = None
        train_loader = None

    val_dataset = DatasetRegistry.create(
        args.dataset,
        root=args.dataset_path,
        train=False,
        rand_trans=False,
        image_size=image_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.loader_num_workers,
        pin_memory=args.loader_pin_memory,
    )
    py_logger.info("created val_dataset: {}".format(val_dataset))

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
        class_type=args.class_type,
    )
    py_logger.info("created model: {}".format(model))

    # loss setup
    loss = LossWrapper(
        loss_fn=TF.cross_entropy,
        extras={"top1acc": TopKAccuracy(1), "top5acc": TopKAccuracy(5)},
    )
    py_logger.info("created loss: {}".format(loss))

    epoch = 0

    if not args.eval_mode:
        # optimizer setup
        if args.optim == "SGD":
            optim_const = SGD
        elif args.optim == "Adam":
            optim_const = Adam
        else:
            raise ValueError(
                "unsupported value given for optim_type of {}".format(args.optim_type)
            )

        optim = optim_const(model.parameters(), lr=args.init_lr, **args.optim_args)
        py_logger.info("created optimizer: {}".format(optim))

        # restore from previous check point
        if args.checkpoint_path:
            epoch = load_optimizer(args.checkpoint_path, optim)
            py_logger.info(
                "restored checkpoint from {} for epoch {}".format(
                    args.checkpoint_path, epoch
                )
            )

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
        optim.adjust_current_step(epoch, 0)  # adjust in case this is restored
        py_logger.info("created manager: {}".format(manager))
    else:
        optim = None
        manager = None

    # device setup
    model, device, device_ids = model_to_device(model, args.device)

    trainer = (
        ModuleTrainer(model, device, loss, optim, loggers=loggers)
        if not args.eval_mode
        else None
    )
    tester = ModuleTester(model, device, loss, loggers=loggers)

    # initial baseline eval run
    tester.run_epoch(val_loader, epoch=epoch - 1, max_steps=args.debug_steps)

    if not args.eval_mode:
        while epoch < manager.max_epochs:
            if args.debug_steps > 0:
                # correct since all optimizer steps are not
                # taken in the epochs for debug mode
                optim.adjust_current_step(epoch, 0)

            trainer.run_epoch(train_loader, epoch, max_steps=args.debug_steps)
            tester.run_epoch(val_loader, epoch, max_steps=args.debug_steps)

            if args.save_epochs and epoch in args.save_epochs:
                save_path = os.path.join(
                    save_dir, "pytorch", "checkpoint-{:04d}.pth".format(epoch)
                )
                py_logger.info("Saving checkpoint to {}".format(save_path))
                save_model(save_path, model, optim, epoch)

            epoch += 1

    # export the final model
    py_logger.info("completed...")

    if not args.eval_mode:
        py_logger.info("Saving final model in {}".format(save_dir))
        exporter = ModuleExporter(model, save_dir)
        exporter.export_pytorch(optim, epoch)
        exporter.export_onnx(torch.randn(1, *input_shape))

    py_logger.info("layer sparsities:")
    for (name, layer) in get_prunable_layers(model):
        py_logger.info(
            "{}.weight: {:.4f}".format(name, tensor_sparsity(layer.weight).item())
        )

    # add sleep to make sure all background processes have finished,
    # ex tensorboard writing
    time.sleep(5)


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
