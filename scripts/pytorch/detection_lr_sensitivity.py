"""
Object Detection learning rate sensitivity script.
Setup to support the following use cases:
- learning rate sensitivity analysis while training the model

Saves the results to a given directory.
Additionally will print the results out to the command line


##########
Command help:
usage: detection_lr_sensitivity.py [-h] --arch-key ARCH_KEY
                                   [--pretrained PRETRAINED]
                                   [--pretrained-dataset PRETRAINED_DATASET]
                                   [--checkpoint-path CHECKPOINT_PATH]
                                   [--device DEVICE] --dataset DATASET
                                   --dataset-path DATASET_PATH --batch-size
                                   BATCH_SIZE
                                   [--loader-num-workers LOADER_NUM_WORKERS]
                                   [--loader-pin-memory LOADER_PIN_MEMORY]
                                   [--steps-per-measurement STEPS_PER_MEASUREMENT]
                                   [--init-lr INIT_LR] [--final-lr FINAL_LR]
                                   [--optim-args OPTIM_ARGS]
                                   [--model-tag MODEL_TAG]
                                   [--save-dir SAVE_DIR]

Run a learning rate sensitivity analysis for a desired object detection
architecture

optional arguments:
  -h, --help            show this help message and exit
  --arch-key ARCH_KEY   The type of model to create, ex: ssd300_resnet50put as
                        help to see the full list (will raise an exception
                        with the list)
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
                        coco, voc, etc
  --checkpoint-path CHECKPOINT_PATH
                        A path to a previous checkpoint to load the state from
                        and resume the state for. If provided, pretrained will
                        be ignored
  --device DEVICE       The device to run on (can also include ids for data
                        parallel), ex: cpu, cuda, cuda:0,1
  --dataset DATASET     The dataset to use for training, ex: coco, voc-
                        detection, etc.
  --dataset-path DATASET_PATH
                        The root path to where the dataset is stored
  --batch-size BATCH_SIZE
                        The batch size to use while training
  --loader-num-workers LOADER_NUM_WORKERS
                        The number of workers to use for data loading
  --loader-pin-memory LOADER_PIN_MEMORY
                        Use pinned memory for data loading
  --steps-per-measurement STEPS_PER_MEASUREMENT
                        The number of steps (batches) to run for each lr
                        measurement
  --init-lr INIT_LR     The initial learning rate to use for the sensitivity
                        analysis
  --final-lr FINAL_LR   The final learning rate to use for the sensitivity
                        analysis
  --optim-args OPTIM_ARGS
                        Additional args to be passed to the optimizer passed
                        in as a json dict
  --model-tag MODEL_TAG
                        A tag to use for the model for saving results under
                        save-dir and in tensorboard, defaults to the model
                        arch and dataset used
  --save-dir SAVE_DIR   The path to the directory for saving results


##########
Example command for running LR sensitivity analysis on mobilenet:
python scripts/pytorch/detection_lr_sensitivity.py \
    --arch-key ssd300_resnet50 --dataset coco \
    --dataset-path ~/datasets/coco-detection --batch-size 2
"""

import argparse
import os
import json

from torch.utils.data import DataLoader
from torch.optim import SGD
import torch.nn.functional as TF

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
    model_to_device,
    default_device,
    PythonLogger,
)
from neuralmagicML.pytorch.recal import (
    lr_loss_sensitivity,
    default_exponential_check_lrs,
)
from neuralmagicML.utils import create_dirs


LOGGER = get_main_logger()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a learning rate sensitivity analysis for a desired object "
        "detection architecture",
    )

    # model args
    parser.add_argument(
        "--arch-key",
        type=str,
        required=True,
        help="The type of model to create, ex: ssd300_resnet50"
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
        " Ex can be set to coco, voc, etc",
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
        help="The dataset to use for training, " "ex: coco, voc-detection, etc.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="The root path to where the dataset is stored",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="The batch size to use while training",
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
        "--steps-per-measurement",
        type=int,
        default=20,
        help="The number of steps (batches) to run for each lr measurement",
    )
    parser.add_argument(
        "--init-lr",
        type=float,
        default=10e-6,
        help="The initial learning rate to use for the sensitivity analysis",
    )
    parser.add_argument(
        "--final-lr",
        type=float,
        default=0.5,
        help="The final learning rate to use for the sensitivity analysis",
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
        default="pytorch_detection_lr_sensitivity",
        help="The path to the directory for saving results",
    )

    return parser.parse_args()


def main(args):
    # logging and saving setup
    save_dir = os.path.abspath(os.path.expanduser(args.save_dir))

    model_tag = (
        "{}_{}".format(args.arch_key.replace("/", "."), args.dataset)
        if not args.model_tag
        else args.model_tag
    )
    model_id = model_tag
    model_inc = 0

    while os.path.exists(os.path.join(save_dir, model_id)):
        model_inc += 1
        model_id = "{}__{:02d}".format(model_tag, model_inc)

    save_dir = os.path.join(save_dir, model_id)
    create_dirs(save_dir)

    # loggers setup
    LOGGER.info("Model id is set to {}".format(model_id))

    # dataset creation
    input_shape = ModelRegistry.input_shape(args.arch_key)
    image_size = input_shape[1]  # assume shape [C, S, S] where S is the image size
    dataset_kwargs = {}
    if "coco" in args.dataset.lower() or "voc" in args.dataset.lower():
        if "ssd" in args.arch_key.lower():
            dataset_kwargs["preprocessing_type"] = "ssd"
        elif "yolo" in args.arch_key.lower():
            dataset_kwargs["preprocessing_type"] = "yolo"

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
        batch_size=args.batch_size,
        collate_fn=ssd_collate_fn if "ssd" in args.arch_key.lower() else yolo_collate_fn,
        shuffle=True,
        num_workers=args.loader_num_workers,
        pin_memory=args.loader_pin_memory,
    )
    LOGGER.info("created train_dataset: {}".format(train_dataset))

    # model creation
    if args.dataset == "imagefolder":
        num_classes = train_dataset.num_classes
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
    LOGGER.info("created model: {}".format(model))

    # optimizer setup
    optim = SGD(model.parameters(), lr=args.init_lr, **args.optim_args)
    LOGGER.info("created optimizer: {}".format(optim))

    # loss setup
    loss = SSDLossWrapper() if "ssd" in args.arch_key.lower() else YoloLossWrapper()
    LOGGER.info("created loss: {}".format(loss))

    # device setup
    module, device, device_ids = model_to_device(model, args.device)

    # learning rate analysis
    LOGGER.info("running analysis: {}".format(loss))
    analysis = lr_loss_sensitivity(
        model,
        train_loader,
        loss,
        optim,
        device,
        args.steps_per_measurement,
        check_lrs=default_exponential_check_lrs(args.init_lr, args.final_lr),
        trainer_loggers=[PythonLogger()],
    )

    # saving and printing results
    LOGGER.info("completed...")
    LOGGER.info("Saving results in {}".format(save_dir))
    analysis.save_json(os.path.join(save_dir, "lr_sensitivity.json"))
    analysis.plot(os.path.join(save_dir, "lr_sensitivity.png"))
    analysis.print_res()


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
