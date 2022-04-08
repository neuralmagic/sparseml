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

"""
#####
Command help:
usage: lr_analysis.py [-h] --batch-size BATCH_SIZE --arch-key ARCH_KEY
                        --dataset DATASET --dataset-path DATASET_PATH [
                        --pretrained PRETRAINED]
                      [--pretrained-dataset PRETRAINED_DATASET] [
                      --model-kwargs MODEL_KWARGS]
                      [--dataset-kwargs DATASET_KWARGS] [--model-tag MODEL_TAG]
                      [--save-dir SAVE_DIR] [--device DEVICE]
                      [--loader-num-workers LOADER_NUM_WORKERS] [
                      --no-loader-pin-memory]
                      [--loader-pin-memory [LOADER_PIN_MEMORY]]
                      [--checkpoint-path CHECKPOINT_PATH] [--init-lr INIT_LR]
                      [--optim-args OPTIM_ARGS] [--final-lr FINAL_LR]
                      [--steps-per-measurement STEPS_PER_MEASUREMENT]

Utility script to Run a learning rate sensitivity analysis for a desired
image classification architecture

optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        The batch size to use for analysis
  --arch-key ARCH_KEY   The type of model to use, ex: resnet50, vgg16, mobilenet
                        put as help to see the full list (will raise an
                        exception with the list)
  --dataset DATASET     The dataset to use for analysis, ex: imagenet,
                        imagenette, cifar10,
                        etc. Set to imagefolder for a generic dataset setup
                        with an imagefolder structure setup like imagenet or
                        loadable by a dataset in sparseml.pytorch.datasets
  --dataset-path DATASET_PATH
                        The root path to where the dataset is stored
  --pretrained PRETRAINED
                        The type of pretrained weights to use, default is
                        true to load the default pretrained weights for the
                        model. Otherwise should be set to the desired weights
                        type: [base, optim, optim-perf]. To not load any
                        weights set to one of [none, false]
  --pretrained-dataset PRETRAINED_DATASET
                        The dataset to load pretrained weights for if pretrained
                        is set. Default is None which will load the default
                        dataset for the architecture. Ex can be set to
                        imagenet, cifar10, etc
  --model-kwargs MODEL_KWARGS
                        Keyword arguments to be passed to model constructor,
                        should be given as a json object
  --dataset-kwargs DATASET_KWARGS
                        Keyword arguments to be passed to dataset
                        constructor, should be given as a json object
  --model-tag MODEL_TAG
                        A tag to use for the model for saving results under
                        save-dir, defaults to the model arch and dataset used
  --save-dir SAVE_DIR   The path to the directory for saving results
  --device DEVICE       The device to run on (can also include ids for
                        data parallel), ex: cpu, cuda, cuda:0,1
  --loader-num-workers LOADER_NUM_WORKERS
                        The number of workers to use for data loading
  --no-loader-pin-memory
                        Do not use pinned memory for data loading
  --loader-pin-memory [LOADER_PIN_MEMORY]
                        Use pinned memory for data loading
  --checkpoint-path CHECKPOINT_PATH
                        A path to a previous checkpoint to load the state from
                        and resume the state for. If provided, pretrained
                        will be ignored. If using a SparseZoo recipe,
                        can also provide 'zoo' to load the base weights
                        associated with that recipe
  --init-lr INIT_LR     The initial learning rate to use for the sensitivity
                        analysis
  --optim-args OPTIM_ARGS
                        Additional args to be passed to the optimizer passed in
                        as a json object
  --final-lr FINAL_LR   The final learning rate to use for the sensitivity
                        analysis
  --steps-per-measurement STEPS_PER_MEASUREMENT The number of steps (batches)
                        to run for each measurement
#########
EXAMPLES
#########

##########
Example command for running LR sensitivity analysis on mobilenet:
python sparseml.image_classification.lr_analysis \
    --arch-key mobilenet --dataset imagenet \
    --dataset-path ~/datasets/ILSVRC2012 --batch-size 2

Example command for running LR sensitivity analysis on mobilenet using
imagenette:
sparseml.image_classification.lr_analysis \
    --arch-key mobilenet --dataset imagenette \
    --dataset-path ~/datasets/ --batch-size 2
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from torch.nn import Module
from torch.optim import SGD
from torch.utils.data import DataLoader

from sparseml import get_main_logger
from sparseml.pytorch.image_classification.utils import NmArgumentParser, helpers
from sparseml.pytorch.models import ModelRegistry
from sparseml.pytorch.optim import default_exponential_check_lrs, lr_loss_sensitivity
from sparseml.pytorch.utils import (
    CrossEntropyLossWrapper,
    PythonLogger,
    default_device,
    model_to_device,
)


CURRENT_TASK = helpers.Tasks.LR_ANALYSIS
LOGGER = get_main_logger()


@dataclass
class LRAnalysisArguments:
    """
    Represents the arguments we use in our PyTorch integration scripts for
    learning rate analysis.
    Using :class:`NmArgumentParser` we can turn this class into `argparse
    <https://docs.python.org/3/library/argparse.html#module-argparse>`
    arguments that can be specified on the command line.

    :param batch_size: The batch size to use for analysis.
    :param arch_key: A str key representing the type of model to use,
        ex:resnet50.
    :param dataset: The dataset to use for analysis, ex imagenet,
        imagenette, etc; Set to `imagefolder` for a custom dataset.
    :param dataset_path: Root path to dataset location.
    :param pretrained: The type of pretrained weights to use default is true to
        load the default pretrained weights for the model. Otherwise should be
        set to the desired weights type: [base, optim, optim-perf];
        To not load any weights set to one of [none, false].
    :param pretrained_dataset: str representing the dataset to load pretrained
        weights for; if pretrained is set; Default is None which will load the
        default dataset for the architecture; Ex can be set to imagenet,
        cifar10, etc.
    :param model_kwargs: json object containing keyword arguments to be passed
        to the model constructor.
    :param dataset_kwargs: json object to load keyword arguments to be passed
        to dataset constructor.
    :param model_tag: A str tag to use for the model for saving results under
        save-dir, defaults to the model arch and dataset used.
    :param save_dir: The path to the directory for saving results,
        default="pytorch_vision".
    :param device: str represnting the device to run on (can also include ids
        for data parallel), ex:{cpu, cuda, cuda:0,1}.
    :param loader_num_workers: int number of workers to use for data loading,
        default=4.
    :param loader_pin_memory: bool to use pinned memory for data loading,
        default=True.
    :param checkpoint_path: A path to a previous checkpoint to load the state
        from and resume the state for; Also works with SparseZoo recipes;
        Set to zoo to automatically download and load weights associated with a
        recipe.
    :param init_lr: float representing the initial learning for analysis,
        default=1e-5.
    :param optim_args: Additional arguments to be passed in to the optimizer as
        a json object.
    :param final_lr: The final learning rate to use for the sensitivity
        analysis.
    :param steps_per_measurement: The number of steps (batches) to run for each
        measurement.
    """

    batch_size: int = field(metadata={"help": "The batch size to use for analysis"})

    dataset: str = field(
        metadata={
            "help": "The dataset to use for analysis, "
            "ex: imagenet, imagenette, cifar10, etc. "
            "Set to imagefolder for a generic dataset setup "
            "with an image folder structure setup like imagenet or "
            "loadable by a dataset in sparseml.pytorch.datasets"
        }
    )

    dataset_path: str = field(
        metadata={
            "help": "The root path to where the dataset is stored",
        }
    )
    pretrained: str = field(
        default=True,
        metadata={
            "help": "The type of pretrained weights to use, "
            "default is true to load the default pretrained weights "
            "for the model. "
            "Otherwise should be set to the desired weights type: "
            "[base, optim, optim-perf]. "
            "To not load any weights set to one of [none, false]"
        },
    )
    arch_key: Optional[str] = field(
        default=None,
        metadata={
            "help": "The type of model to use, ex: resnet50, vgg16, mobilenet "
            "put as help to see the full list"
            "(will raise an exception with the list)",
        },
    )

    pretrained_dataset: str = field(
        default=None,
        metadata={
            "help": "The dataset to load pretrained weights for if pretrained is"
            "set. Default is None which will load the default dataset for "
            "the architecture. Ex can be set to imagenet, cifar10, etc.",
        },
    )

    model_kwargs: json.loads = field(
        default_factory=lambda: {},
        metadata={
            "help": "Keyword arguments to be passed to model constructor, should "
            "be given as a json object"
        },
    )

    dataset_kwargs: json.loads = field(
        default_factory=lambda: {},
        metadata={
            "help": "Keyword arguments to be passed to dataset constructor,"
            " should be given as a json object",
        },
    )

    model_tag: str = field(
        default=None,
        metadata={
            "help": "A tag to use for the model for saving results under save-dir, "
            "defaults to the model arch and dataset used",
        },
    )

    save_dir: str = field(
        default="pytorch_vision",
        metadata={
            "help": "The path to the directory for saving results",
        },
    )

    device: str = field(
        default=default_device(),
        metadata={
            "help": "The device to run on (can also include ids for "
            "data parallel), ex:cpu, cuda, cuda:0,1"
        },
    )

    loader_num_workers: int = field(
        default=4, metadata={"help": "The number of workers to use for data loading"}
    )

    loader_pin_memory: bool = field(
        default=True, metadata={"help": "Use pinned memory for data loading"}
    )

    checkpoint_path: str = field(
        default=None,
        metadata={
            "help": "A path to a previous checkpoint to load the state from "
            "and resume the state for. If provided, pretrained will be"
            "ignored. If using a SparseZoo recipe, can also provide "
            "'zoo' to load the base weights associated with that recipe"
        },
    )

    init_lr: float = field(
        default=1e-5,
        metadata={
            "help": "The initial learning rate to use for the sensitivity analysis"
        },
    )

    optim_args: json.loads = field(
        default_factory=lambda: {},
        metadata={
            "help": "Additional args to be passed to the optimizer passed in"
            " as a json object"
        },
    )

    final_lr: float = field(
        default=0.5,
        metadata={
            "help": "The final learning rate to use for the sensitivity analysis",
        },
    )

    steps_per_measurement: int = field(
        default=20,
        metadata={"help": "The number of steps (batches) to run for each measurement"},
    )

    def __post_init__(self):
        self.arch_key = helpers.get_arch_key(
            arch_key=self.arch_key,
            checkpoint_path=self.checkpoint_path,
        )

        if "preprocessing_type" not in self.dataset_kwargs and (
            "coco" in self.dataset.lower() or "voc" in self.dataset.lower()
        ):
            if "ssd" in self.arch_key.lower():
                self.dataset_kwargs["preprocessing_type"] = "ssd"
            elif "yolo" in self.arch_key.lower():
                self.dataset_kwargs["preprocessing_type"] = "yolo"

        self.is_main_process = True
        self.local_rank = -1
        self.rank = -1


def lr_sensitivity(
    args,
    model: Module,
    train_loader: DataLoader,
    save_dir: str,
) -> None:
    """
    Utility function to run learning rate sensitivity analysis

    :param args: An LRAnalysisArguments object containing config for current
        LR analysis task.
    :param model: loaded model architecture to analyse
    :param train_loader: A DataLoader for training data
    :param save_dir: Directory to save results
    """
    # optimizer setup
    optim = SGD(model.parameters(), lr=args.init_lr, **args.optim_args)
    LOGGER.info(f"created optimizer: {optim}")

    # loss setup
    loss = CrossEntropyLossWrapper()
    LOGGER.info(f"created loss: {loss}")

    # device setup
    model, device, device_ids = model_to_device(model, args.device)

    # learning rate analysis
    LOGGER.info(f"running analysis: {loss}")
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
    LOGGER.info(f"Saving results in {save_dir}")
    analysis.save_json(os.path.join(save_dir, "lr_sensitivity.json"))
    analysis.plot(os.path.join(save_dir, "lr_sensitivity.png"))
    analysis.print_res()


def main():
    """
    Driver function for the script
    """
    _parser = NmArgumentParser(
        dataclass_types=LRAnalysisArguments,
        description="Utility script to Run a "
        "learning rate sensitivity analysis "
        "for a desired image classification architecture",
    )
    args_, _ = _parser.parse_args_into_dataclasses()
    save_dir, loggers = helpers.get_save_dir_and_loggers(
        task=CURRENT_TASK,
        is_main_process=args_.is_main_process,
        save_dir=args_.save_dir,
        arch_key=args_.arch_key,
        model_tag=args_.model_tag,
        dataset_name=args_.dataset,
    )

    input_shape = ModelRegistry.input_shape(args_.arch_key)
    # assume shape [C, S, S] where S is the image size
    image_size = input_shape[1]

    train_dataset, train_loader = helpers.get_dataset_and_dataloader(
        dataset_name=args_.dataset,
        dataset_path=args_.dataset_path,
        batch_size=args_.batch_size,
        image_size=image_size,
        dataset_kwargs=args_.dataset_kwargs,
        training=True,
        loader_num_workers=args_.loader_num_workers,
        loader_pin_memory=args_.loader_pin_memory,
    )

    num_classes = helpers.infer_num_classes(
        train_dataset=train_dataset,
        val_dataset=None,
        dataset=args_.dataset,
        model_kwargs=args_.model_kwargs,
    )
    model, args_.arch_key = helpers.create_model(
        checkpoint_path=args_.checkpoint_path,
        recipe_path=None,
        num_classes=num_classes,
        arch_key=args_.arch_key,
        pretrained=args_.pretrained,
        pretrained_dataset=args_.pretrained_dataset,
        local_rank=args_.local_rank,
        **args_.model_kwargs,
    )
    lr_sensitivity(args_, model, train_loader, save_dir)


if __name__ == "__main__":
    main()
