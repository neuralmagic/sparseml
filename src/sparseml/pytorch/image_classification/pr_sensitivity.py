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
Usage: sparseml.image_classification.pr_sensitivity [OPTIONS]

  Run kernel sparsity (pruning) analysis for a desired image classification
  architecture

Options:
  --dataset TEXT                  The dataset to use for analysis, ex:
                                  `imagenet`, `imagenette`, `cifar10`, etc.
                                  Set to `imagefolder` for a generic dataset
                                  setup with imagefolder type structure like
                                  imagenet or loadable by a dataset in
                                  `sparseml.pytorch.datasets`  [required]
  --dataset-path, --dataset_path DIRECTORY
                                  The root dir path where the dataset is
                                  stored or should be downloaded to if
                                  available  [required]
  --arch_key, --arch-key TEXT     The architecture key for image
                                  classification model; example: `resnet50`,
                                  `mobilenet`. Note: Will be read from the
                                  checkpoint if not specified
  --checkpoint-path, --checkpoint_path TEXT
                                  A path to a previous checkpoint to load the
                                  state from and resume the state for. If
                                  provided, pretrained will be ignored
  --pretrained TEXT               The type of pretrained weights to use, loads
                                  default pretrained weights for the model if
                                  not specified or set to `True`. Otherwise
                                  should be set to the desired weights type:
                                  [base, optim, optim-perf]. To not load any
                                  weights set to one of [none, false]
                                  [default: True]
  --pretrained-dataset, --pretrained_dataset TEXT
                                  The dataset to load pretrained weights for
                                  if pretrained is set. Load the default
                                  dataset for the architecture if set to None.
                                  examples:`imagenet`, `cifar10`, etc...
  --model-kwargs, --model_kwargs TEXT
                                  Keyword arguments to be passed to model
                                  constructor, should be given as a json
                                  object
  --dataset-kwargs, --dataset_kwargs TEXT
                                  Keyword arguments to be passed to dataset
                                  constructor, should be specified as a json
                                  object
  --model-tag, --model_tag TEXT   A tag for saving results under save-dir,
                                  defaults to the model arch and dataset used
  --save-dir, --save_dir DIRECTORY
                                  The path to the directory for saving results
                                  [default: pytorch_vision]
  --device TEXT                   The device to run on (can also include ids
                                  for data parallel), ex: cpu, cuda, cuda:0,1
                                  [default: cpu]
  --loader-num-workers, --loader_num_workers INTEGER
                                  The number of workers to use for data
                                  loading
  --loader-pin-memory, --loader_pin_memory / --loader-no-pin-memory,
  --loader_no_pin_memory
                                  Use pinned memory for data loading
                                  [default: loader-pin-memory]
  --steps-per-measurement, --steps_per_measurement INTEGER
                                  The number of steps (batches) to run for
                                  each measurement  [default: 20]
  --batch-size, --batch_size INTEGER
                                  The batch size to use for analysis
                                  [default: 64]
  --approximate                   approximate without running data through the
                                  model(uses one shot analysis if
                                  --approximate not passed)  [default: False]
  --image-size, --image_size INTEGER
                                  The size of the image input to the model.
                                  Value should be equal to S for [C, S, S] or
                                  [S, S, C] dimensional input  [default: 224]
  --help                          Show this message and exit.

#########
EXAMPLES
#########

##########
Example command for running one shot KS sens analysis on ssd300_resnet50 for
coco:
sparseml.image_classification.pr_sensitivity \
    --arch-key ssd300_resnet50 --dataset coco \
    --dataset-path ~/datasets/coco-detection

Note: Might need to install pycocotools using pip
"""
import json
import os
from typing import Any, Dict, List, Optional, Union

from torch.nn import Module
from torch.utils.data import DataLoader

import click
from sparseml import get_main_logger
from sparseml.pytorch.image_classification.utils import cli_helpers, helpers
from sparseml.pytorch.optim import (
    pruning_loss_sens_magnitude,
    pruning_loss_sens_one_shot,
)
from sparseml.pytorch.utils import (
    CrossEntropyLossWrapper,
    default_device,
    model_to_device,
)


CURRENT_TASK = helpers.Tasks.PR_SENSITIVITY
LOGGER = get_main_logger()


@click.command()
@click.option(
    "--dataset",
    type=str,
    required=True,
    help="The dataset to use for analysis, "
    "ex: `imagenet`, `imagenette`, `cifar10`, etc. "
    "Set to `imagefolder` for a generic dataset setup with "
    "imagefolder type structure like imagenet or loadable by "
    "a dataset in `sparseml.pytorch.datasets`",
)
@click.option(
    "--dataset-path",
    "--dataset_path",
    type=click.Path(dir_okay=True, file_okay=False),
    callback=cli_helpers.create_dir_callback,
    required=True,
    help="The root dir path where the dataset is stored or should "
    "be downloaded to if available",
)
@click.option(
    "--arch_key",
    "--arch-key",
    type=str,
    default=None,
    help="The architecture key for image classification model; "
    "example: `resnet50`, `mobilenet`. "
    "Note: Will be read from the checkpoint if not specified",
)
@click.option(
    "--checkpoint-path",
    "--checkpoint_path",
    type=str,
    default=None,
    help="A path to a previous checkpoint to load the state from "
    "and resume the state for. If provided, pretrained will "
    "be ignored",
)
@click.option(
    "--pretrained",
    type=str,
    default=True,
    show_default=True,
    help="The type of pretrained weights to use, "
    "loads default pretrained weights for "
    "the model if not specified or set to `True`. "
    "Otherwise should be set to the desired weights "
    "type: [base, optim, optim-perf]. To not load any weights set "
    "to one of [none, false]",
)
@click.option(
    "--pretrained-dataset",
    "--pretrained_dataset",
    type=str,
    default=None,
    show_default=True,
    help="The dataset to load pretrained weights for if pretrained is "
    "set. Load the default dataset for the architecture if set to None. "
    "examples:`imagenet`, `cifar10`, etc...",
)
@click.option(
    "--model-kwargs",
    "--model_kwargs",
    default=json.dumps({}),
    type=str,
    callback=cli_helpers.parse_json_callback,
    help="Keyword arguments to be passed to model constructor, should "
    "be given as a json object",
)
@click.option(
    "--dataset-kwargs",
    "--dataset_kwargs",
    default=json.dumps({}),
    type=str,
    callback=cli_helpers.parse_json_callback,
    help="Keyword arguments to be passed to dataset constructor, "
    "should be specified as a json object",
)
@click.option(
    "--model-tag",
    "--model_tag",
    type=str,
    default=None,
    help="A tag for saving results under save-dir, "
    "defaults to the model arch and dataset used",
)
@click.option(
    "--save-dir",
    "--save_dir",
    type=click.Path(dir_okay=True, file_okay=False),
    default="pytorch_vision",
    callback=cli_helpers.create_dir_callback,
    show_default=True,
    help="The path to the directory for saving results",
)
@click.option(
    "--device",
    default=default_device(),
    show_default=True,
    help="The device to run on (can also include ids for data "
    "parallel), ex: cpu, cuda, cuda:0,1",
)
@click.option(
    "--loader-num-workers",
    "--loader_num_workers",
    type=int,
    default=4,
    help="The number of workers to use for data loading",
)
@click.option(
    "--loader-pin-memory/--loader-no-pin-memory",
    "--loader_pin_memory/--loader_no_pin_memory",
    default=True,
    is_flag=True,
    show_default=True,
    help="Use pinned memory for data loading",
)
@click.option(
    "--steps-per-measurement",
    "--steps_per_measurement",
    type=int,
    default=20,
    show_default=True,
    help="The number of steps (batches) to run for each measurement",
)
@click.option(
    "--batch-size",
    "--batch_size",
    type=int,
    default=64,
    show_default=True,
    help="The batch size to use for analysis",
)
@click.option(
    "--approximate",
    is_flag=True,
    show_default=True,
    help="approximate without running data through the model"
    "(uses one shot analysis if --approximate not passed)",
)
@click.option(
    "--image-size",
    "--image_size",
    type=int,
    default=224,
    show_default=True,
    help="The size of the image input to the model. Value should be "
    "equal to S for [C, S, S] or [S, S, C] dimensional input",
)
def main(
    dataset: str,
    dataset_path: str,
    arch_key: Optional[str],
    checkpoint_path: Optional[str],
    pretrained: Union[str, bool],
    pretrained_dataset: Optional[str],
    model_kwargs: Dict[str, Any],
    dataset_kwargs: Dict[str, Any],
    model_tag: Optional[str],
    save_dir: str,
    device: Optional[str],
    loader_num_workers: int,
    loader_pin_memory: bool,
    steps_per_measurement: int,
    batch_size: int,
    approximate: bool,
    image_size: int,
):
    """
    Run kernel sparsity (pruning) analysis for a
    desired image classification architecture
    """
    is_main_process = True
    local_rank = -1

    save_dir, loggers = helpers.get_save_dir_and_loggers(
        task=CURRENT_TASK,
        is_main_process=is_main_process,
        save_dir=save_dir,
        arch_key=arch_key,
        model_tag=model_tag,
        dataset_name=dataset,
    )

    train_dataset, train_loader = (
        helpers.get_dataset_and_dataloader(
            dataset_name=dataset,
            dataset_path=dataset_path,
            batch_size=batch_size,
            image_size=image_size,
            dataset_kwargs=dataset_kwargs,
            training=True,
            loader_num_workers=loader_num_workers,
            loader_pin_memory=loader_pin_memory,
        )
        if not approximate
        else (None, None)
    )

    val_dataset = None

    num_classes = helpers.infer_num_classes(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        dataset=dataset,
        model_kwargs=model_kwargs,
    )

    model, arch_key, _ = helpers.create_model(
        checkpoint_path=checkpoint_path,
        recipe_path=None,
        num_classes=num_classes,
        arch_key=arch_key,
        pretrained=pretrained,
        pretrained_dataset=pretrained_dataset,
        local_rank=local_rank,
        **model_kwargs,
    )

    pruning_loss_sensitivity(
        model=model,
        train_loader=train_loader,
        save_dir=save_dir,
        loggers=loggers,
        approximate=approximate,
        device=device,
        steps_per_measurement=steps_per_measurement,
    )


def pruning_loss_sensitivity(
    model: Module,
    train_loader: DataLoader,
    save_dir: str,
    loggers: List[Any],
    approximate: bool,
    device: Union[str, int],
    steps_per_measurement: int,
) -> None:
    """
    Utility function for pruning sensitivity analysis

    :param model: loaded model architecture to analyse
    :param train_loader: A DataLoader for training data
    :param save_dir: Directory to save results
    :param loggers: List of loggers to use during analysis
    :param approximate: Whether to use one shot analysis
    :param device: Device to use for analysis
    :param steps_per_measurement: Number of steps to run for each measurement
    """
    # loss setup
    if not approximate:
        loss = CrossEntropyLossWrapper()
        LOGGER.info(f"created loss: {loss}")
    else:
        loss = None

    # device setup
    if not approximate:
        module, device, device_ids = model_to_device(
            model=model,
            device=device,
        )
    else:
        device = None

    # kernel sparsity analysis
    if approximate:
        analysis = pruning_loss_sens_magnitude(model)
    else:
        analysis = pruning_loss_sens_one_shot(
            module=model,
            data=train_loader,
            loss=loss,
            device=device,
            steps_per_measurement=steps_per_measurement,
            tester_loggers=loggers,
        )

    # saving and printing results
    LOGGER.info("completed...")
    LOGGER.info(f"Saving results in {save_dir}")
    output_name = (
        "ks_approx_sensitivity.json" if approximate else "ks_one_shot_sensitivity.json"
    )
    analysis.save_json(
        os.path.join(
            save_dir,
            f"{output_name}.json",
        )
    )
    analysis.plot(
        os.path.join(
            save_dir,
            os.path.join(
                save_dir,
                f"{output_name}.png",
            ),
        ),
        plot_integral=True,
    )
    analysis.print_res()


if __name__ == "__main__":
    main()
