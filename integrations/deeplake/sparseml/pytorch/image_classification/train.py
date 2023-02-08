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
######
Command help:
Usage: sparseml.image_classification.train [OPTIONS]

  PyTorch training integration with SparseML for image classification models

Options:
  --train-batch-size, --train_batch_size INTEGER
                                  Train batch size  [required]
  --test-batch-size, --test_batch_size INTEGER
                                  Test/Validation batch size  [required]
  --dataset TEXT                  The dataset to use for training, ex:
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
                                  provided, pretrained will be ignored . If
                                  using a SparseZoo recipe, can also provide
                                  'zoo' to load the base weights associated
                                  with that recipe. Additionally, can also
                                  provide a SparseZoo model stub to load model
                                  weights from SparseZoo
  --init-lr FLOAT                 The initial learning rate to use while
                                  training, the actual initial value used will
                                  be set by the sparseml recipe  [default:
                                  1e-09]
  --gradient-accum-steps, --gradient_accum_steps INTEGER
                                  Gradient accumulation steps
  --recipe-path, --recipe_path TEXT
                                  The path to the yaml/md file containing the
                                  modifiers and schedule to apply them with.
                                  Can also provide a SparseZoo stub prefixed
                                  with 'zoo:' with an optional '?recipe_type='
                                  argument
  --eval-mode, --eval_mode / --no-eval-mode, --no_eval_mode
                                  Puts model into evaluation mode (Model
                                  weights are not updated)  [default: no-eval-
                                  mode]
  --optim [Adadelta|Adagrad|Adam|AdamW|SparseAdam|Adamax|ASGD|SGD|Rprop|RMSprop
  |LBFGS]
                                  The optimizer type to use, one of
                                  ['Adadelta', 'Adagrad', 'Adam', 'AdamW',
                                  'SparseAdam', 'Adamax', 'ASGD', 'SGD',
                                  'Rprop', 'RMSprop', 'LBFGS'].  [default:
                                  SGD]
  --optim-args, --optim_args TEXT
                                  Additional args to be passed to the
                                  optimizer; should be specified as a json
                                  object. Default args set for SGD
  --logs-dir, --logs_dir DIRECTORY
                                  The path to the directory for saving logs
                                  [default: pytorch_vision_train/tensorboard-
                                  logs]
  --save-best-after, --save_best_after INTEGER
                                  Save the best validation result after the
                                  given epoch completes until the end of
                                  training  [default: 1]
  --save-epochs, --save_epochs TEXT
                                  Epochs to save checkpoints at
  --use-mixed-precision, --use_mixed_precision
                                  Trains model using mixed precision.
                                  Supported environments are single GPU and
                                  multiple GPUs using
                                  `DistributedDataParallel` with one GPU per
                                  process
  --pretrained TEXT               The type of pretrained weights to use, loads
                                  default pretrained weights for the model if
                                  not specified or set to `True`. Otherwise,
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
  -is, --image-size, --image_size INTEGER
                                  The size of the image input to the model.
                                  Value should be equal to S for [C, S, S] or
                                  [S, S, C] dimensional input  [default: 224]
  --ffcv                          Use `ffcv` for loading data  [default:
                                  False]
  --recipe-args, --recipe_args TEXT
                                  json parsable dict of recipe variable names
                                  to values to overwrite with
  --max-train-steps, --max_train_steps INTEGER
                                  The maximum number of training steps to run
                                  per epoch. If negative, will run for the
                                  entire dataset. [default: -1]
  --max-eval-steps, --max_eval_steps INTEGER
                                  The maximum number of eval steps to run per
                                  epoch. If negative, will run for the entire
                                  dataset. [default: -1]
  --one-shot, --one_shot / --no-one-shot, --no_one_shot
                                  Apply recipe in a one-shot fashion and save
                                  the model  [default: no-one-shot]
  --help                          Show this message and exit.

#########
EXAMPLES
#########

##########
Example command for pruning resnet50 on imagenet dataset:
sparseml.image_classification.train \
    --recipe-path ~/sparseml_recipes/pruning_resnet50.yaml \
    --arch-key resnet50 --dataset imagenet --dataset-path ~/datasets/ILSVRC2012 \
    --train-batch-size 256 --test-batch-size 1024

##########
Example command for transfer learning sparse mobilenet_v1 on an image folder dataset:
sparseml.image_classification.train \
    --recipe-path  ~/sparseml_recipes/pruning_mobilenet.yaml \
    --arch-key mobilenet_v1 --pretrained pruned-moderate \
    --dataset imagefolder --dataset-path ~/datasets/my_imagefolder_dataset \
    --train-batch-size 256 --test-batch-size 1024

##########
Example command for training resnet50 on imagenette for 100 steps:
sparseml.image_classification.train \
    --train_batch_size 32 \
    --test_batch_size 32 \
    --dataset imagenette \
    --dataset-path imagenette \
    --max-train-steps 100 \
    --arch-key resnet50 \
    --recipe-path [RESNET_RECIPE] \
    --checkpoint-path zoo

##########
Template command for running training with this script on multiple GPUs using
`DistributedDataParallel` using mixed precision. Note - DDP support in this
script only tested for torch==1.7.0.
python -m torch.distributed.launch \
--nproc_per_node <NUM GPUs> \
sparseml.image_classification.train \
--use-mixed-precision \
<TRAIN.PY ARGUMENTS>
"""
import json
import os
from typing import Any, Dict, Optional, Tuple, Union

import torch

import click
from sparseml import get_main_logger
from sparseml.pytorch.image_classification.utils import (
    DEFAULT_OPTIMIZER,
    OPTIMIZERS,
    ImageClassificationTrainer,
    cli_helpers,
    helpers,
)
from sparseml.pytorch.utils import default_device, get_prunable_layers, tensor_sparsity
from sparseml.pytorch.utils.distributed import record


CURRENT_TASK = helpers.Tasks.TRAIN
LOGGER = get_main_logger()
METADATA_ARGS = [
    "arch_key",
    "dataset",
    "device",
    "pretrained",
    "test_batch_size",
    "train_batch_size",
]


@click.command()
@click.option(
    "--train-batch-size",
    "--train_batch_size",
    type=int,
    required=True,
    help="Train batch size",
)
@click.option(
    "--test-batch-size",
    "--test_batch_size",
    type=int,
    required=True,
    help="Test/Validation batch size",
)
@click.option(
    "--dataset",
    type=str,
    required=True,
    help="The dataset to use for training, "
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
    "--local_rank",
    "--local-rank",
    type=int,
    default=None,
    help="Local rank for distributed training",
    hidden=True,  # should not be modified by user
)
@click.option(
    "--checkpoint-path",
    "--checkpoint_path",
    type=str,
    default=None,
    help="A path to a previous checkpoint to load the state from "
    "and resume the state for. If provided, pretrained will "
    "be ignored . If using a SparseZoo recipe, can also "
    "provide 'zoo' to load the base weights associated with "
    "that recipe. Additionally, can also provide a SparseZoo model stub "
    "to load model weights from SparseZoo",
)
@click.option(
    "--init-lr",
    "--init_lr",
    type=float,
    default=1e-9,
    show_default=True,
    help="The initial learning rate to use while training, "
    "the actual initial value used will be set by the"
    " sparseml recipe",
)
@click.option(
    "--gradient-accum-steps",
    "--gradient_accum_steps",
    type=int,
    default=1,
    show_default=True,
    help="gradient accumulation steps",
)
@click.option(
    "--recipe-path",
    "--recipe_path",
    type=str,
    default=None,
    help="The path to the yaml/md file containing the modifiers and "
    "schedule to apply them with. Can also provide a "
    "SparseZoo stub prefixed with 'zoo:' with an optional "
    "'?recipe_type=' argument",
)
@click.option(
    "--eval-mode/--no-eval-mode",
    "--eval_mode/--no_eval_mode",
    is_flag=True,
    show_default=True,
    help="Puts model into evaluation mode (Model weights are not updated)",
)
@click.option(
    "--optim",
    type=click.Choice(OPTIMIZERS, case_sensitive=True),
    default=DEFAULT_OPTIMIZER,
    show_default=True,
    help=f"The optimizer type to use, one of {OPTIMIZERS}.",
)
@click.option(
    "--optim-args",
    "--optim_args",
    default=json.dumps(
        {
            "momentum": 0.9,
            "nesterov": True,
            "weight_decay": 0.0001,
        }
    ),
    type=str,
    callback=cli_helpers.parse_json_callback,
    help="Additional args to be passed to the optimizer; "
    "should be specified as a json object. "
    f"Default args set for {DEFAULT_OPTIMIZER}",
)
@click.option(
    "--logs-dir",
    "--logs_dir",
    type=click.Path(dir_okay=True, file_okay=False),
    default=os.path.join("pytorch_vision_train", "tensorboard-logs"),
    callback=cli_helpers.create_dir_callback,
    show_default=True,
    help="The path to the directory for saving logs",
)
@click.option(
    "--save-best-after",
    "--save_best_after",
    type=int,
    default=1,
    show_default=True,
    help="Save the best validation result after the given "
    "epoch completes until the end of training",
)
@click.option(
    "--save-epochs",
    "--save_epochs",
    cls=cli_helpers.OptionEatAllArguments,
    callback=cli_helpers.parse_into_tuple_of_ints,
    help="Epochs to save checkpoints at",
)
@click.option(
    "--use-mixed-precision",
    "--use_mixed_precision",
    is_flag=True,
    help="Trains model using mixed precision. Supported "
    "environments are single GPU and multiple GPUs using "
    "`DistributedDataParallel` with one GPU per process",
)
@click.option(
    "--pretrained",
    type=str,
    default=True,
    show_default=True,
    help="The type of pretrained weights to use, "
    "loads default pretrained weights for "
    "the model if not specified or set to `True`. "
    "Otherwise, should be set to the desired weights "
    "type: [base, optim, optim-perf]. To not load any weights set"
    " to one of [none, false]",
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
    "--image-size",
    "--image_size",
    type=int,
    default=224,
    show_default=True,
    help="The size of the image input to the model. Value should be "
    "equal to S for [C, S, S] or [S, S, C] dimensional input",
)
@click.option(
    "--ffcv",
    is_flag=True,
    show_default=True,
    help="Use `ffcv` for loading data",
)
@click.option(
    "--recipe-args",
    "--recipe_args",
    type=str,
    default=None,
    help="json parsable dict of recipe variable names to values to overwrite with",
)
@click.option(
    "--max-train-steps",
    "--max_train_steps",
    default=-1,
    type=int,
    show_default=True,
    help="The maximum number of training steps to run per epoch. If negative, "
    "will run for the entire dataset",
)
@click.option(
    "--max-eval-steps",
    "--max_eval_steps",
    default=-1,
    type=int,
    show_default=True,
    help="The maximum number of eval steps to run per epoch. If negative, "
    "will run for the entire dataset",
)
@click.option(
    "--one-shot/--no-one-shot",
    "--one_shot/--no_one_shot",
    default=False,
    is_flag=True,
    show_default=True,
    help="Apply recipe in a one-shot fashion and save the model",
)
@record
def main(
    train_batch_size: int,
    test_batch_size: int,
    dataset: str,
    dataset_path: str,
    arch_key: Optional[str] = None,
    local_rank: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    init_lr: float = 1e-9,
    gradient_accum_steps: int = 1,
    recipe_path: Optional[str] = None,
    eval_mode: bool = False,
    optim: str = DEFAULT_OPTIMIZER,
    optim_args: Dict[str, Any] = {
        "momentum": 0.9,
        "nesterov": True,
        "weight_decay": 0.0001,
    },
    logs_dir: str = os.path.join("pytorch_vision_train", "tensorboard-logs"),
    save_best_after: int = 1,
    save_epochs: Tuple[int, ...] = (5, 10),
    use_mixed_precision: bool = False,
    pretrained: Union[str, bool] = True,
    pretrained_dataset: Optional[str] = None,
    model_kwargs: Dict[str, Any] = {},
    dataset_kwargs: Dict[str, Any] = {},
    model_tag: Optional[str] = None,
    save_dir: str = "pytorch_vision",
    device: Optional[str] = default_device(),
    loader_num_workers: int = 4,
    loader_pin_memory: bool = True,
    image_size: int = 224,
    ffcv: bool = False,
    recipe_args: Optional[str] = None,
    max_train_steps: int = -1,
    max_eval_steps: int = -1,
    one_shot: bool = False,
):
    """
    PyTorch training integration with SparseML for image classification models
    """
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", -1))
    # support for legacy torch.distributed.launch and new torch.distributed.run
    local_rank = local_rank or int(os.environ.get("LOCAL_RANK", -1))

    # training requires recipe path
    if not eval_mode and recipe_path is None:
        raise ValueError("Must include --recipe-path when not running in eval mode")

    if eval_mode and "recipe_type=transfer" in checkpoint_path:
        checkpoint_path = checkpoint_path.replace(
            "recipe_type=transfer", "recipe_type=original"
        )

    # non DDP execution or 0th DDP process
    is_main_process = rank in (-1, 0)

    if not train_batch_size % world_size == 0:
        raise ValueError(
            f"Invalid training batch size for world size {world_size} "
            f"given batch size {train_batch_size}. "
            "world size must divide training batch size evenly."
        )

    train_batch_size = train_batch_size // world_size
    helpers.set_seeds(local_rank=local_rank)

    if not eval_mode:
        train_dataset, train_loader, = helpers.get_dataset_and_dataloader(
            dataset_name=dataset,
            dataset_path=dataset_path,
            batch_size=train_batch_size,
            image_size=image_size,
            dataset_kwargs=dataset_kwargs,
            training=True,
            loader_num_workers=loader_num_workers,
            loader_pin_memory=loader_pin_memory,
            ffcv=ffcv,
            device=device,
            rank=rank,
        )
    else:
        train_dataset = None
        train_loader = None

    val_dataset, val_loader = (
        helpers.get_dataset_and_dataloader(
            dataset_name=dataset,
            dataset_path=dataset_path,
            batch_size=test_batch_size,
            image_size=image_size,
            dataset_kwargs=dataset_kwargs,
            training=False,
            loader_num_workers=loader_num_workers,
            loader_pin_memory=loader_pin_memory,
            ffcv=ffcv,
            device=device,
        )
        if is_main_process
        else (None, None)
    )

    num_classes = helpers.infer_num_classes(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        dataset=dataset,
        model_kwargs=model_kwargs,
    )

    model, arch_key, checkpoint_path = helpers.create_model(
        checkpoint_path=checkpoint_path,
        recipe_path=recipe_path,
        num_classes=num_classes,
        arch_key=arch_key,
        pretrained=pretrained,
        pretrained_dataset=pretrained_dataset,
        local_rank=local_rank,
        **model_kwargs,
    )

    save_dir, loggers = helpers.get_save_dir_and_loggers(
        task=CURRENT_TASK,
        is_main_process=is_main_process,
        save_dir=save_dir,
        arch_key=arch_key,
        model_tag=model_tag,
        dataset_name=dataset,
        logs_dir=logs_dir,
    )

    LOGGER.info(f"created model with key {arch_key}: {model}")

    ddp, device, model = helpers.ddp_aware_model_move(
        device=device,
        local_rank=local_rank,
        model=model,
        rank=rank,
    )

    metadata = helpers.extract_metadata(
        metadata_args=METADATA_ARGS,
        training_args_dict=locals(),
    )

    LOGGER.info(f"running on device {device}")

    trainer = ImageClassificationTrainer(
        model=model,
        key=arch_key,
        recipe_path=recipe_path,
        checkpoint_path=checkpoint_path,
        metadata=metadata,
        ddp=ddp,
        device=device,
        use_mixed_precision=use_mixed_precision,
        val_loader=val_loader,
        train_loader=train_loader,
        is_main_process=is_main_process,
        loggers=loggers,
        loss_fn=helpers.get_loss_wrapper,
        init_lr=init_lr,
        optim_name=optim,
        optim_kwargs=optim_args,
        recipe_args=recipe_args,
        max_train_steps=max_train_steps,
        one_shot=one_shot,
        gradient_accum_steps=gradient_accum_steps,
    )

    train(
        trainer=trainer,
        save_dir=save_dir,
        max_eval_steps=max_eval_steps,
        eval_mode=eval_mode,
        is_main_process=is_main_process,
        save_best_after=save_best_after,
        save_epochs=save_epochs,
        rank=rank,
    )


def train(
    trainer: ImageClassificationTrainer,
    save_dir: str,
    max_eval_steps: int,
    eval_mode: bool,
    is_main_process: bool,
    save_best_after: int,
    save_epochs: Tuple[int, ...],
    rank: int,
):
    """
    Utility function to run the training loop

    :param trainer: The ImageClassificationTrainer object
    :param save_dir: The directory to save checkpoints to
    :param max_eval_steps: The number of steps to run for validation
    :param eval_mode: Whether to run in evaluation mode
    :param is_main_process: Whether this is the main process
    :param save_best_after: The number of epochs to wait before saving
        a new best model
    :param save_epochs: The epochs to save checkpoints for
    :param rank: The rank of the process
    """

    val_res = None
    if not trainer.one_shot:
        # Baseline eval run
        val_res = trainer.run_one_epoch(
            mode="val",
            max_steps=max_eval_steps,
            baseline_run=True,
        )

        LOGGER.info(f"\nInitial validation results: {val_res}")

        if eval_mode:
            eval_results_path = os.path.join(save_dir, "eval.txt")
            helpers.write_validation_results(eval_results_path, val_res)

    if not (eval_mode or trainer.one_shot):
        LOGGER.info(f"Starting training from epoch {trainer.epoch}")

        val_metric = best_metric = None

        while trainer.epoch < trainer.max_epochs:
            train_res = trainer.run_one_epoch(
                mode="train",
                max_steps=trainer.max_train_steps,
            )
            LOGGER.info(f"\nEpoch {trainer.epoch} training results: {train_res}")
            # testing steps
            if is_main_process:
                val_res = trainer.run_one_epoch(
                    mode="val",
                    max_steps=max_eval_steps,
                )
                val_metric = val_res.result_mean(trainer.target_metric).item()

                should_save_epoch = trainer.epoch >= save_best_after and (
                    best_metric is None
                    or (
                        val_metric <= best_metric
                        if trainer.target_metric != "top1acc"
                        else val_metric >= best_metric
                    )
                )
                if should_save_epoch:
                    helpers.save_model_training(
                        model=trainer.model,
                        optim=trainer.optim,
                        manager=trainer.manager,
                        checkpoint_manager=trainer.checkpoint_manager,
                        save_name="checkpoint-best",
                        save_dir=save_dir,
                        epoch=trainer.epoch,
                        val_res=val_res,
                        arch_key=trainer.key,
                    )
                    # Best metric is based on validation results
                    best_metric = val_metric

            # save checkpoints
            should_save_epoch = (
                is_main_process and save_epochs and trainer.epoch in save_epochs
            )
            if should_save_epoch:
                save_name = (
                    f"checkpoint-{trainer.epoch:04d}-{val_metric:.04f}"
                    if val_metric
                    else f"checkpoint-{trainer.epoch:04d}"
                )
                helpers.save_model_training(
                    model=trainer.model,
                    optim=trainer.optim,
                    manager=trainer.manager,
                    checkpoint_manager=trainer.checkpoint_manager,
                    save_name=save_name,
                    save_dir=save_dir,
                    epoch=trainer.epoch,
                    val_res=val_res,
                    arch_key=trainer.key,
                )

            trainer.epoch += 1

        # export the final model
    LOGGER.info("completed...")
    if is_main_process and not eval_mode:
        # Convert QAT -> quantized ONNX graph for finalized model only
        save_name = "model" if not trainer.one_shot else "model-one-shot"
        helpers.save_model_training(
            model=trainer.model,
            optim=trainer.optim,
            manager=trainer.manager,
            checkpoint_manager=trainer.checkpoint_manager,
            save_name=save_name,
            save_dir=save_dir,
            epoch=trainer.epoch - 1 if not trainer.one_shot else None,
            val_res=val_res,
        )

        LOGGER.info("layer sparsities:")
        for (name, layer) in get_prunable_layers(trainer.model):
            LOGGER.info(f"{name}.weight: {tensor_sparsity(layer.weight).item():.4f}")

    # close DDP
    if rank != -1:
        assert hasattr(torch, "distributed")
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
