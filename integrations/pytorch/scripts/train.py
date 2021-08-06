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
Train and/or prune an image classification model on a dataset
"""
import argparse

from helpers import *
from sparseml.pytorch.models import ModelRegistry
from sparseml.pytorch.utils import (
    ModuleDeviceContext,
    ModuleTester,
    ModuleTrainer,
    get_prunable_layers,
    model_to_device,
    tensor_sparsity,
)


CURRENT_TASK = "train"


def parse_args():
    """
    Utility method to parse command line arguments for Training specific tasks
    """
    parser = argparse.ArgumentParser(description=__doc__)
    # DDP argument, necessary for launching via torch.distributed
    add_local_rank(parser)
    add_universal_args(parser, task=CURRENT_TASK)

    add_device_args(parser)
    add_workers_args(parser)
    add_pin_memory_args(parser)

    add_learning_rate(parser, task=CURRENT_TASK)
    add_optimizer_args(parser, task=CURRENT_TASK)
    add_training_specific_args(parser)

    args = parser.parse_args()
    args = parse_ddp_args(args, task=CURRENT_TASK)

    append_preprocessing_args(args)

    return args


def train(args, model, train_loader, val_loader, input_shape, save_dir, loggers):
    # loss setup
    val_loss = get_loss_wrapper(args, training=True)
    LOGGER.info("created loss for validation: {}".format(val_loss))

    train_loss = get_loss_wrapper(args, training=True)
    LOGGER.info("created loss for training: {}".format(train_loss))

    # training setup
    if not args.eval_mode:
        epoch, optim, manager = create_scheduled_optimizer(
            args,
            model,
            train_loader,
            loggers,
        )
    else:
        epoch = 0
        train_loss = None
        optim = None
        manager = None

    # device setup
    if args.rank == -1:
        device = args.device
        ddp = False
    else:
        torch.cuda.set_device(args.local_rank)
        device = args.local_rank
        ddp = True

    model, device, device_ids = model_to_device(model, device, ddp=ddp)
    LOGGER.info("running on device {} for ids {}".format(device, device_ids))

    trainer = (
        ModuleTrainer(
            model,
            device,
            train_loss,
            optim,
            loggers=loggers,
            device_context=ModuleDeviceContext(
                use_mixed_precision=args.use_mixed_precision, world_size=args.world_size
            ),
        )
        if not args.eval_mode
        else None
    )

    if args.is_main_process:  # only test on one DDP process if using DDP
        tester = ModuleTester(model, device, val_loss, loggers=loggers, log_steps=-1)

        # initial baseline eval run
        tester.run_epoch(val_loader, epoch=epoch - 1, max_steps=args.debug_steps)

    if not args.eval_mode:
        save_recipe(recipe_manager=manager, save_dir=save_dir)
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

            if args.rank != -1:  # sync DDP dataloaders
                train_loader.sampler.set_epoch(epoch)

            trainer.run_epoch(
                train_loader,
                epoch,
                max_steps=args.debug_steps,
                show_progress=args.is_main_process,
            )

            # testing steps
            if args.is_main_process:  # only test and save on main process
                val_res = tester.run_epoch(
                    val_loader, epoch, max_steps=args.debug_steps
                )
                val_loss = val_res.result_mean(DEFAULT_LOSS_KEY).item()

                if epoch >= args.save_best_after and (
                    best_loss is None or val_loss <= best_loss
                ):
                    save_model_training(
                        model,
                        optim,
                        input_shape,
                        "checkpoint-best",
                        save_dir,
                        epoch,
                        val_res,
                    )
                    best_loss = val_loss

            # save checkpoints
            if args.is_main_process and args.save_epochs and epoch in args.save_epochs:
                save_model_training(
                    model,
                    optim,
                    input_shape,
                    "checkpoint-{:04d}-{:.04f}".format(epoch, val_loss),
                    save_dir,
                    epoch,
                    val_res,
                )

            epoch += 1

        # export the final model
        LOGGER.info("completed...")
        if args.is_main_process:
            # only convert qat -> quantized ONNX graph for finalized model
            # TODO: change this to all checkpoints when conversion times improve
            save_model_training(
                model, optim, input_shape, "model", save_dir, epoch - 1, val_res, True
            )

            LOGGER.info("layer sparsities:")
            for (name, layer) in get_prunable_layers(model):
                LOGGER.info(
                    "{}.weight: {:.4f}".format(
                        name, tensor_sparsity(layer.weight).item()
                    )
                )

    # close DDP
    if args.rank != -1:
        torch.distributed.destroy_process_group()


def main():
    """
    Driver function for the script
    """
    args_ = parse_args()
    distributed_setup(args_.local_rank)
    save_dir, loggers = get_save_dir_and_loggers(args_, task=CURRENT_TASK)

    input_shape = ModelRegistry.input_shape(args_.arch_key)
    image_size = input_shape[1]  # assume shape [C, S, S] where S is the image size

    (
        train_dataset,
        train_loader,
        val_dataset,
        val_loader,
    ) = get_train_and_validation_loaders(args_, image_size, task=CURRENT_TASK)

    # model creation
    num_classes = infer_num_classes(args_, train_dataset, val_dataset)

    model = create_model(args_, num_classes)

    train(args_, model, train_loader, val_loader, input_shape, save_dir, loggers)


if __name__ == "__main__":
    main()
