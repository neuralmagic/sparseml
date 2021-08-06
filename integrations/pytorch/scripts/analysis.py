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
Utility script to Run a kernel sparsity (pruning) analysis for a desired image
classification architecture
"""
from helpers import *
from sparseml.pytorch.optim import (
    default_exponential_check_lrs,
    lr_loss_sensitivity,
    pruning_loss_sens_magnitude,
    pruning_loss_sens_one_shot,
)
from sparseml.pytorch.utils import model_to_device


CURRENT_TASK = "analysis"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    # DDP argument, necessary for launching via torch.distributed
    add_local_rank(parser)

    subparsers = parser.add_subparsers(dest="command")
    pruning_sensitivity_parser = subparsers.add_parser(
        "pr_sensitivity",
        description="Run a kernel sparsity (pruning) analysis for a given model",
    )
    lr_sensitivity_parser = subparsers.add_parser(
        "lr_sensitivity",
        description="Run a learning rate sensitivity analysis for a desired image "
        "classification or detection architecture",
    )

    subtasks = [
        (pruning_sensitivity_parser, "pr_sensitivity"),
        (lr_sensitivity_parser, "lr_sensitivity"),
    ]

    for _parser, subtask in subtasks:
        add_universal_args(parser=_parser, task=None)
        add_device_args(parser=_parser)
        add_workers_args(parser=_parser)
        add_pin_memory_args(parser=_parser)

        if _parser == lr_sensitivity_parser:
            add_learning_rate(parser=_parser, task=subtask)
            add_optimizer_args(parser=_parser, task=subtask)
            add_lr_sensitivity_specific_args(parser=_parser)

        add_steps_per_measurement(parser=_parser, task=subtask)
        add_batch_size_arg(parser=_parser, task=subtask)

        if _parser == pruning_sensitivity_parser:
            add_pruning_specific_args(parser=_parser)

    args = parser.parse_args()
    args = parse_ddp_args(args, task=CURRENT_TASK)
    append_preprocessing_args(args)

    return args


def pruning_loss_sensitivity(args, model, train_loader, save_dir, loggers):
    # loss setup
    if not args.approximate:
        loss = get_loss_wrapper(args)
        LOGGER.info("created loss: {}".format(loss))
    else:
        loss = None

    # device setup
    if not args.approximate:
        module, device, device_ids = model_to_device(model, args.device)
    else:
        device = None

    # kernel sparsity analysis
    if args.approximate:
        analysis = pruning_loss_sens_magnitude(model)
    else:
        analysis = pruning_loss_sens_one_shot(
            model,
            train_loader,
            loss,
            device,
            args.steps_per_measurement,
            tester_loggers=loggers,
        )

    # saving and printing results
    LOGGER.info("completed...")
    LOGGER.info("Saving results in {}".format(save_dir))
    analysis.save_json(
        os.path.join(
            save_dir,
            "ks_approx_sensitivity.json"
            if args.approximate
            else "ks_one_shot_sensitivity.json",
        )
    )
    analysis.plot(
        os.path.join(
            save_dir,
            os.path.join(
                save_dir,
                "ks_approx_sensitivity.png"
                if args.approximate
                else "ks_one_shot_sensitivity.png",
            ),
        ),
        plot_integral=True,
    )
    analysis.print_res()


def lr_sensitivity(args, model, train_loader, save_dir, loggers):
    # optimizer setup
    optim = SGD(model.parameters(), lr=args.init_lr, **args.optim_args)
    LOGGER.info("created optimizer: {}".format(optim))

    # loss setup
    loss = get_loss_wrapper(args)
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

    num_classes = infer_num_classes(args_, train_dataset, val_dataset)

    model = create_model(args_, num_classes)

    if args_.command == "lr_sensitivity":
        lr_sensitivity(args_, model, train_loader, save_dir, loggers)
    else:
        pruning_loss_sensitivity(args_, model, train_loader, save_dir, loggers)


if __name__ == "__main__":
    main()
