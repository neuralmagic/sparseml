"""
Script to create either a table for pruning or a config.yaml to work with the recal
setup in neuralmagicML. Takes in an onnx model file as well as performance
and loss pruning sensitivity analysis. See model_pruning_loss_sensitivity.py
to create the necessary json files for the loss and model_pruning_perf_sensitivity.py
to create the necessary json files for the performance.

Multiple sensitivity files can be supplied for both loss and performance.
When multiple have been supplied, it takes the average across all of them
to figure out what buckets to put into.

When both loss and perf are supplied it does a balanced analysis between the two
for figuring out which layers should be applied at what sparsity for the config files.
The mapping for this is defined in the following example when 3 buckets are used:

|              Perf Bot 5   Perf Low   Perf Med   Perf High
| Loss Top 5   -1           -1         -1         -1
| Loss High    -1           0          0          1
| Loss Med     -1           0          1          2
| Loss Low     -1           1          2          2


In the table output, the important features will be:
- ONNX node info which should be apparent from the titles
- Loss buckets and scores: a lower bucket / higher score means larger effect on loss
- Perf buckets and scores: a lower bucket / higher score means smaller effect on perf
- Note that buckets with -1 are the ones that have the most negative effect on model


##########
Command help:
usage: model_pruning_config.py [-h] {table,pytorch,tensorflow} ...

Create configs or info tables for pruning

positional arguments:
  {table,pytorch,tensorflow}

optional arguments:
  -h, --help            show this help message and exit


##########
table command help:
usage: model_pruning_config.py table [-h] --onnx-file-path ONNX_FILE_PATH
                                     [--loss-sensitivities LOSS_SENSITIVITIES [LOSS_SENSITIVITIES ...]]
                                     [--perf-sensitivities PERF_SENSITIVITIES [PERF_SENSITIVITIES ...]]
                                     [--json-path JSON_PATH]
                                     [--csv-path CSV_PATH]

create a sensitivity and info able for pruning and export as csv

optional arguments:
  -h, --help            show this help message and exit
  --onnx-file-path ONNX_FILE_PATH
                        Path to the local onnx file to analyze
  --loss-sensitivities LOSS_SENSITIVITIES [LOSS_SENSITIVITIES ...]
                        Paths to loss sensitivity files to include in the
                        analysis and output (generated from
                        model_pruning_loss_sensitivity.py script)
  --perf-sensitivities PERF_SENSITIVITIES [PERF_SENSITIVITIES ...]
                        Paths to performance sensitivity files to include in
                        the analysis and output (generated from
                        model_pruning_perf_sensitivity.py script)
  --json-path JSON_PATH
                        Path to save the output json file to, defaults to save
                        next to the onnx-file-path
  --csv-path CSV_PATH   Path to save the output csv file to, defaults to save
                        next to the onnx-file-path


##########
pytorch command help
usage: model_pruning_config.py pytorch [-h] --onnx-file-path ONNX_FILE_PATH
                                       [--loss-sensitivities LOSS_SENSITIVITIES [LOSS_SENSITIVITIES ...]]
                                       [--perf-sensitivities PERF_SENSITIVITIES [PERF_SENSITIVITIES ...]]
                                       --train-epochs TRAIN_EPOCHS
                                       [--pruning-buckets PRUNING_BUCKETS [PRUNING_BUCKETS ...]]
                                       [--ignore-percent IGNORE_PERCENT [IGNORE_PERCENT ...]]
                                       [--init-lr INIT_LR]
                                       [--final-lr FINAL_LR]
                                       [--yaml-path YAML_PATH]

create a default PyTorch config file for pruning a model

optional arguments:
  -h, --help            show this help message and exit
  --onnx-file-path ONNX_FILE_PATH
                        Path to the local onnx file to analyze
  --loss-sensitivities LOSS_SENSITIVITIES [LOSS_SENSITIVITIES ...]
                        Paths to loss sensitivity files to include in the
                        analysis and output (generated from
                        model_pruning_loss_sensitivity.py script)
  --perf-sensitivities PERF_SENSITIVITIES [PERF_SENSITIVITIES ...]
                        Paths to performance sensitivity files to include in
                        the analysis and output (generated from
                        model_pruning_perf_sensitivity.py script)
  --train-epochs TRAIN_EPOCHS
                        The number of epochs originally used to train the
                        model. Handles setting up the stabilization, pruning,
                        and fine-tuning periods
  --pruning-buckets PRUNING_BUCKETS [PRUNING_BUCKETS ...]
                        The layer buckets and associated sparsity levels to
                        create the config for
  --ignore-percent IGNORE_PERCENT [IGNORE_PERCENT ...]
                        The percentage (as a decimal) of layers to not prune
                        in the model, takes the layers that affect the loss
                        the most and / or the performance the least
  --init-lr INIT_LR     The initial learning rate originally used to train the
                        model. If supplied will add a SetLearningRateModifier
                        at the beginning
  --final-lr FINAL_LR   The final learning rate originally used to train the
                        model, applicable for SGD learning rate schedule
                        training types. If supplied will add a
                        LearningRateModifier for the fine-tuning phase
  --yaml-path YAML_PATH
                        Path to save the output config.yaml file to, defaults
                        to save next to the onnx-file-path


##########
tensorflow command help
usage: model_pruning_config.py tensorflow [-h] --onnx-file-path ONNX_FILE_PATH
                                          [--loss-sensitivities LOSS_SENSITIVITIES [LOSS_SENSITIVITIES ...]]
                                          [--perf-sensitivities PERF_SENSITIVITIES [PERF_SENSITIVITIES ...]]
                                          --train-epochs TRAIN_EPOCHS
                                          [--pruning-buckets PRUNING_BUCKETS [PRUNING_BUCKETS ...]]
                                          [--ignore-percent IGNORE_PERCENT [IGNORE_PERCENT ...]]
                                          [--init-lr INIT_LR]
                                          [--final-lr FINAL_LR]
                                          [--yaml-path YAML_PATH]

create a default TensorFlow config file for pruning a model

optional arguments:
  -h, --help            show this help message and exit
  --onnx-file-path ONNX_FILE_PATH
                        Path to the local onnx file to analyze
  --loss-sensitivities LOSS_SENSITIVITIES [LOSS_SENSITIVITIES ...]
                        Paths to loss sensitivity files to include in the
                        analysis and output (generated from
                        model_pruning_loss_sensitivity.py script)
  --perf-sensitivities PERF_SENSITIVITIES [PERF_SENSITIVITIES ...]
                        Paths to performance sensitivity files to include in
                        the analysis and output (generated from
                        model_pruning_perf_sensitivity.py script)
  --train-epochs TRAIN_EPOCHS
                        The number of epochs originally used to train the
                        model. Handles setting up the stabilization, pruning,
                        and fine-tuning periods
  --pruning-buckets PRUNING_BUCKETS [PRUNING_BUCKETS ...]
                        The layer buckets and associated sparsity levels to
                        create the config for
  --ignore-percent IGNORE_PERCENT [IGNORE_PERCENT ...]
                        The percentage (as a decimal) of layers to not prune
                        in the model, takes the layers that affect the loss
                        the most and / or the performance the least
  --init-lr INIT_LR     The initial learning rate originally used to train the
                        model. If supplied will add a SetLearningRateModifier
                        at the beginning
  --final-lr FINAL_LR   The final learning rate originally used to train the
                        model, applicable for SGD learning rate schedule
                        training types. If supplied will add a
                        LearningRateModifier for the fine-tuning phase
  --yaml-path YAML_PATH
                        Path to save the output config.yaml file to, defaults
                        to save next to the onnx-file-path


##########
table example for ResNet 50:
python scripts/onnx/model_pruning_config.py table \
    --onnx-file-path resnet-v1_50_imagenet_base/model.onnx \
    --loss-sensitivities resnet-v1_50_imagenet_base/model.onnx.loss_approx-sensitivity.json \
    --perf-sensitivities resnet-v1_50_imagenet_base/model.onnx.perf-sensitivity_bs-1_c-10.json \
    resnet-v1_50_imagenet_base/model.onnx.perf-sensitivity_bs-64_c-10.json


##########
pytorch example for ResNet 50:
python scripts/onnx/model_pruning_config.py pytorch \
    --onnx-file-path resnet-v1_50_imagenet_base/model.onnx \
    --loss-sensitivities resnet-v1_50_imagenet_base/model.onnx.loss_approx-sensitivity.json \
    --perf-sensitivities resnet-v1_50_imagenet_base/model.onnx.perf-sensitivity_bs-1_c--1.json \
    resnet-v1_50_imagenet_base/model.onnx.perf-sensitivity_bs-64_c--1.json \
    --train-epochs 90  \
    --init-lr 0.1 \
    --final-lr 0.001


##########
tensorflow example for ResNet 50:
python scripts/onnx/model_pruning_config.py pytorch \
    --onnx-file-path resnet-v1_50_imagenet_base/model.onnx \
    --loss-sensitivities resnet-v1_50_imagenet_base/model.onnx.loss_approx-sensitivity.json \
    --perf-sensitivities resnet-v1_50_imagenet_base/model.onnx.perf-sensitivity_bs-1_c--1.json \
    resnet-v1_50_imagenet_base/model.onnx.perf-sensitivity_bs-64_c--1.json \
    --train-epochs 90  \
    --init-lr 0.1 \
    --final-lr 0.001
"""

import argparse
from typing import List, Dict, Tuple, Callable
import math

from neuralmagicML import get_main_logger
from neuralmagicML.onnx.recal import SensitivityModelInfo, optimized_balanced_buckets
from neuralmagicML.onnx.utils import check_load_model, get_node_by_id, get_node_params

from neuralmagicML.pytorch.recal import (
    ScheduledModifierManager as PTScheduledModifierManager,
    EpochRangeModifier as PTEpochRangeModifier,
    GradualKSModifier as PTGradualKSModifier,
    SetLearningRateModifier as PTSetLearningRateModifier,
    LearningRateModifier as PTLearningRateModifier,
)
from neuralmagicML.tensorflow.recal import (
    ScheduledModifierManager as TFScheduledModifierManager,
    EpochRangeModifier as TFEpochRangeModifier,
    GradualKSModifier as TFGradualKSModifier,
    SetLearningRateModifier as TFSetLearningRateModifier,
    LearningRateModifier as TFLearningRateModifier,
)


LOGGER = get_main_logger()
TABLE_COMMAND = "table"
PYTORCH_COMMAND = "pytorch"
TENSORFLOW_COMMAND = "tensorflow"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create configs or info tables for pruning"
    )

    subparsers = parser.add_subparsers(dest="command")

    table_parser = subparsers.add_parser(
        TABLE_COMMAND,
        description="create a sensitivity and info able for pruning and export as csv",
    )
    pytorch_parser = subparsers.add_parser(
        PYTORCH_COMMAND,
        description="create a default PyTorch config file for pruning a model",
    )
    tensorflow_parser = subparsers.add_parser(
        TENSORFLOW_COMMAND,
        description="create a default TensorFlow config file for pruning a model",
    )

    for index, par in enumerate([table_parser, pytorch_parser, tensorflow_parser]):
        par.add_argument(
            "--onnx-file-path",
            type=str,
            required=True,
            help="Path to the local onnx file to analyze",
        )
        par.add_argument(
            "--loss-sensitivities",
            type=str,
            default=[],
            nargs="+",
            help="Paths to loss sensitivity files to include in the analysis and output"
            " (generated from model_pruning_loss_sensitivity.py script)",
        )
        par.add_argument(
            "--perf-sensitivities",
            type=str,
            default=[],
            nargs="+",
            help="Paths to performance sensitivity files to include in the analysis "
            "and output (generated from model_pruning_perf_sensitivity.py script)",
        )

        if index == 0:
            par.add_argument(
                "--json-path",
                type=str,
                default=None,
                help="Path to save the output json file to, "
                "defaults to save next to the onnx-file-path",
            )
            par.add_argument(
                "--csv-path",
                type=str,
                default=None,
                help="Path to save the output csv file to, "
                "defaults to save next to the onnx-file-path",
            )
        else:
            par.add_argument(
                "--train-epochs",
                type=float,
                required=True,
                help="The number of epochs originally used to train the model. "
                "Handles setting up the stabilization, pruning, and fine-tuning "
                "periods",
            )
            par.add_argument(
                "--pruning-buckets",
                type=float,
                default=[0.75, 0.825, 0.9],
                nargs="+",
                help="The layer buckets and associated sparsity levels to create the "
                "config for",
            )
            par.add_argument(
                "--ignore-percent",
                type=float,
                default=0.05,
                help="The percentage (as a decimal) of layers to not prune in the model"
                ", takes the layers that affect the loss the most and / or the "
                "performance the least",
            )
            par.add_argument(
                "--init-lr",
                type=float,
                default=None,
                help="The initial learning rate originally used to train the model. "
                "If supplied will add a SetLearningRateModifier at the beginning",
            )
            par.add_argument(
                "--final-lr",
                type=float,
                default=None,
                help="The final learning rate originally used to train the model, "
                "applicable for SGD learning rate schedule training types. "
                "If supplied will add a LearningRateModifier for the fine-tuning phase",
            )
            par.add_argument(
                "--yaml-path",
                type=str,
                default=None,
                help="Path to save the output config.yaml file to, "
                "defaults to save next to the onnx-file-path",
            )

    return parser.parse_args()


def _table(args):
    LOGGER.info("creating model sensitivity info...")
    model_info = SensitivityModelInfo.from_sensitivities(
        model=args.onnx_file_path,
        perf_analysis=args.perf_sensitivities,
        loss_analysis=args.loss_sensitivities,
    )

    json_path = (
        args.json_path
        if args.json_path is not None
        else "{}.sensitivity_info.json".format(args.onnx_file_path)
    )
    LOGGER.info("saving json to {}".format(json_path))
    model_info.save_json(json_path)
    LOGGER.info("saved json to {}".format(json_path))

    csv_path = (
        args.csv_path
        if args.csv_path is not None
        else "{}.sensitivity_info.csv".format(args.onnx_file_path)
    )
    LOGGER.info("saving csv to {}".format(csv_path))
    model_info.save_csv(csv_path)
    LOGGER.info("saved csv to {}".format(csv_path))


def _config(
    args,
    manager_const: Callable,
    epoch_const: Callable,
    set_lr_const: Callable,
    lr_const: Callable,
    ks_const: Callable,
):
    stabilization_epochs = 1
    pruning_epochs = math.ceil(args.train_epochs / 3.0)  # set to 1/3 original train
    fine_tuning_epochs = math.ceil(args.train_epochs / 3.0)  # set to 1/3 original train

    LOGGER.info(
        "adding EpochRangeModifier from {} to {}".format(
            0, stabilization_epochs + pruning_epochs + fine_tuning_epochs
        )
    )
    modifiers = [
        epoch_const(
            start_epoch=0.0,
            end_epoch=stabilization_epochs + pruning_epochs + fine_tuning_epochs,
        )
    ]

    # add in learning rate modifiers
    if args.init_lr:
        assert args.init_lr > 0.0
        pruning_lr = (
            args.init_lr if not args.final_lr else (args.init_lr + args.final_lr) / 2.0
        )
        LOGGER.info(
            "adding SetLearningRateModifier at epoch {} for LR {}".format(0, pruning_lr)
        )
        modifiers.append(set_lr_const(learning_rate=pruning_lr, start_epoch=0.0))

        if args.final_lr:
            assert args.final_lr < args.init_lr
            assert args.final_lr > 0.0
            start_epoch = stabilization_epochs + pruning_epochs + 1
            gamma = 0.25
            init_lr = pruning_lr * gamma
            target_final_lr = args.final_lr * 0.1
            num_steps = math.log(target_final_lr / init_lr) / math.log(
                gamma
            )  # final_lr = init_lr * gamma ^ n : solve for n
            step_size = math.floor((fine_tuning_epochs - 1.0) / num_steps)

            LOGGER.info(
                (
                    "adding StepLR LearningRateModifier at epoch {} with "
                    "init_lr {}, step_size {}, gamma {}"
                ).format(start_epoch, init_lr, step_size, gamma)
            )
            modifiers.append(
                lr_const(
                    lr_class="StepLR",
                    lr_kwargs={"step_size": step_size, "gamma": gamma},
                    start_epoch=stabilization_epochs + pruning_epochs + 1,
                    init_lr=init_lr,
                )
            )

    LOGGER.info(
        (
            "getting optimized_balanced_buckets for prunable layers in model at {} "
            "with perf sensitivities {} and loss sensitivities {}"
        ).format(args.onnx_file_path, args.perf_sensitivities, args.loss_sensitivities)
    )
    optimized_buckets = optimized_balanced_buckets(
        model=args.onnx_file_path,
        perf_analysis=args.perf_sensitivities,
        loss_analysis=args.loss_sensitivities,
        num_buckets=len(args.pruning_buckets),
        edge_percent=args.ignore_percent,
    )
    bucketed_weights = {}  # type: Dict[int, List[str]]
    model = check_load_model(args.onnx_file_path)

    for node_id, bucket_info in optimized_buckets.items():
        node = get_node_by_id(model, node_id)
        weight_info, _ = get_node_params(model, node)
        bucket = bucket_info[0]

        if bucket not in bucketed_weights:
            bucketed_weights[bucket] = []

        bucketed_weights[bucket].append(weight_info.name)

    bucketed_weights = [
        (key, val) for key, val in bucketed_weights.items()
    ]  # type: List[Tuple[int, List[str]]]
    bucketed_weights.sort(key=lambda v: v[0])
    index_offset = (
        0 if len(bucketed_weights) == len(args.pruning_buckets) else -1
    )  # offset case for if top percent were cut out from pruning or not

    for index, (bucket, weights) in enumerate(bucketed_weights):
        if bucket < 0:
            LOGGER.info(
                (
                    "ignoring weights {} for pruning; affected loss the most "
                    "and/or performance the least"
                ).format(weights)
            )
            continue

        sparsity = args.pruning_buckets[index + index_offset]
        start_epoch = stabilization_epochs
        end_epoch = stabilization_epochs + pruning_epochs
        update_frequency = 0.5 if pruning_epochs >= 15 else 0.1
        LOGGER.info(
            (
                "adding GradualKSModifier with start_epoch {}, "
                "end_epoch {}, final_sparsity {}, and weights {}"
            ).format(start_epoch, end_epoch, sparsity, weights)
        )
        modifiers.append(
            ks_const(
                init_sparsity=0.05,
                final_sparsity=sparsity,
                start_epoch=start_epoch,
                end_epoch=end_epoch,
                update_frequency=update_frequency,
                params=weights,
            )
        )

    LOGGER.info("creating manager for {} modifiers".format(len(modifiers)))
    manager = manager_const(modifiers)

    yaml_path = (
        args.yaml_path
        if args.yaml_path is not None
        else "{}.config.yaml".format(args.onnx_file_path)
    )
    LOGGER.info("saving config yaml to {}".format(yaml_path))
    manager.save(yaml_path)
    LOGGER.info("saved config yaml to {}".format(yaml_path))


def main(args):
    if args.command == TABLE_COMMAND:
        _table(args)
    elif args.command == PYTORCH_COMMAND:
        _config(
            args,
            PTScheduledModifierManager,
            PTEpochRangeModifier,
            PTSetLearningRateModifier,
            PTLearningRateModifier,
            PTGradualKSModifier,
        )
    elif args.command == TENSORFLOW_COMMAND:
        _config(
            args,
            TFScheduledModifierManager,
            TFEpochRangeModifier,
            TFSetLearningRateModifier,
            TFLearningRateModifier,
            TFGradualKSModifier,
        )
    else:
        raise ValueError("unknown command given of {}".format(args.command))


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
