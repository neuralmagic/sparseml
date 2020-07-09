import os
from typing import Dict, List

import yaml
from neuralmagicML.onnx.project.models import RecalModel
from neuralmagicML.onnx.project.recal_project import RecalProject
from neuralmagicML.pytorch.recal import EpochRangeModifier as PyEpochRangeModifer
from neuralmagicML.pytorch.recal import GradualKSModifier as PyGradualKSModifier
from neuralmagicML.pytorch.recal import LearningRateModifier as PyLearningRateModifier
from neuralmagicML.pytorch.recal import (
    SetLearningRateModifier as PySetLearningRateModifier,
)
from neuralmagicML.recal import BaseManager, BaseModifier
from neuralmagicML.tensorflow.recal import EpochRangeModifier as TfEpochRangeModifier
from neuralmagicML.tensorflow.recal import GradualKSModifier as TfGradualKSModifier
from neuralmagicML.tensorflow.recal import (
    LearningRateModifier as TfLearningRateModifier,
)
from neuralmagicML.tensorflow.recal import (
    SetLearningRateModifier as TfSetLearningRateModifier,
)

__all__ = ["RecalConfig"]


def _chunk_array(array: list, chunk_blocks: int = 3, top_percent: float = 0.05):
    """
    Groups array into subarray consisting of the first top percentage of an array as one group
    and the rest of the array sliced into equal sized groups

    :param array: Sorted array
    :param chunk_blocks: Amount of equal sized groups to create along with the top percentage of the array. Default 3
    :param top_percent: Percent of first elements of an array to create group 1.
    :return: Array of size chunk_blocks + 1 containing the top percentage group followed by the in order array chunks.
    """
    if chunk_blocks <= 0:
        raise Exception("chunk_blocks must be equal to or greater than 1")
    if top_percent < 0 or top_percent > 1:
        raise Exception("top_percent must a value between 0 and 1")
    bottom_five_percent = max(int(len(array) * top_percent), 1)

    chunk_length = int((len(array) - bottom_five_percent) / chunk_blocks)

    chunks = [array[:bottom_five_percent]]
    for i in range(chunk_blocks):
        chunks.append(
            array[
                chunk_length * i
                + bottom_five_percent : chunk_length * (i + 1)
                + bottom_five_percent
            ]
        )

    return chunks


def _get_max_loss(layer_analysis: Dict):
    """
    :param layer_analysis: A loss analysis for a layer
    :return: The max loss of said layer
    """
    return max(layer_analysis["sparse"], key=lambda sparse_info: sparse_info["loss"])[
        "loss"
    ]


def _get_timing_change(layer_analysis: Dict, target_sparsity: int = 0.875):
    """
    :param layer_analysis: A perf analysis for a layer
    :param target_sparsity: The target sparsity to attempt the timing of
    :return: The timing change compared to the baseline timing of the sparsity level closest to the
        provided target_sparsity
    """
    baseline_timing = layer_analysis["baseline"]["timing"]

    return (
        baseline_timing
        - min(
            layer_analysis["sparse"],
            key=lambda sparse_info: abs(target_sparsity - sparse_info["sparsity"]),
        )["timing"]
    )


class RecalConfig:
    """
    Class representing a recal config

    :param project_path: Path of the config file's project
    :param config_yaml: The config in yaml format
    :param config_file: The name to save the config file. Default to recal.config.yaml
    """

    def __init__(self, project_path, config_yaml: str, config_file="recal.config.yaml"):
        self._project_path = project_path
        self._config_yaml = config_yaml
        self._name = (
            config_file if config_file.endswith(".yaml") else f"{config_file}.yaml"
        )

    @property
    def yaml(self):
        """
        :return: The yaml file
        """
        return self._config_yaml

    @property
    def path(self):
        """
        :return: The path to the config file
        """
        return os.path.join(self._project_path, self._name)

    def save(self):
        """
        Saves the config file to project path with provided name
        """
        with open(self.path, "w+") as config_file:
            config_file.write(self.yaml)

    @staticmethod
    def create_config(
        recal_project: RecalProject,
        pruning_profile: str,
        sparsities: List[float],
        pruning_update_frequency: float,
        init_pruning_sparsity: float = 0.05,
        loss_analysis: str = None,
        perf_analysis: str = None,
        training_epochs: int = None,
        stabilization_epochs: int = None,
        pruning_epochs: int = None,
        fine_tuning_epochs: int = None,
        framework: str = "pytorch",
        config_file: str = "recal.config.yaml",
        init_lr: float = None,
        init_training_lr: float = None,
        final_training_lr: float = None,
        fine_tuning_gamma: float = 0.2,
    ):
        """
        Creates a recal config for the provided project using the provided pruning strategy.

        Initial learning rate used will either be the average between initial and final learning
        rates used during training or the user provided initial learning rate.

        During fine tuning stage, learning rate will be decreased by gamma factor in 4
        equally spaced amount of epochs.

        :param recal_project: The RecalProject being used to create the config
        :param pruning_profile: The pruning profile to use for the config. Must be:
            [uniform, loss, performance, balanced]
        :param sparsities: A list of max sparsities to prune for each bucket in order of
            low pruning, mid pruning, and high pruning,
        :param pruning_update_frequency: The number of epochs or fraction of epochs to update pruning at
            between start and end
        :param tuning_class: The name of the lr scheduler class to use in fine tuning:
            [StepLR, MultiStepLR, ExponentialLR, CosineAnnealingWarmRestarts]
        :param tuning_lr_kwargs: The dictionary of keyword arguments to pass to the constructor
            for the tuning_class
        :param init_pruning_sparsity: Initial sparsity to start with at the start of the pruning stage. Default 0.05
        :param loss_analysis: File name of the loss analysis. Required if pruning_profile is set to balanced or loss
        :param perf_analysis: File name of the perf analysis. Required if pruning_profile is set to balanced or
            performance
        :param training_epochs: The number of epochs used to train the original mode. Required to provide either
            training_epochs or all of stabilization_epochs, pruning_epochs, and fine_tuning_epochs
        :param stabilization_epochs: The amount of epochs during the stabilization stage
        :param pruning_epochs: The amount of epochs during the pruning stage
        :param fine_tuning_epochs: The amount of epochs during the fine tuning stage
        :param framework: Training framework used between [tensorflow, pytorch]. Defaults to pytorch
        :param config_file: Name of the config file. Default to recal.config.yaml
        :param init_lr: Initial learning rate for pruning.
        :param init_training_lr: Initial learning rate in training
        :param final_training_lr: Final learning rate in training
        :param fine_tuning_gamma: Factor to decrease learning rate at each step.
        """

        recal_model = recal_project.model
        loss_analysis = recal_project.get_sparse_analysis_loss(loss_analysis)
        perf_analysis = recal_project.get_sparse_analysis_perf(perf_analysis)

        if framework == "pytorch":
            EpochRangeModifier = PyEpochRangeModifer
            GradualKSModifier = PyGradualKSModifier
            LearningRateModifier = PyLearningRateModifier
            SetLearningRateModifier = PySetLearningRateModifier
        elif framework == "tensorflow":
            EpochRangeModifier = TfEpochRangeModifer
            GradualKSModifier = TfGradualKSModifier
            LearningRateModifier = TfLearningRateModifier
            SetLearningRateModifier = TfSetLearningRateModifier
        else:
            raise Exception("Only framework pytorch and tensorflow supported")

        if training_epochs is None and (
            stabilization_epochs is None
            or pruning_epochs is None
            or fine_tuning_epochs is None
        ):
            raise Exception(
                "Must either specify training_epochs or "
                + "specify stabilization_epoch, pruning_epochs, and fine_tuning_epochs"
            )

        stabilization_epochs = (
            stabilization_epochs if stabilization_epochs is not None else 1
        )
        pruning_epochs = (
            pruning_epochs if pruning_epochs is not None else int(training_epochs / 3)
        )
        fine_tuning_epochs = (
            fine_tuning_epochs
            if fine_tuning_epochs is not None
            else int(training_epochs / 3)
        )
        total_epochs = stabilization_epochs + pruning_epochs + fine_tuning_epochs

        if pruning_profile == "loss":
            if loss_analysis is None:
                raise Exception(
                    "Must specify loss_analysis when pruning_profile is loss"
                )
            buckets = RecalConfig.create_loss_buckets(recal_model, loss_analysis)

        elif pruning_profile == "performance":
            if perf_analysis is None:
                raise Exception(
                    "Must specify perf_analysis when pruning_profile is performance"
                )
            buckets = RecalConfig.create_perf_buckets(recal_model, perf_analysis)

        elif pruning_profile == "balanced":
            if loss_analysis is None or perf_analysis is None:
                raise Exception(
                    "Must specify loss_analysis and perf_analysis when pruning_profile is balanced"
                )
            buckets = RecalConfig.create_balanced_buckets(
                recal_model, loss_analysis, perf_analysis
            )

        elif pruning_profile == "uniform":
            buckets = RecalConfig.create_uniform_buckets(recal_model)

        else:
            raise Exception(
                "Must set pruning_profile to be one of loss, performance, balanced or uniform"
            )

        key_to_name = {}
        for node in recal_model.prunable_nodes:
            key_to_name[node.node_key] = node.node_name

        modifiers = []

        if init_lr is not None:
            modifiers.append(SetLearningRateModifier(init_lr, start_epoch=0))
        elif init_training_lr is not None and final_training_lr is not None:
            init_lr = (final_training_lr + init_training_lr) / 2
            modifiers.append(SetLearningRateModifier(init_lr, start_epoch=0))
        else:
            raise Exception(
                "Must specifiy either init_training_lr and final_training_lr or init_lr"
            )

        modifiers.append(EpochRangeModifier(0, total_epochs))

        step = int(fine_tuning_epochs / 4)
        milestones = [pruning_epochs + stabilization_epochs]
        for _ in range(3):
            milestones.append(milestones[-1] + step)
        modifiers.append(
            LearningRateModifier(
                "MultiStepLR",
                {"gamma": fine_tuning_gamma, "milestones": milestones},
                init_lr,
                0,
            )
        )

        for index, bucket in enumerate(buckets[1:]):
            final_sparsitiy = sparsities[index]
            modifier = GradualKSModifier(
                init_pruning_sparsity,
                final_sparsitiy,
                start_epoch=stabilization_epochs,
                end_epoch=pruning_epochs,
                update_frequency=pruning_update_frequency,
                params=[key_to_name[layer_key] for layer_key in bucket],
            )
            modifiers.append(modifier)

        return RecalConfig(
            recal_project.path,
            BaseModifier.list_to_yaml(modifiers),
            config_file=config_file,
        )

    @staticmethod
    def create_loss_buckets(recal_model: RecalModel, loss_analysis: Dict) -> List:
        """
        Creates pruning buckets from the model's layers based on maximum loss for each layer.

        Layers with higher loss will have lower pruning, with the top 5 percent of layers with
        the highest losses not being pruned

        :param recal_model: RecalModel
        :param loss_analysis: Loss analysis of the recal model
        :return: 4  lists of layer in order of no pruning, low pruning, mid pruning, high pruning.
        """
        id_to_analysis = {}
        for analysis in loss_analysis:
            id_to_analysis[analysis["id"]] = analysis

        unprunable_layers = [
            layer.node_key
            for layer in recal_model.prunable_nodes
            if layer.contains_input or layer.node_key not in id_to_analysis
        ]
        prunable_layers = [
            id_to_analysis[layer.node_key]
            for layer in recal_model.prunable_nodes
            if not layer.contains_input and layer.node_key in id_to_analysis
        ]

        sorted_analysis = sorted(
            prunable_layers,
            key=lambda layer_analysis: _get_max_loss(layer_analysis),
            reverse=True,
        )
        sorted_ids = [layer["id"] for layer in sorted_analysis]
        buckets = _chunk_array(sorted_ids)
        buckets[0] += unprunable_layers
        return buckets

    @staticmethod
    def create_perf_buckets(recal_model: RecalModel, perf_analysis):
        """
        Creates pruning buckets from the model's layers based on change in timing for each layer.

        Layers with higher changes in timing will have higher pruning, with the bottom 5 percent of layers with
        the lowest change not being pruned

        :param recal_model: RecalModel
        :param perf_analysis: Performance analysis of the recal model
        :return: 4 lists of layer in order of no pruning, low pruning, mid pruning, high pruning.
        """
        id_to_analysis = {}
        for analysis in perf_analysis:
            id_to_analysis[analysis["id"]] = analysis

        unprunable_layers = [
            layer.node_key
            for layer in recal_model.prunable_nodes
            if layer.contains_input or layer.node_key not in id_to_analysis
        ]
        prunable_layers = [
            id_to_analysis[layer.node_key]
            for layer in recal_model.prunable_nodes
            if not layer.contains_input and layer.node_key in id_to_analysis
        ]

        sorted_analysis = sorted(
            prunable_layers,
            key=lambda layer_analysis: _get_timing_change(layer_analysis),
        )
        sorted_ids = [layer["id"] for layer in sorted_analysis]

        buckets = _chunk_array(sorted_ids)
        buckets[0] += unprunable_layers
        return buckets

    @staticmethod
    def create_balanced_buckets(recal_model: RecalModel, loss_analysis, perf_analysis):
        """
        Creates pruning buckets from the model's layers based on losses and change in timing for each layer.

                            | Bottom 5 perc Perf Sens | Low Perf Sens | Mid Perf Sens | High Perf Sens
        Low Loss Sens       | Don't prune             | Prune Mid     | Prune High    | Prune High
        Mid Loss Sens       | Don't prune             | Prune Low     | Prune Mid     | Prune High
        High Loss Sens      | Don't prune             | Prune Low     | Prune Low     | Prune Mid
        Top 5 perc Loss Sens| Don't prune             | Don't prune   | Don't prune   | Don't prune

        :param recal_model: RecalModel
        :param perf_analysis: Performance analysis of the recal model
        :return: 4 lists of layer in order of no pruning, low pruning, mid pruning, high pruning.
        """
        loss_buckets = [
            set(bucket)
            for bucket in RecalConfig.create_loss_buckets(recal_model, loss_analysis)
        ]
        perf_buckets = [
            set(bucket)
            for bucket in RecalConfig.create_perf_buckets(recal_model, perf_analysis)
        ]

        unpruned_bucket = loss_buckets[0] | perf_buckets[0]
        low_bucket = (
            (loss_buckets[1] & perf_buckets[1])
            | (loss_buckets[2] & perf_buckets[1])
            | (loss_buckets[1] & perf_buckets[2])
        )
        mid_bucket = (
            (loss_buckets[3] & perf_buckets[1])
            | (loss_buckets[2] & perf_buckets[2])
            | (loss_buckets[1] & perf_buckets[3])
        )
        high_bucket = (
            (loss_buckets[3] & perf_buckets[2])
            | (loss_buckets[3] & perf_buckets[3])
            | (loss_buckets[2] & perf_buckets[3])
        )

        return [
            list(unpruned_bucket),
            list(low_bucket),
            list(mid_bucket),
            list(high_bucket),
        ]

    @staticmethod
    def create_uniform_buckets(recal_model: RecalModel):
        """
        Creates 2 buckets from a model's layers.
            - first bucket contains layers that can't be prune e.g. layers with model input as input
            - second buckets contains layers that can be pruned

        :param recal_model: Recal model
        :return: List of layer lists representing buckets described above
        """
        layers_with_inputs = [
            layer.node_key
            for layer in recal_model.prunable_nodes
            if layer.contains_input
        ]
        layers_without_inputs = [
            layer.node_key
            for layer in recal_model.prunable_nodes
            if not layer.contains_input
        ]
        return [layers_with_inputs, layers_without_inputs]
