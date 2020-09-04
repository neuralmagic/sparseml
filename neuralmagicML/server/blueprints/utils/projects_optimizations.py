"""
Helper functions and classes for flask blueprints specific to project optimizations
"""

from typing import Union, NamedTuple, Tuple, Dict, List, Any
import logging
import math

from marshmallow import ValidationError
from peewee import JOIN

from neuralmagicML.server.blueprints.utils.helpers import HTTPNotFoundError
from neuralmagicML.server.blueprints.utils.projects import get_project_by_id
from neuralmagicML.server.blueprints.utils.projects_optimizations_pruning import (
    PruningSettings,
    PruningModelEvaluator,
)
from neuralmagicML.server.schemas import (
    data_dump_and_validation,
    ProjectOptimizationModifierTrainableNodeSchema,
    ProjectOptimizationModifierLRSchema,
)
from neuralmagicML.server.models import (
    Project,
    ProjectModel,
    ProjectPerfProfile,
    ProjectLossProfile,
    ProjectOptimization,
    ProjectOptimizationModifierTrainable,
    ProjectOptimizationModifierLRSchedule,
    ProjectOptimizationModifierPruning,
    ProjectOptimizationModifierQuantization,
)


__all__ = [
    "optim_validate_and_get_project_by_id",
    "get_project_optimizer_by_ids",
    "OptimEpochs",
    "default_epochs_distribution",
    "default_pruning_settings",
    "get_profiles_by_id",
    "optim_trainable_default_nodes",
    "optim_lr_sched_default_mods",
    "optim_trainable_updater",
    "optim_pruning_updater",
    "optim_lr_sched_updater",
    "optim_updater",
    "create_config",
    "validate_pruning_nodes",
]


_LOGGER = logging.getLogger(__name__)


def optim_validate_and_get_project_by_id(project_id: str) -> Project:
    """
    Get a project by project id and validate that it is setup correctly for optims.
    Raises not found errors for no project
    and validation errors for no project model, and no project analysis.

    :param project_id: id of the project to get
    :return: the retrieved project
    """
    project = get_project_by_id(project_id)

    if project.model is None:
        _LOGGER.error("could not find model for project_id {}".format(project_id))
        raise ValidationError(
            "could not find model for project_id {}".format(project_id)
        )

    if not project.model.analysis:
        _LOGGER.error(
            "could not find model analysis for project_id {}".format(project_id)
        )
        raise ValidationError(
            "could not find model analysis for project_id {}".format(project_id)
        )

    return project


def get_project_optimizer_by_ids(project_id: str, optim_id: str) -> ProjectOptimization:
    """
    Get a project optimizer by its project_id and optim_id

    :param project_id: project id of the optimizer
    :param optim_id: optim id of the optimizer
    :return: Project optimizer with provided ids
    """
    query = (
        ProjectOptimization.select(
            ProjectOptimization,
            ProjectOptimizationModifierLRSchedule,
            ProjectOptimizationModifierPruning,
            ProjectOptimizationModifierQuantization,
            ProjectOptimizationModifierTrainable,
        )
        .join_from(
            ProjectOptimization, ProjectOptimizationModifierLRSchedule, JOIN.LEFT_OUTER,
        )
        .join_from(
            ProjectOptimization, ProjectOptimizationModifierPruning, JOIN.LEFT_OUTER,
        )
        .join_from(
            ProjectOptimization,
            ProjectOptimizationModifierQuantization,
            JOIN.LEFT_OUTER,
        )
        .join_from(
            ProjectOptimization, ProjectOptimizationModifierTrainable, JOIN.LEFT_OUTER,
        )
        .where(
            ProjectOptimization.project_id == project_id,
            ProjectOptimization.optim_id == optim_id,
        )
        .group_by(ProjectOptimization)
    )

    optim = None
    for ref in query:
        optim = ref
        break

    if optim is None:
        _LOGGER.error(
            "could not find project optimizer for project {} with optim_id {}".format(
                project_id, optim_id
            )
        )
        raise HTTPNotFoundError(
            "could not find project optimizer for project {} with optim_id {}".format(
                project_id, optim_id
            )
        )

    return optim


def get_profiles_by_id(
    profile_perf_id: Union[None, str], profile_loss_id: Union[None, str]
) -> Tuple[ProjectPerfProfile, ProjectLossProfile]:
    """
    Get a performance and loss profile by their ids.
    If not found will return None instead of raising not found.

    :param profile_perf_id: id of the performance profile to get
    :param profile_loss_id: id of the loss profile to get
    :return: tuple containing (performance profile, loss profile)
    """
    profile_perf = (
        ProjectPerfProfile.get_or_none(ProjectPerfProfile.profile_id == profile_perf_id)
        if profile_perf_id
        else None
    )
    profile_loss = (
        ProjectLossProfile.get_or_none(ProjectLossProfile.profile_id == profile_loss_id)
        if profile_loss_id
        else None
    )

    return profile_perf, profile_loss


OptimEpochs = NamedTuple(
    "OptimEpochs",
    [
        ("training_epochs", int),
        ("start_epoch", int),
        ("stabilization_epochs", int),
        ("pruning_epochs", int),
        ("fine_tuning_epochs", int),
        ("end_epoch", int),
        ("pruning_start_epoch", int),
        ("pruning_end_epoch", int),
        ("pruning_update_frequency", float),
        ("fine_tuning_start_epoch", float),
    ],
)


def default_epochs_distribution(training_epochs: Union[None, int]) -> OptimEpochs:
    """
    Create default epochs distribution for optimizing a model given a number
    of training epochs.

    :param training_epochs: the original training epochs, if not set will default to 100
    :return: the default epochs distribution for optimizing a model
    """
    if not training_epochs or training_epochs < 1:
        training_epochs = 100

    start_epoch = 0
    stabilization_epochs = 1
    pruning_epochs = int(training_epochs / 3)
    fine_tuning_epochs = int(training_epochs / 4)
    end_epoch = stabilization_epochs + pruning_epochs + fine_tuning_epochs
    pruning_start_epoch = stabilization_epochs
    pruning_end_epoch = pruning_start_epoch + pruning_epochs
    pruning_update_frequency = pruning_epochs / 40.0  # make sure we have 40 updates
    fine_tuning_start_epoch = pruning_end_epoch

    return OptimEpochs(
        training_epochs,
        start_epoch,
        stabilization_epochs,
        pruning_epochs,
        fine_tuning_epochs,
        end_epoch,
        pruning_start_epoch,
        pruning_end_epoch,
        pruning_update_frequency,
        fine_tuning_start_epoch,
    )


def default_pruning_settings():
    """
    :return: the default pruning settings for optimizing a model
    """
    mask_type = "unstructured"  # TODO: update based on quantization
    sparsity = 0.85  # TODO: dynamically choose sparsity level
    balance_perf_loss = 0.5
    filter_min_sparsity = 0.4
    filter_min_perf_gain = 0.9
    filter_max_loss_drop = None  # TODO: dynamically fill in from model

    return PruningSettings(
        mask_type,
        sparsity,
        balance_perf_loss,
        filter_min_sparsity,
        filter_min_perf_gain,
        filter_max_loss_drop,
    )


def optim_trainable_default_nodes(
    default_trainable: bool,
    model_analysis: Dict,
    node_overrides: Union[None, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Create the default trainable nodes for optimizing a model.
    Creates a node for all prunable nodes in the model with trainable set to
    default_trainable.

    :param default_trainable: True to default all prunable nodes to trainable,
        False otherwise
    :param model_analysis: the analysis for the model
    :param node_overrides: specific node overrides to use instead of default_trainable
    :return: the default trainable nodes
    """
    trainable_nodes = []

    for node in model_analysis["nodes"]:
        if not node["prunable"]:
            continue

        trainable_val = default_trainable

        if node_overrides:
            for override in node_overrides:
                if override["node_id"] == node["id"]:
                    trainable_val = override["trainable"]
                    break

        trainable_nodes.append(
            data_dump_and_validation(
                ProjectOptimizationModifierTrainableNodeSchema(),
                {"node_id": node["id"], "trainable": trainable_val},
            )
        )

    return trainable_nodes


def optim_lr_sched_default_mods(
    training_init_lr: Union[float, None],
    training_final_lr: Union[float, None],
    start_epoch: Union[float, None],
    start_fine_tuning_epoch: Union[float, None],
    end_epoch: Union[float, None],
) -> List[Dict[str, Any]]:
    """
    Default modifiers for an LR schedule for pruning a model.
    If training_init_lr is set, adds a set LR modifier.
    If training_init_lr and training_final_lr are set, adds a step LR modifier.

    :param training_init_lr: the initial LR for training
    :param training_final_lr: the final LR for training
    :param start_epoch: the epoch training should start at
    :param start_fine_tuning_epoch: the epoch fine tuning should start at
    :param end_epoch: the final epoch for training
    :return: the default modifiers for an LR schedule
    """
    optim_lr_mods = []

    if training_init_lr is not None and start_epoch is not None:
        pruning_lr = (
            training_init_lr
            if not training_final_lr
            else (training_init_lr + training_final_lr) / 2.0
        )
        optim_lr_mods.append(
            data_dump_and_validation(
                ProjectOptimizationModifierLRSchema(),
                {
                    "clazz": "set",
                    "start_epoch": start_epoch,
                    "end_epoch": -1.0,
                    "init_lr": pruning_lr,
                    "args": {},
                },
            )
        )

        if (
            training_final_lr is not None
            and start_fine_tuning_epoch is not None
            and end_epoch is not None
        ):
            fine_tuning_epochs = end_epoch - start_fine_tuning_epoch
            gamma = 0.25
            init_lr = pruning_lr * gamma
            target_final_lr = training_final_lr * 0.1
            num_steps = math.log(target_final_lr / init_lr) / math.log(
                gamma
            )  # final_lr = init_lr * gamma ^ n : solve for n
            step_size = math.floor((fine_tuning_epochs - 1.0) / num_steps)
            optim_lr_mods.append(
                data_dump_and_validation(
                    ProjectOptimizationModifierLRSchema(),
                    {
                        "clazz": "step",
                        "start_epoch": start_epoch,
                        "end_epoch": -1.0,
                        "init_lr": pruning_lr,
                        "args": {"step_size": step_size, "gamma": gamma},
                    },
                )
            )

    return optim_lr_mods


def optim_trainable_updater(
    trainable: ProjectOptimizationModifierTrainable,
    start_epoch: Union[None, float] = None,
    end_epoch: Union[None, float] = None,
    nodes: Union[None, List[Dict[str, Any]]] = None,
    global_start_epoch: Union[None, float] = None,
    global_end_epoch: Union[None, float] = None,
):
    """
    Update a trainable DB model

    :param trainable: the DB model
    :param start_epoch: the start_epoch to set, if any
    :param end_epoch: the end_epoch to set, if any
    :param nodes: the nodes to set, if any
    :param global_start_epoch: the optim's start epoch,
        if set and greater than current start_epoch will set start_epoch to this
    :param global_end_epoch: the optim's end epoch,
        if set and less than current end_epoch will set end_epoch to this
    """
    if start_epoch is not None:
        trainable.start_epoch = start_epoch

    if end_epoch is not None:
        trainable.end_epoch = end_epoch

    if nodes is not None:
        trainable.nodes = nodes

    if (
        global_start_epoch is not None
        and trainable.start_epoch is not None
        and global_start_epoch > trainable.start_epoch
    ):
        trainable.start_epoch = global_start_epoch

    if (
        global_end_epoch is not None
        and trainable.end_epoch is not None
        and global_end_epoch < trainable.end_epoch
    ):
        trainable.end_epoch = global_end_epoch


def optim_pruning_updater(
    pruning: ProjectOptimizationModifierPruning,
    start_epoch: Union[None, float] = None,
    end_epoch: Union[None, float] = None,
    update_frequency: Union[None, float] = None,
    pruning_settings: Union[PruningSettings, None] = None,
    mask_type: Union[str, None] = None,
    sparsity: Union[float, None] = None,
    balance_perf_loss: Union[float, None] = None,
    filter_min_sparsity: Union[float, None] = None,
    filter_min_perf_gain: Union[float, None] = None,
    filter_max_loss_drop: Union[float, None] = None,
    nodes: Union[None, List[Dict[str, Any]]] = None,
    model: Union[None, ProjectModel] = None,
    profile_perf: Union[None, ProjectPerfProfile] = None,
    profile_loss: Union[None, ProjectLossProfile] = None,
    global_start_epoch: Union[None, float] = None,
    global_end_epoch: Union[None, float] = None,
):
    """
    Update a pruning DB model

    :param pruning: the DB model
    :param start_epoch: the start_epoch to set, if any
    :param end_epoch: the end_epoch to set, if any
    :param update_frequency: the update_frequency to set, if any
    :param pruning_settings: the pruning_settings to use for updating /
        automatically generating new sparsity levels for nodes.
        If provided, overrides mask_type, sparsity, balance_perf_loss,
        filter_min_sparsity, filter_min_perf_gain, and filter_max_loss_drop
    :param mask_type: the mask_type to set, if any
    :param sparsity: the sparsity level to set, if set will update /
        automatically generate new sparsity levels for nodes
    :param balance_perf_loss: the balance_perf_loss to set, if any
    :param filter_min_sparsity: the filter_min_sparsity to set, if any
    :param filter_min_perf_gain: the filter_min_perf_gain to set, if any
    :param filter_max_loss_drop: the filter_max_loss_drop to set, if any
    :param nodes: the nodes to set, if set will update
        node and model metrics for perf and loss
    :param model: the model to use to update values with
    :param profile_perf: the performance profile to use to update values with,
        if set will update nodes
    :param profile_loss: the loss profile to use to update values with,
        if set will update nodes
    :param global_start_epoch: the optim's start epoch,
        if set and greater than current start_epoch will set start_epoch to this
    :param global_end_epoch: the optim's end epoch,
        if set and less than current end_epoch will set end_epoch to this
    """
    if start_epoch is not None:
        pruning.start_epoch = start_epoch

    if end_epoch is not None:
        pruning.end_epoch = end_epoch

    if update_frequency is not None:
        pruning.update_frequency = update_frequency

    if pruning_settings is not None:
        mask_type = pruning_settings.mask_type
        sparsity = pruning_settings.sparsity
        balance_perf_loss = pruning_settings.balance_perf_loss
        filter_min_sparsity = pruning_settings.filter_min_sparsity
        filter_min_perf_gain = pruning_settings.filter_min_perf_gain
        filter_max_loss_drop = pruning_settings.filter_max_loss_drop

    if mask_type is not None:
        pruning.mask_type = mask_type

    if (
        sparsity is not None
        or nodes is not None
        or profile_perf is not None
        or profile_loss is not None
    ):
        model = PruningModelEvaluator(
            model.analysis,
            profile_perf.analysis if profile_perf else None,
            profile_loss.analysis if profile_loss else None,
        )
        model.create_rescale_functions()
        model.eval_baseline(default_pruning_settings().sparsity)

        if sparsity is not None:
            settings = PruningSettings(
                mask_type,
                sparsity,
                balance_perf_loss if balance_perf_loss is not None else 0.5,
                filter_min_sparsity,
                filter_min_perf_gain,
                filter_max_loss_drop,
            )
            pruning.sparsity = settings.sparsity
            pruning.balance_perf_loss = settings.balance_perf_loss
            pruning.filter_min_sparsity = settings.filter_min_sparsity
            pruning.filter_min_perf_gain = settings.filter_min_perf_gain
            pruning.filter_max_loss_drop = settings.filter_max_loss_drop
            model.eval_pruning(settings)
        elif profile_perf is not None or profile_loss is not None and nodes is None:
            # only profiles changed, update with current nodes
            nodes = pruning.nodes

        if nodes is not None:
            model.apply_node_overrides(nodes)

        nodes_res, model_res = model.to_dict_values()
        pruning.nodes = nodes_res
        pruning.est_recovery = model_res["est_recovery"]
        pruning.est_perf_gain = model_res["est_perf_gain"]
        pruning.est_time = model_res["est_time"]
        pruning.est_time_baseline = model_res["est_time_baseline"]

    if (
        global_start_epoch is not None
        and pruning.start_epoch is not None
        and global_start_epoch > pruning.start_epoch
    ):
        pruning.start_epoch = global_start_epoch

    if (
        global_end_epoch is not None
        and pruning.end_epoch is not None
        and global_end_epoch < pruning.end_epoch
    ):
        pruning.end_epoch = global_end_epoch


def optim_lr_sched_updater(
    lr_sched: ProjectOptimizationModifierLRSchedule,
    lr_mods: Union[None, List[Dict[str, Any]]] = None,
    global_start_epoch: Union[None, float] = None,
    global_end_epoch: Union[None, float] = None,
):
    """
    Update an LR schedule DB model.
    Will always update schedule level details from the contained lr_mods

    :param lr_sched: the DB model
    :param lr_mods: the mods to set, if any
    :param global_start_epoch: the optim's start epoch,
        if set and greater than current start_epoch will set start_epoch to this
    :param global_end_epoch: the optim's end epoch,
        if set and less than current end_epoch will set end_epoch to this
    """
    if lr_mods is None:
        lr_mods = lr_sched.lr_mods

    if lr_mods:
        for lr_mod in lr_mods:
            if global_start_epoch is not None:
                if (
                    lr_mod["start_epoch"] is not None
                    and global_start_epoch > lr_mod["start_epoch"]
                ):
                    lr_mod["start_epoch"] = global_start_epoch

            if global_end_epoch is not None:
                if (
                    lr_mod["end_epoch"] is not None
                    and global_end_epoch < lr_mod["end_epoch"]
                ):
                    lr_mod["end_epoch"] = global_end_epoch

    start_epoch = None
    end_epoch = None
    init_lr = None
    final_lr = None

    if lr_mods:
        for node in lr_mods:
            if (
                (node["start_epoch"] or node["start_epoch"] == 0.0)
                and node["start_epoch"] >= 0.0
                and (start_epoch is None or node["start_epoch"] < start_epoch)
            ):
                start_epoch = node["start_epoch"]
                init_lr = node["init_lr"]

            if (node["end_epoch"] or node["end_epoch"] == 0.0) and (
                end_epoch is None or node["end_epoch"] > end_epoch
            ):
                end_epoch = node["end_epoch"]

    lr_sched.lr_mods = lr_mods
    lr_sched.start_epoch = start_epoch
    lr_sched.end_epoch = end_epoch
    lr_sched.init_lr = init_lr
    lr_sched.final_lr = final_lr


def optim_updater(
    optim: ProjectOptimization,
    name: Union[None, str] = None,
    profile_perf: Union[None, ProjectPerfProfile] = -1,
    profile_loss: Union[None, ProjectLossProfile] = -1,
    start_epoch: Union[None, float] = None,
    end_epoch: Union[None, float] = None,
    mod_start_epoch: Union[None, float] = None,
    mod_end_epoch: Union[None, float] = None,
):
    """
    Update an optim DB model

    :param optim: the DB model
    :param name: the name to set, if any
    :param profile_perf: the performance profile to set, if any
    :param profile_loss: the loss profile to set, if any
    :param start_epoch: the start_epoch to set, if any
    :param end_epoch: the end_epoch to set, if any
    :param mod_start_epoch: a contained modifier's updated start epoch,
        if set and greater than current start_epoch will set start_epoch to this
    :param mod_end_epoch: a contained modifier's updated end epoch,
        if set and less than current end_epoch will set end_epoch to this
    """
    if name is not None:
        optim.name = name

    if profile_perf != -1:
        optim.profile_perf = profile_perf

    if profile_loss != -1:
        optim.profile_loss = profile_loss

    if start_epoch is not None:
        optim.start_epoch = start_epoch

    if end_epoch is not None:
        optim.end_epoch = end_epoch

    if (
        mod_start_epoch is not None
        and optim.start_epoch is not None
        and mod_start_epoch < optim.start_epoch
    ):
        optim.start_epoch = mod_start_epoch

    if (
        mod_end_epoch is not None
        and optim.end_epoch is not None
        and mod_end_epoch < optim.end_epoch
    ):
        optim.end_epoch = mod_end_epoch


def create_config(project: Project, optim: ProjectOptimization, framework: str) -> str:
    """
    Creates a optimization config yaml for a given project and optimization

    :param project: project to create with
    :param optim: project optimizer to create with
    :param framework: the framework to create the config for
    """
    # add imports in function so they don't fail if they don't have env setup
    # for frameworks other than the requested
    if framework == "pytorch":
        from neuralmagicML.pytorch.recal import (
            ScheduledModifierManager,
            EpochRangeModifier,
            SetLearningRateModifier,
            LearningRateModifier,
            GradualKSModifier,
        )
    elif framework == "tensorflow":
        from neuralmagicML.pytorch.recal import (
            ScheduledModifierManager,
            EpochRangeModifier,
            SetLearningRateModifier,
            LearningRateModifier,
            GradualKSModifier,
        )
    else:
        _LOGGER.error("Unsupported framework {} provided".format(framework))
        raise ValidationError("Unsupported framework {} provided".format(framework))

    mods = [
        EpochRangeModifier(
            start_epoch=optim.start_epoch if optim.start_epoch is not None else -1,
            end_epoch=optim.end_epoch if optim.end_epoch is not None else -1,
        )
    ]
    node_weight_name_lookup = {
        node["id"]: node["weight_name"]
        for node in project.model.analysis["nodes"]
        if node["prunable"]
    }

    for mod in optim.pruning_modifiers:
        sparsity_to_params = {}

        for node in mod.nodes:
            # node is coming from DB, so already had prunable checks
            # add assert here to fail early for non prunable nodes
            assert node["node_id"] in node_weight_name_lookup

            sparsity = node["sparsity"]
            node_id = node["node_id"]
            weight_name = node_weight_name_lookup[node_id]

            if sparsity is None:
                continue

            if sparsity not in sparsity_to_params:
                sparsity_to_params[sparsity] = []

            sparsity_to_params[sparsity].append(weight_name)

        for sparsity, params in sparsity_to_params.items():
            grad_ks = GradualKSModifier(
                init_sparsity=0.05,
                final_sparsity=sparsity,
                start_epoch=mod.start_epoch if mod.start_epoch is not None else -1,
                end_epoch=mod.end_epoch if mod.end_epoch is not None else -1,
                update_frequency=mod.update_frequency if mod.update_frequency else -1,
                params=params,
            )

            if mod.mask_type:
                grad_ks.mask_type = mod.mask_type

            mods.append(grad_ks)

    for lr_schedule_modifier in optim.lr_schedule_modifiers:
        for mod in lr_schedule_modifier.lr_mods:
            mod = ProjectOptimizationModifierLRSchema().dump(mod)
            start_epoch = mod["start_epoch"] if mod["start_epoch"] is not None else -1
            end_epoch = mod["end_epoch"] if mod["end_epoch"] is not None else -1

            if mod["clazz"] == "set":
                mods.append(
                    SetLearningRateModifier(
                        learning_rate=mod["init_lr"], start_epoch=start_epoch,
                    )
                )
            else:
                lr_class_mapping = {
                    "step": "StepLR",
                    "multi_step": "MultiStepLR",
                    "exponential": "ExponentialLR",
                }
                assert mod["clazz"] in lr_class_mapping
                mods.append(
                    LearningRateModifier(
                        lr_class=lr_class_mapping[mod["clazz"]],
                        lr_kwargs=mod["args"],
                        init_lr=mod["init_lr"],
                        start_epoch=start_epoch,
                        end_epoch=end_epoch,
                    )
                )

    # TODO: add trainable
    # TODO: add quantization support when ready

    return str(ScheduledModifierManager(mods))


def validate_pruning_nodes(project: Project, nodes: List[Dict[str, Any]]):
    """
    Validate a list of given nodes are prunable in the model.
    Raises a validation error if nodes are set that are not prunable.

    :param project: the project to validate with
    :param nodes: the nodes to validate
    """
    prunable_nodes = set(
        [node["id"] for node in project.model.analysis["nodes"] if node["prunable"]]
    )

    for node in nodes:
        if node["node_id"] not in prunable_nodes:
            _LOGGER.error("Node {} is not prunable".format(node["node_id"]))
            raise ValidationError("Node {} is not prunable".format(node["node_id"]))
