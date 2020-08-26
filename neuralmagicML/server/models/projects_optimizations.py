"""
DB model classes for project's optimizations and modifiers
"""

import logging
import uuid

from peewee import (
    CharField,
    TextField,
    ForeignKeyField,
    FloatField,
)

from neuralmagicML.server.models.base import (
    BaseCreatedModifiedModel,
    ListObjField,
)
from neuralmagicML.server.models.projects import Project
from neuralmagicML.server.models.projects_profiles import (
    ProjectPerfProfile,
    ProjectLossProfile,
)


__all__ = [
    "ProjectOptimization",
    "ProjectOptimizationModifierPruning",
    "ProjectOptimizationModifierQuantization",
    "ProjectOptimizationModifierLRSchedule",
    "ProjectOptimizationModifierTrainable",
]


_LOGGER = logging.getLogger(__name__)


class ProjectOptimization(BaseCreatedModifiedModel):
    """
    DB model for a project's optimization (stores modifier settings).
    A project may have multiple optimizations stored in the DB.
    """

    optim_id = CharField(primary_key=True, default=lambda: uuid.uuid4().hex)
    project = ForeignKeyField(Project, backref="loss_profiles", on_delete="CASCADE")
    name = TextField(null=True, default="")
    profile_perf = ForeignKeyField(ProjectPerfProfile, null=True, default=None)
    profile_loss = ForeignKeyField(ProjectLossProfile, null=True, default=None)
    start_epoch = FloatField(null=True, default=None)
    end_epoch = FloatField(null=True, default=None)


class ProjectOptimizationModifierPruning(BaseCreatedModifiedModel):
    """
    DB model for a project's optimization pruning modifier.
    """

    modifier_id = CharField(primary_key=True, default=lambda: uuid.uuid4().hex)
    optim = ForeignKeyField(ProjectOptimization, backref="pruning_modifiers")
    start_epoch = FloatField(null=True, default=None)
    end_epoch = FloatField(null=True, default=None)
    update_frequency = FloatField(null=True, default=None)
    mask_type = TextField(null=True, default=None)
    sparsity = FloatField(null=True, default=None)
    sparsity_perf_loss_balance = FloatField(null=True, default=None)
    filter_min_sparsity = FloatField(null=True, default=None)
    filter_min_perf_gain = FloatField(null=True, default=None)
    filter_max_loss_drop = FloatField(null=True, default=None)
    nodes = ListObjField(null=True, default=None)
    est_recovery = FloatField(null=True, default=None)
    est_perf_gain = FloatField(null=True, default=None)
    est_time = FloatField(null=True, default=None)
    est_time_baseline = FloatField(null=True, default=None)


class ProjectOptimizationModifierQuantization(BaseCreatedModifiedModel):
    """
    DB model for a project's optimization quantization modifier.
    """

    modifier_id = CharField(primary_key=True, default=lambda: uuid.uuid4().hex)
    optim = ForeignKeyField(
        ProjectOptimization, backref="quantization_modifiers", on_delete="CASCADE"
    )
    start_epoch = FloatField(null=True, default=None)
    end_epoch = FloatField(null=True, default=None)
    level = TextField(null=True, default=None)
    sparsity_perf_loss_balance = FloatField(null=True, default=None)
    filter_min_perf_gain = FloatField(null=True, default=None)
    filter_max_loss_drop = FloatField(null=True, default=None)
    nodes = ListObjField(null=True, default=None)
    est_recovery = FloatField(null=True, default=None)
    est_perf_gain = FloatField(null=True, default=None)
    est_time = FloatField(null=True, default=None)
    est_time_baseline = FloatField(null=True, default=None)


class ProjectOptimizationModifierLRSchedule(BaseCreatedModifiedModel):
    """
    DB model for a project's learning rate schedule modifier.
    """

    modifier_id = CharField(primary_key=True, default=lambda: uuid.uuid4().hex)
    optim = ForeignKeyField(
        ProjectOptimization, backref="lr_schedule_modifiers", on_delete="CASCADE"
    )
    start_epoch = FloatField(null=True, default=None)
    end_epoch = FloatField(null=True, default=None)
    lr_mods = ListObjField(null=True, default=None)


class ProjectOptimizationModifierTrainable(BaseCreatedModifiedModel):
    """
    DB model for a project's optimization trainable modifier.
    """

    modifier_id = CharField(primary_key=True, default=lambda: uuid.uuid4().hex)
    optim = ForeignKeyField(
        ProjectOptimization, backref="trainable_modifiers", on_delete="CASCADE"
    )
    start_epoch = FloatField(null=True, default=None)
    end_epoch = FloatField(null=True, default=None)
    nodes = ListObjField(null=True, default=None)
