"""
DB model classes for project's profiles such as performance and loss
"""

import logging
import datetime
import uuid

from peewee import (
    CharField,
    TextField,
    ForeignKeyField,
    DateTimeField,
    BooleanField,
    IntegerField,
)
from playhouse.sqlite_ext import JSONField

from neuralmagicML.server.models.base import BaseModel, CSVField
from neuralmagicML.server.models.projects import Project
from neuralmagicML.server.models.jobs import Job


__all__ = ["BaseProjectProfile", "ProjectLossProfile", "ProjectPerfProfile"]


_LOGGER = logging.getLogger(__name__)


class BaseProjectProfile(BaseModel):
    """
    Base DB model for project's profiles such as loss and perf
    """

    profile_id = CharField(primary_key=True, default=lambda: uuid.uuid4().hex)
    created = DateTimeField(default=datetime.datetime.now)
    source = TextField(null=True, default=None)
    job = ForeignKeyField(Job, null=True, default=None)
    analysis = JSONField(null=True, default=None)
    name = TextField(null=True, default="")


class ProjectLossProfile(BaseProjectProfile):
    """
    DB model for a project's loss profile.
    A project may have multiple loss profiles stored in the DB.
    """

    project = ForeignKeyField(Project, backref="profiles_loss")
    pruning_estimations = BooleanField(default=False)
    pruning_estimation_type = TextField(null=True, default=None)
    pruning_structure = TextField(null=True, default=None)
    quantized_estimations = BooleanField(default=False)
    quantized_estimation_type = TextField(null=True, default=None)


class ProjectPerfProfile(BaseProjectProfile):
    """
    DB model for a project's performance profile.
    A project may have multiple perf profiles stored in the DB
    """

    project = ForeignKeyField(Project, backref="profiles_perf")
    batch_size = IntegerField(null=True, default=None)
    core_count = IntegerField(null=True, default=None)
    instruction_sets = CSVField(null=True, default=None)
    pruning_estimations = BooleanField(default=False)
    quantized_estimations = BooleanField(default=False)
    iterations_per_check = IntegerField(null=True, default=None)
    warmup_iterations_per_check = IntegerField(null=True, default=None)
