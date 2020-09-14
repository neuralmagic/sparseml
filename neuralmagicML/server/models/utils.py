import os

from flask import Flask
from playhouse.flask_utils import FlaskDB
from neuralmagicML.server.models import (
    database,
    storage,
    Job,
    Project,
    ProjectModel,
    ProjectData,
    ProjectLossProfile,
    ProjectPerfProfile,
    ProjectOptimization,
    ProjectOptimizationModifierPruning,
    ProjectOptimizationModifierQuantization,
    ProjectOptimizationModifierLRSchedule,
    ProjectOptimizationModifierTrainable,
)

__all__ = ["database_setup"]


def database_setup(working_dir: str, app: Flask = None):
    storage.init(working_dir)
    db_path = os.path.join(working_dir, "db.sqlite")
    database.init(
        db_path,
        max_connections=10,
        stale_timeout=300,
        timeout=0,
        check_same_thread=False,
    )
    if app:
        FlaskDB(app, database)

    database.connect()
    models = [
        Job,
        Project,
        ProjectModel,
        ProjectData,
        ProjectLossProfile,
        ProjectPerfProfile,
        ProjectOptimization,
        ProjectOptimizationModifierPruning,
        ProjectOptimizationModifierQuantization,
        ProjectOptimizationModifierLRSchedule,
        ProjectOptimizationModifierTrainable,
    ]
    database.create_tables(
        models=models,
        safe=True,
    )
    for model in models:
        model.raw("PRAGMA foreign_keys=ON").execute()
    database.close()
