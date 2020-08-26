"""
Flask blueprint setup for serving UI files for the server application
"""
from peewee import JOIN
import logging

from neuralmagicML.server.models import (
    Project,
    ProjectModel,
    ProjectData,
    ProjectOptimization,
    ProjectOptimizationModifierTrainable,
    ProjectOptimizationModifierLRSchedule,
    ProjectOptimizationModifierPruning,
    ProjectOptimizationModifierQuantization,
)

_LOGGER = logging.getLogger(__name__)


__all__ = [
    "API_ROOT_PATH",
    "HTTPNotFoundError",
    "get_project_by_id",
    "get_project_model_by_project_id",
    "get_project_optimizer_by_ids",
]


API_ROOT_PATH = "/api"


class HTTPNotFoundError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


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


def get_project_model_by_project_id(project_id: str) -> ProjectModel:
    """
    Get a project model by its project_id

    :param project_id: project id of the project model
    :return: Project model with the project id
    """
    query = ProjectModel.get_or_none(ProjectModel.project_id == project_id)
    if query is None:
        raise HTTPNotFoundError(
            "could not find model for project_id {}".format(project_id)
        )

    return query


def get_project_by_id(project_id: str) -> Project:
    """
    Get a project by its project_id, with project model and project data joined.

    :param project_id: project id of the project
    :return: Project with the project id
    """
    query = (
        Project.select(Project, ProjectModel, ProjectData)
        .join_from(Project, ProjectModel, JOIN.LEFT_OUTER)
        .join_from(Project, ProjectData, JOIN.LEFT_OUTER)
        .where(Project.project_id == project_id)
        .group_by(Project)
    )
    project = None

    for res in query:
        project = res
        break

    if not project:
        raise HTTPNotFoundError(
            "could not find project with project_id {}".format(project_id)
        )

    return project
