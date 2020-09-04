"""
Helper functions and classes for flask blueprints specific to projects
"""

from peewee import JOIN
import logging

from neuralmagicML.server.models import (
    Project,
    ProjectModel,
    ProjectData,
)
from neuralmagicML.server.blueprints.utils.helpers import HTTPNotFoundError

_LOGGER = logging.getLogger(__name__)


__all__ = [
    "get_project_by_id",
    "get_project_model_by_project_id",
]


def get_project_model_by_project_id(
    project_id: str, raise_not_found: bool = True
) -> ProjectModel:
    """
    Get a project model by its project_id

    :param project_id: project id of the project model
    :param raise_not_found: if no model is found raise an HTTPNotFoundError,
        otherwise return the result no matter what
    :return: Project model with the project id
    """
    query = ProjectModel.get_or_none(ProjectModel.project_id == project_id)

    if query is None and raise_not_found:
        _LOGGER.error("could not find model for project_id {}".format(project_id))
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
        _LOGGER.error("could not find project with project_id {}".format(project_id))
        raise HTTPNotFoundError(
            "could not find project with project_id {}".format(project_id)
        )

    project.model = None

    for model in project.models:
        project.model = model
        break

    return project
