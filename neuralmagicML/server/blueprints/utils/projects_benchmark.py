"""
Helper functions and classes for flask blueprints specific to project benchmark
"""
import logging

from neuralmagicML.server.models import (
    Project,
    ProjectBenchmark,
)
from neuralmagicML.server.blueprints.utils.helpers import HTTPNotFoundError

__all__ = ["get_project_benchmark_by_ids"]

_LOGGER = logging.getLogger(__name__)


def get_project_benchmark_by_ids(project_id: str, benchmark_id: str) -> ProjectBenchmark:
    """
    Get a project benchmark by its project_id and benchmark_id

    :param project_id: project id of the optimizer
    :param benchmark_id: benchmark id of the optimizer
    :return: Project benchmark with provided ids
    """
    benchmark = ProjectBenchmark.get_or_none(
        ProjectBenchmark.project_id == project_id,
        ProjectBenchmark.benchmark_id == benchmark_id)

    if benchmark is None:
        _LOGGER.error(
            "could not find project benchmark for project {} with benchmark_id {}".format(
                project_id, benchmark_id
            )
        )
        raise HTTPNotFoundError(
            "could not find project benchmark for project {} with benchmark_id {}".format(
                project_id, benchmark_id
            )
        )

    return benchmark
