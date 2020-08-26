"""
Code related to the base implementations for job workers
"""

from abc import abstractmethod
from typing import Dict, Any, Iterator


__all__ = ["JobWorkerRegistryHolder", "BaseJobWorker"]


class JobWorkerRegistryHolder(type):
    """
    Registry class for handling and storing BaseJobWorker sub class instances.
    All subclasses are added to the the REGISTRY property
    """

    REGISTRY = {}

    def __new__(cls, name, bases, attrs):
        new_cls = type.__new__(cls, name, bases, attrs)
        cls.REGISTRY[new_cls.__name__] = new_cls

        return new_cls


class BaseJobWorker(object, metaclass=JobWorkerRegistryHolder):
    """
    The base job worker instance all job workers must extend

    :param job_id: the id of the job the worker is being run for
    :param project_id: the id of the project the job belongs to
    """

    @classmethod
    def get_type(cls) -> str:
        """
        :return: the type of job worker
        """
        return cls.__name__

    @classmethod
    @abstractmethod
    def format_args(cls, **kwargs) -> Dict[str, Any]:
        """
        Format a given args into proper args to be stored for later use
        in the constructor for the job worker.

        :param kwargs: the args to format
        :return: the formatted args to be stored for later use
        """
        raise NotImplementedError()

    def __init__(self, job_id: str, project_id: str):
        self._job_id = job_id
        self._project_id = project_id

    @property
    def job_id(self) -> str:
        """
        :return: the id of the job the worker is being run for
        """
        return self._job_id

    @property
    def project_id(self) -> str:
        """
        :return: the id of the project the job belongs to
        """
        return self._project_id

    @abstractmethod
    def run(self) -> Iterator[Dict[str, Any]]:
        """
        Perform the work for the job.
        Must be implemented as an iterator that returns a
        dictionary containing the progress object on each progress step.

        :return: an iterator containing progress update information
        """
        raise NotImplementedError()
