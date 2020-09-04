"""
Code related to the base implementations for job workers
"""

from typing import Iterator, Dict, Any, Union
import logging
import os
import shutil
from tempfile import NamedTemporaryFile, gettempdir

from neuralmagicML.utils import is_url, download_file_iter, models_download_file_iter
from neuralmagicML.onnx.utils import validate_onnx_file
from neuralmagicML.server.schemas import JobProgressSchema
from neuralmagicML.server.models import database, ProjectModel
from neuralmagicML.server.workers.base import BaseJobWorker


__all__ = ["ModelFromPathJobWorker", "ModelFromRepoJobWorker"]


_LOGGER = logging.getLogger(__name__)


class _ModelLoaderJobWorker(BaseJobWorker):
    """
    A base job worker for retrieving a model from a given uri.

    :param job_id: the id of the job this worker is running under
    :param project_id: the id of the project the worker is running for
    :param model_id: the id of the model the worker is running for
    :param uri: the uri to retrieve
    """

    @classmethod
    def format_args(
        cls, model_id: str, uri: str, **kwargs
    ) -> Union[None, Dict[str, Any]]:
        """
        Format a given args into proper args to be stored for later use
        in the constructor for the job worker.

        :param model_id: the id of the model the worker is running for
        :param uri: the uri to retrieve
        :return: the formatted args to be stored for later use
        """
        return {
            "model_id": model_id,
            "uri": uri,
        }

    def __init__(self, job_id: str, project_id: str, model_id: str, uri: str):
        super().__init__(job_id, project_id)
        self._model_id = model_id
        self._uri = uri

    @property
    def model_id(self) -> str:
        """
        :return: the id of the model the worker is running for
        """
        return self._model_id

    @property
    def uri(self) -> str:
        """
        :return: the uri to retrieve
        """
        return self._uri

    def run(self) -> Iterator[Dict[str, Any]]:
        """
        Perform the work for the job.

        :return: an iterator containing progress update information
        """
        raise NotImplementedError()

    def _get_project_model(self) -> ProjectModel:
        model = ProjectModel.get_or_none(ProjectModel.model_id == self._model_id)

        if model is None:
            raise ValueError(
                "ProjectModel with model_id {} was not found".format(self._model_id)
            )

        return model

    @staticmethod
    def _save_project_model(model: ProjectModel, path: str):
        validate_onnx_file(path)

        with database.atomic() as transaction:
            try:
                model.setup_filesystem()
                model.file = "model.onnx"
                shutil.copy(path, model.file_path)
                # revalidate to make sure the copy worked
                validate_onnx_file(model.file_path)
                model.save()
            except Exception as err:
                _LOGGER.error(
                    "error while creating new project model, rolling back: {}".format(
                        err
                    )
                )

                try:
                    os.remove(model.file_path)
                except OSError as err:
                    pass

                transaction.rollback()
                raise err


class ModelFromPathJobWorker(_ModelLoaderJobWorker):
    """
    A job worker for retrieving a model (currently ONNX) from a given uri.
    The uri can be either a local file path or a public url.

    :param job_id: the id of the job this worker is running under
    :param project_id: the id of the project the worker is running for
    :param model_id: the id of the model the worker is running for
    :param uri: the uri to retrieve
    """

    def __init__(self, job_id: str, project_id: str, model_id: str, uri: str):
        super().__init__(job_id, project_id, model_id, uri)

    def run(self) -> Iterator[Dict[str, Any]]:
        """
        Perform the work for the job.
        Downloads the model from a public url if the uri is a public url.
        Copies the model if the uri is accessible through the local file system.

        :return: an iterator containing progress update information
        """
        if is_url(self._uri):
            for progress in self._run_download():
                yield progress
        else:
            for progress in self._run_local():
                yield progress

    def _run_local(self) -> Iterator[Dict[str, Any]]:
        _LOGGER.info(
            (
                "adding model file to project_id {} and "
                "model_id {} from file path {}"
            ).format(self.project_id, self.model_id, self.uri)
        )

        # yield start progress to mark the expected flow
        yield JobProgressSchema().dump({"iter_indefinite": True, "iter_class": "copy"})

        model = self._get_project_model()

        if not os.path.exists(self._uri):
            raise ValueError("local path of {} does not exist".format(self._uri))

        ModelFromPathJobWorker._save_project_model(model, self._uri)

        _LOGGER.info(
            (
                "added model file to project_id {} and " "model_id {} from file path {}"
            ).format(self.project_id, self.model_id, self.uri)
        )

    def _run_download(self) -> Iterator[Dict[str, Any]]:
        _LOGGER.info(
            (
                "adding model file to project_id {} and " "model_id {} from url {}"
            ).format(self.project_id, self.model_id, self.uri)
        )

        # yield start progress to mark the expected flow
        yield JobProgressSchema().dump(
            {"iter_indefinite": False, "iter_class": "download", "iter_val": 0.0}
        )

        model = self._get_project_model()

        with NamedTemporaryFile() as temp:
            temp_path = os.path.join(gettempdir(), temp.name)

            for download_progress in download_file_iter(
                self._uri, temp_path, overwrite=True
            ):
                progress_val = (
                    float(download_progress.downloaded)
                    / float(download_progress.content_length)
                    if download_progress.content_length
                    else None
                )

                yield JobProgressSchema().dump(
                    {
                        "iter_indefinite": False,
                        "iter_class": "download",
                        "iter_val": progress_val,
                    }
                )

            ModelFromPathJobWorker._save_project_model(model, temp_path)

        _LOGGER.info(
            ("added model file to project_id {} and " "model_id {} from url {}").format(
                self.project_id, self.model_id, self.uri
            )
        )


class ModelFromRepoJobWorker(_ModelLoaderJobWorker):
    """
    A job worker for retrieving a model (currently ONNX) from a given uri
    from within the Neural Magic model repo.

    :param job_id: the id of the job this worker is running under
    :param project_id: the id of the project the worker is running for
    :param model_id: the id of the model the worker is running for
    :param uri: the uri to retrieve
    """

    def __init__(self, job_id: str, project_id: str, model_id: str, uri: str):
        super().__init__(job_id, project_id, model_id, uri)

    def run(self) -> Iterator[Dict[str, Any]]:
        """
        Perform the work for the job.
        Downloads the model from the model repo.

        :return: an iterator containing progress update information
        """
        _LOGGER.info(
            (
                "adding model file to project_id {} and "
                "model_id {} from model repo {}"
            ).format(self.project_id, self.model_id, self.uri)
        )

        # yield start progress to mark the expected flow
        yield JobProgressSchema().dump(
            {"iter_indefinite": False, "iter_class": "download", "iter_val": 0.0}
        )

        model = self._get_project_model()

        with NamedTemporaryFile() as temp:
            temp_path = os.path.join(gettempdir(), temp.name)

            for download_progress in models_download_file_iter(
                self._uri, overwrite=True, save_path=temp_path
            ):
                progress_val = (
                    float(download_progress.downloaded)
                    / float(download_progress.content_length)
                    if download_progress.content_length
                    else None
                )

                yield JobProgressSchema().dump(
                    {
                        "iter_indefinite": False,
                        "iter_class": "download",
                        "iter_val": progress_val,
                    }
                )

            ModelFromPathJobWorker._save_project_model(model, temp_path)

        _LOGGER.info(
            (
                "added model file to project_id {} and "
                "model_id {} from model repo {}"
            ).format(self.project_id, self.model_id, self.uri)
        )
