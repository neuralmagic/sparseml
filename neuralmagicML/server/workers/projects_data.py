"""
Code related to the project_data implementations for job workers
"""
from typing import Iterator, Dict, Any, Union
import tarfile
import logging
import os
import shutil
from tempfile import NamedTemporaryFile, gettempdir, TemporaryDirectory

from neuralmagicML.utils import is_url, download_file_iter, models_download_file_iter
from neuralmagicML.server.blueprints.utils import validate_model_data
from neuralmagicML.server.schemas import JobProgressSchema
from neuralmagicML.server.models import database, ProjectData, ProjectModel
from neuralmagicML.server.workers.base import BaseJobWorker

_LOGGER = logging.getLogger(__name__)

__all__ = ["DataFromPathJobWorker", "DataFromRepoJobWorker"]


class _DataLoaderJobWorker(BaseJobWorker):
    @classmethod
    def format_args(
        cls, data_id: str, uri: str, **kwargs
    ) -> Union[None, Dict[str, Any]]:
        """
        Format a given args into proper args to be stored for later use
        in the constructor for the job worker.

        :param data_id: the id of the data the worker is running for
        :param uri: the uri to retrieve
        :return: the formatted args to be stored for later use
        """
        return {"data_id": data_id, "uri": uri}

    def __init__(self, job_id: str, project_id: str, data_id: str, uri: str):
        super().__init__(job_id, project_id)
        self._data_id = data_id
        self._uri = uri

    @property
    def data_id(self) -> str:
        """
        :return: the id of the data the worker is running for
        """
        return self._data_id

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
        model = ProjectModel.get_or_none(ProjectModel.project_id == self._project_id)

        if model is None:
            raise ValueError(
                "ProjectModel with project_id {} was not found".format(self._project_id)
            )

        return model

    def _create_project_data(self) -> ProjectData:
        original = self._get_project_data()
        data = ProjectData.create(
            project=original.project, source=original.source, job=original.job
        )
        return data

    def _get_project_data(self) -> ProjectData:
        data = ProjectData.get_or_none(ProjectData.data_id == self._data_id)

        if data is None:
            raise ValueError(
                "ProjectData with data_id {} was not found".format(self._data_id)
            )

        return data

    @staticmethod
    def _save_project_data(data: ProjectData, path: str, model_path: str):
        validate_model_data(path, model_path)

        try:
            data.setup_filesystem()
            data.file = "{}.npz".format(data.data_id)
            shutil.copy(path, data.file_path)
            # revalidate to make sure the copy worked
            validate_model_data(data.file_path, model_path)
            data.save()
        except:
            if data:
                try:
                    os.remove(data.file_path)
                except OSError as err:
                    pass

                try:
                    data.delete_instance()
                except Exception as rollback_Err:
                    _LOGGER.error(
                        "error while rolling back new data: {}".format(rollback_err)
                    )

            _LOGGER.error(
                "error while creating new project data, rolling back: {}".format(err)
            )
            raise err

    def _run_copy_folder(self, path: str) -> Iterator[ProjectData]:
        files = []
        if len(os.listdir(path)) == 0:
            raise ValueError("Directory {} is empty".format(path))

        # Obtains all file names in path. Goes in one folder level if there is a directory under path
        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)
            if os.path.isdir(file_path):
                files += [
                    os.path.join(file_path, subfile_name)
                    for subfile_name in os.listdir(file_path)
                    if not os.path.isdir(os.path.join(file_path, subfile_name))
                ]
            else:
                files.append(file_path)

        for index, file_path in enumerate(files):
            if index == 0:
                self._run_copy_file(file_path, self._get_project_data())
                yield (index + 1) / len(files), self._get_project_data()
            else:
                project_data = self._create_project_data()
                self._run_copy_file(file_path, project_data)
                yield (index + 1) / len(files), project_data

    def _run_copy_file(self, path: str, data: ProjectData) -> ProjectData:
        _LOGGER.info(
            (
                "adding data file to project_id {} and data_id {} from file path {}"
            ).format(self.project_id, data.data_id, path)
        )
        DataFromPathJobWorker._save_project_data(
            data, path, self._get_project_model().file_path
        )

        _LOGGER.info(
            (
                "added data file to project_id {} and data_id {} from file path {}"
            ).format(self.project_id, data.data_id, path)
        )
        return data


class DataFromPathJobWorker(_DataLoaderJobWorker):
    """
    A job worker for retrieving .npz data files from a given uri.
    The uri can be either a local file path or a public url.

    :param job_id: the id of the job this worker is running under
    :param project_id: the id of the project the worker is running for
    :param data_id: the id of the data the worker is running for
    :param uri: the uri to retrieve
    """

    def __init__(self, job_id: str, project_id: str, data_id: str, uri: str):
        super().__init__(job_id, project_id, data_id, uri)

    def run(self) -> Iterator[Dict[str, Any]]:
        """
        Perform the work for the job.
        Downloads the data files from a public url if the uri is a public url.
        Copies the data if the uri is accessible through the local file system.
        If the uri points to tar file, extract and save any additional data objects

        :return: an iterator containing progress update information
        """
        # Assert project model has been set
        self._get_project_model()
        try:
            if is_url(self.uri):
                for progress in self._run_download():
                    yield progress
            else:
                for progress in self._run_local():
                    yield progress
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise e

    def _run_download(self):
        _LOGGER.info(
            ("adding data file(s) to project_id {} from url {}").format(
                self.project_id, self.uri
            )
        )

        yield JobProgressSchema().dump(
            {"iter_indefinite": False, "iter_class": "download", "iter_val": 0.0}
        )

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
                        "iter_val": progress_val / 3 if progress_val else 0,
                        "step_class": "download",
                        "step_index": 0,
                    }
                )

            if tarfile.is_tarfile(temp_path):
                _LOGGER.info("Untarring file downloaded from {}".format(self.uri))
                yield JobProgressSchema().dump(
                    {
                        "iter_indefinite": False,
                        "iter_class": "download",
                        "iter_val": 1 / 3,
                        "step_class": "untarring",
                        "step_index": 1,
                    }
                )
                with TemporaryDirectory() as extract_path, tarfile.open(
                    temp_path, "r"
                ) as tar:
                    tar.extractall(extract_path)
                    yield JobProgressSchema().dump(
                        {
                            "iter_indefinite": False,
                            "iter_class": "download",
                            "iter_val": 2 / 3,
                            "step_class": "untarring",
                            "step_index": 1,
                        }
                    )
                    for progress, _ in self._run_copy_folder(extract_path):
                        yield JobProgressSchema().dump(
                            {
                                "iter_indefinite": False,
                                "iter_class": "download",
                                "iter_val": 2 / 3 + progress / 3,
                                "step_class": "copy_folder",
                                "step_index": 2,
                            }
                        )
            else:
                self._run_copy_file(temp_path, self._get_project_data())
                yield JobProgressSchema().dump(
                    {
                        "iter_indefinite": False,
                        "iter_class": "download",
                        "iter_val": 1,
                        "step_class": "copy",
                        "step_index": 1,
                    }
                )

    def _run_local(self):
        _LOGGER.info(
            ("adding data file(s) to project_id {} from file path {}").format(
                self.project_id, self.uri
            )
        )
        if os.path.isdir(self.uri):
            _LOGGER.info("Path {} is directory".format(self.uri))
            yield JobProgressSchema().dump(
                {"iter_indefinite": False, "iter_val": 0.0, "iter_class": "copy_folder"}
            )
            for progress, _ in self._run_copy_folder(self.uri):
                yield JobProgressSchema().dump(
                    {
                        "iter_indefinite": False,
                        "iter_val": progress,
                        "iter_class": "copy_folder",
                    }
                )
        else:
            _LOGGER.info("Path {} is file".format(self.uri))
            yield JobProgressSchema().dump(
                {"iter_indefinite": True, "iter_class": "copy"}
            )
            self._run_copy_file(self.uri, self._get_project_data())


class DataFromRepoJobWorker(_DataLoaderJobWorker):
    """
    A job worker for retrieving .npz data files from a given uri.
    The uri can be either a local file path or a public url.

    :param job_id: the id of the job this worker is running under
    :param project_id: the id of the project the worker is running for
    :param data_id: the id of the data the worker is running for
    :param uri: the uri to retrieve
    """

    def __init__(self, job_id: str, project_id: str, data_id: str, uri: str):
        super().__init__(job_id, project_id, data_id, uri)

    def run(self) -> Iterator[Dict[str, Any]]:
        _LOGGER.info(
            (
                "adding data file to project_id {} and " "data_id {} from model repo {}"
            ).format(self.project_id, self.data_id, self.uri)
        )

        # yield start progress to mark the expected flow
        yield JobProgressSchema().dump(
            {"iter_indefinite": False, "iter_class": "download_repo", "iter_val": 0.0}
        )

        try:
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
                            "iter_class": "download_repo",
                            "iter_val": progress_val / 3,
                            "step_index": 0,
                            "step_class": "download",
                        }
                    )

                with TemporaryDirectory() as extract_path, tarfile.open(
                    temp_path, "r"
                ) as tar:
                    tar.extractall(extract_path)
                    yield JobProgressSchema().dump(
                        {
                            "iter_indefinite": False,
                            "iter_class": "download_repo",
                            "iter_val": 2 / 3,
                            "step_class": "untarring",
                            "step_index": 1,
                        }
                    )
                    for progress, _ in self._run_copy_folder(extract_path):
                        yield JobProgressSchema().dump(
                            {
                                "iter_indefinite": False,
                                "iter_class": "download_repo",
                                "iter_val": 2 / 3 + progress / 3,
                                "step_class": "copy_folder",
                                "step_index": 2,
                            }
                        )

            _LOGGER.info(
                (
                    "added model file to project_id {} and "
                    "data_id {} from model repo {}"
                ).format(self.project_id, self.data_id, self.uri)
            )
        except Exception as e:
            raise e
