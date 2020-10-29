"""
Server routes related to projects data files
"""

import logging
from http import HTTPStatus
from tempfile import NamedTemporaryFile, gettempdir
import os
import shutil

from flask import Blueprint, current_app, request, jsonify, send_file
from flasgger import swag_from

from marshmallow import ValidationError

from neuralmagicML.server.blueprints.projects import PROJECTS_ROOT_PATH
from neuralmagicML.server.blueprints.utils import (
    get_project_by_id,
    get_project_model_by_project_id,
    get_project_data_by_ids,
    validate_model_data,
)
from neuralmagicML.server.models import database, Project, ProjectData, Job
from neuralmagicML.server.schemas import (
    data_dump_and_validation,
    ErrorSchema,
    ResponseProjectDataSingleSchema,
    ResponseProjectDataSchema,
    ResponseProjectDataDeletedSchema,
    SetProjectDataFromSchema,
    CreateUpdateProjectDataSchema,
    SearchProjectDataSchema,
)
from neuralmagicML.server.workers import (
    JobWorkerManager,
    DataFromPathJobWorker,
    DataFromRepoJobWorker,
)

__all__ = ["PROJECT_DATA_PATH", "projects_data_blueprint"]


PROJECT_DATA_PATH = "{}/<project_id>/data".format(PROJECTS_ROOT_PATH)

_LOGGER = logging.getLogger(__name__)

projects_data_blueprint = Blueprint(
    PROJECT_DATA_PATH, __name__, url_prefix=PROJECT_DATA_PATH
)


@projects_data_blueprint.route("/upload", methods=["POST"])
@swag_from(
    {
        "tags": ["Projects Data"],
        "summary": "Upload a new input data file for the project. ",
        "description": "Numpy files or collections of numpy files only currently. "
        "Model file must already be uploaded. ",
        "consumes": ["multipart/form-data"],
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to upload the data for",
                "required": True,
                "type": "string",
            },
            {
                "in": "formData",
                "name": "data_file",
                "description": "The numpy data file",
                "required": True,
                "type": "file",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The saved project's data details",
                "schema": ResponseProjectDataSingleSchema,
            },
            HTTPStatus.BAD_REQUEST.value: {
                "description": "Information for the error that occurred",
                "schema": ErrorSchema,
            },
            HTTPStatus.NOT_FOUND.value: {
                "description": "Information for the error that occurred",
                "schema": ErrorSchema,
            },
            HTTPStatus.INTERNAL_SERVER_ERROR.value: {
                "description": "Information for the error that occurred",
                "schema": ErrorSchema,
            },
        },
    },
)
def upload_data(project_id: str):
    """
    Route for uploading a data file to a project.
    Raises an HTTPNotFoundError if the project is not found in the database.

    :param project_id: the id of the project to upload the data for
    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info("uploading data file for project {}".format(project_id))
    project = get_project_by_id(project_id)
    project_model = get_project_model_by_project_id(project_id)

    if "data_file" not in request.files:
        _LOGGER.error("missing uploaded file 'data_file'")
        raise ValidationError("missing uploaded file 'data_file'")

    data_file = request.files["data_file"]

    with NamedTemporaryFile() as temp:
        data_path = gettempdir()
        tempname = os.path.join(data_path, temp.name)
        data_file.save(tempname)

        try:
            _LOGGER.info(project_model.file_path)

            validate_model_data(os.path.join(data_path, "*"), project_model.file_path)

            data = CreateUpdateProjectDataSchema().dump(
                {"source": "uploaded", "job": None}
            )
            project_data = ProjectData.create(project=project, **data)
            project_data.file = "{}.npz".format(project_data)
            project_data.setup_filesystem()
            shutil.copy(tempname, project_data.file_path)

            project_data.validate_filesystem()
            validate_model_data(project_data.file_path, project_model.file_path)
            project_data.save()
        except Exception as err:
            if project_data:
                try:
                    os.remove(project_data.file_path)
                except OSError as err:
                    pass

                try:
                    project_data.delete_instance()
                except Exception as rollback_Err:
                    _LOGGER.error(
                        "error while rolling back new data: {}".format(rollback_err)
                    )

            _LOGGER.error(
                "error while creating new project data, rolling back: {}".format(err)
            )
            raise err

    resp_data = data_dump_and_validation(
        ResponseProjectDataSingleSchema(), {"data": project_data}
    )
    _LOGGER.info("created project data {}".format(resp_data))

    return jsonify(resp_data), HTTPStatus.OK.value


@projects_data_blueprint.route("/upload-from-path", methods=["POST"])
@swag_from(
    {
        "tags": ["Projects Data"],
        "summary": "Set the input data files for the project, "
        "from a given path (on server or url).",
        "description": "Numpy files or collections of numpy files only currently. "
        "Model file must already be uploaded. "
        "Creates a background job to do this, pull the status using the jobs api "
        "for the returned file_source_job_id.",
        "consumes": ["application/json"],
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to set the data for",
                "required": True,
                "type": "string",
            },
            {
                "in": "body",
                "name": "body",
                "description": "The data for the data to add from a path",
                "required": True,
                "schema": SetProjectDataFromSchema,
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The saved project's data details, "
                "contains the id for the job that was kicked off",
                "schema": ResponseProjectDataSingleSchema,
            },
            HTTPStatus.BAD_REQUEST.value: {
                "description": "Information for the error that occurred",
                "schema": ErrorSchema,
            },
            HTTPStatus.NOT_FOUND.value: {
                "description": "Information for the error that occurred",
                "schema": ErrorSchema,
            },
            HTTPStatus.INTERNAL_SERVER_ERROR.value: {
                "description": "Information for the error that occurred",
                "schema": ErrorSchema,
            },
        },
    },
)
def load_data_from_path(project_id: str):
    """
    Route for loading data file(s) for a project from a given uri path;
    either public url or from the local file system.
    Starts a background job in the JobWorker setup to run.
    The state of the job can be checked after.
    Raises an HTTPNotFoundError if the project is not found in the database.

    :param project_id: the id of the project to load the data for
    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info(
        "loading data from path for project {} for request json {}".format(
            project_id, request.json
        )
    )
    project = get_project_by_id(project_id)
    project_model = get_project_model_by_project_id(project_id)
    data = SetProjectDataFromSchema().load(request.get_json(force=True))

    try:
        project_data = ProjectData.create(
            project=project, source="downloaded_path", job=None
        )
        job = Job.create(
            project_id=project.project_id,
            type_=DataFromPathJobWorker.get_type(),
            worker_args=DataFromPathJobWorker.format_args(
                data_id=project_data.data_id, uri=data["uri"]
            ),
        )
        project_data.job = job
        project_data.save()
        project_data.setup_filesystem()
        project_data.validate_filesystem()
    except Exception as err:
        if project_data:
            try:
                os.remove(project_data.file_path)
            except OSError as err:
                pass

            try:
                project_data.delete_instance()
            except Exception as rollback_Err:
                _LOGGER.error(
                    "error while rolling back new data: {}".format(rollback_err)
                )

        if job:
            try:
                job.delete_instance()
            except Exception as rollback_Err:
                _LOGGER.error(
                    "error while rolling back new data: {}".format(rollback_err)
                )

        _LOGGER.error(
            "error while creating new project data, rolling back: {}".format(err)
        )
        raise err

    # call into JobWorkerManager to kick off job if it's not already running
    JobWorkerManager().refresh()

    resp_data = data_dump_and_validation(
        ResponseProjectDataSingleSchema(), {"data": project_data}
    )
    _LOGGER.info("created project data from path {}".format(resp_data))

    return jsonify(resp_data), HTTPStatus.OK.value


@projects_data_blueprint.route("/upload-from-repo", methods=["POST"])
@swag_from(
    {
        "tags": ["Projects Data"],
        "summary": "Set the input data files for the project, "
        "from the Neural Magic Model Repo.",
        "description": "Numpy files or collections of numpy files only currently. "
        "Model file must already be uploaded. "
        "Creates a background job to do this, pull the status using the jobs api "
        "for the returned file_source_job_id.",
        "consumes": ["application/json"],
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to set the data for",
                "required": True,
                "type": "string",
            },
            {
                "in": "body",
                "name": "body",
                "description": "The data for the data to add from data repo",
                "required": True,
                "schema": SetProjectDataFromSchema,
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The saved project's data details, "
                "contains the id for the job that was kicked off",
                "schema": ResponseProjectDataSchema,
            },
            HTTPStatus.BAD_REQUEST.value: {
                "description": "Information for the error that occurred",
                "schema": ErrorSchema,
            },
            HTTPStatus.NOT_FOUND.value: {
                "description": "Information for the error that occurred",
                "schema": ErrorSchema,
            },
            HTTPStatus.INTERNAL_SERVER_ERROR.value: {
                "description": "Information for the error that occurred",
                "schema": ErrorSchema,
            },
        },
    },
)
def load_data_from_repo(project_id: str):
    """
    Route for loading data file(s) for a project from the Neural Magic model repo.
    Starts a background job in the JobWorker setup to run.
    The state of the job can be checked after.
    Raises an HTTPNotFoundError if the project is not found in the database.

    :param project_id: the id of the project to load the data for
    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info(
        "loading data from repo for project {} for request json {}".format(
            project_id, request.json
        )
    )
    project = get_project_by_id(project_id)
    project_model = get_project_model_by_project_id(project_id)
    data = SetProjectDataFromSchema().load(request.get_json(force=True))

    try:
        project_data = ProjectData.create(
            project=project, source="downloaded_path", job=None
        )
        job = Job.create(
            project_id=project.project_id,
            type_=DataFromRepoJobWorker.get_type(),
            worker_args=DataFromRepoJobWorker.format_args(
                data_id=project_data.data_id, uri=data["uri"]
            ),
        )
        project_data.job = job
        project_data.save()
        project_data.setup_filesystem()
        project_data.validate_filesystem()
    except Exception as err:
        if project_data:
            try:
                os.remove(project_data.file_path)
            except OSError as err:
                pass

            try:
                project_data.delete_instance()
            except Exception as rollback_Err:
                _LOGGER.error(
                    "error while rolling back new data: {}".format(rollback_err)
                )

        if job:
            try:
                job.delete_instance()
            except Exception as rollback_Err:
                _LOGGER.error(
                    "error while rolling back new data: {}".format(rollback_err)
                )

        _LOGGER.error(
            "error while creating new project data, rolling back: {}".format(err)
        )
        transaction.rollback()
        raise err

    # call into JobWorkerManager to kick off job if it's not already running
    JobWorkerManager().refresh()

    resp_data = data_dump_and_validation(
        ResponseProjectDataSingleSchema(), {"data": project_data}
    )
    _LOGGER.info("created project data from path {}".format(resp_data))

    return jsonify(resp_data), HTTPStatus.OK.value


@projects_data_blueprint.route("/")
@swag_from(
    {
        "tags": ["Projects Data"],
        "summary": "Get a list of input data items for a project",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to get the data for",
                "required": True,
                "type": "string",
            },
            {
                "in": "query",
                "name": "page",
                "type": "integer",
                "description": "The page (one indexed) to get of the project data. "
                "Default 1",
            },
            {
                "in": "query",
                "name": "page_length",
                "type": "integer",
                "description": "The length of the page to get (number of project data). "
                "Default 20",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The project's data details",
                "schema": ResponseProjectDataSchema,
            },
            HTTPStatus.BAD_REQUEST.value: {
                "description": "Information for the error that occurred",
                "schema": ErrorSchema,
            },
            HTTPStatus.NOT_FOUND.value: {
                "description": "Information for the error that occurred",
                "schema": ErrorSchema,
            },
            HTTPStatus.INTERNAL_SERVER_ERROR.value: {
                "description": "Information for the error that occurred",
                "schema": ErrorSchema,
            },
        },
    },
)
def get_data_details(project_id: str):
    """
    Route to get the details for all data for a given project matching the project_id.
    Raises an HTTPNotFoundError if the project is not found in the database.

    :param project_id: the project_id to get the data details for
    :return: a tuple containing (json response, http status code)
    """
    args = {key: val for key, val in request.args.items()}
    _LOGGER.info(
        "getting all the data for project_id {} and request args {}".format(
            project_id, args
        )
    )

    args = SearchProjectDataSchema().load(args)

    # Validate project and model
    project = get_project_by_id(project_id)
    project_model = get_project_model_by_project_id(project_id)

    project_data = (
        ProjectData.select()
        .where(ProjectData.project_id == project_id)
        .group_by(ProjectData)
        .order_by(ProjectData.created)
        .paginate(args["page"], args["page_length"])
    )

    resp_data = data_dump_and_validation(
        ResponseProjectDataSchema(), {"data": project_data}
    )
    _LOGGER.info("sending project data {}".format(resp_data))

    return jsonify(resp_data), HTTPStatus.OK.value


@projects_data_blueprint.route("/<data_id>")
@swag_from(
    {
        "tags": ["Projects Data"],
        "summary": "Get an input data item for a project",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to get the data for",
                "required": True,
                "type": "string",
            },
            {
                "in": "path",
                "name": "data_id",
                "description": "ID of the data to get",
                "required": True,
                "type": "string",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The project's data details",
                "schema": ResponseProjectDataSingleSchema,
            },
            HTTPStatus.BAD_REQUEST.value: {
                "description": "Information for the error that occurred",
                "schema": ErrorSchema,
            },
            HTTPStatus.NOT_FOUND.value: {
                "description": "Information for the error that occurred",
                "schema": ErrorSchema,
            },
            HTTPStatus.INTERNAL_SERVER_ERROR.value: {
                "description": "Information for the error that occurred",
                "schema": ErrorSchema,
            },
        },
    },
)
def get_data_single_details(project_id: str, data_id: str):
    """
    Route to get the details for all data for a given project matching the project_id.
    Raises an HTTPNotFoundError if the project or data is not found in the database.

    :param project_id: the project_id to get the data details for
    :param data_id: the data_id to get the data details for
    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info(
        "getting the data with data_id {} for project_id {}".format(data_id, project_id)
    )
    project = get_project_by_id(project_id)
    project_model = get_project_model_by_project_id(project_id)
    project_data = get_project_data_by_ids(project_id, data_id)

    resp_data = data_dump_and_validation(
        ResponseProjectDataSingleSchema(), {"data": project_data}
    )
    _LOGGER.info("sending project data from {}".format(project_data.file_path))

    return jsonify(resp_data), HTTPStatus.OK.value


@projects_data_blueprint.route("/<data_id>", methods=["DELETE"])
@swag_from(
    {
        "tags": ["Projects Data"],
        "summary": "Delete an input data item for a project",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to delete the data for",
                "required": True,
                "type": "string",
            },
            {
                "in": "path",
                "name": "data_id",
                "description": "ID of the data to delete",
                "required": True,
                "type": "string",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The project's deleted data details",
                "schema": ResponseProjectDataDeletedSchema,
            },
            HTTPStatus.BAD_REQUEST.value: {
                "description": "Information for the error that occurred",
                "schema": ErrorSchema,
            },
            HTTPStatus.NOT_FOUND.value: {
                "description": "Information for the error that occurred",
                "schema": ErrorSchema,
            },
            HTTPStatus.INTERNAL_SERVER_ERROR.value: {
                "description": "Information for the error that occurred",
                "schema": ErrorSchema,
            },
        },
    },
)
def delete_data(project_id: str, data_id: str):
    """
    Route to delete a data file for a given project matching the project_id.
    Raises an HTTPNotFoundError if the project or data is not found in the database.

    :param project_id: the project_id to get the model details for
    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info(
        "deleting data with data_id {} for project_id {}".format(data_id, project_id)
    )
    project = get_project_by_id(project_id)
    project_model = get_project_model_by_project_id(project_id)
    project_data = get_project_data_by_ids(project_id, data_id)

    try:
        project_data.delete_instance()
        project_data.delete_filesystem()
    except Exception as err:
        _LOGGER.error(
            "error while deleting project data for {}, rolling back: {}".format(
                data_id, err
            )
        )

        if not args["force"]:
            raise err

    resp_deleted = data_dump_and_validation(
        ResponseProjectDataDeletedSchema(),
        {"project_id": project_id, "data_id": data_id},
    )
    _LOGGER.info(
        "deleted data for project_id {} and data_id {} from path {}".format(
            project_id, project_data.data_id, project_data.dir_path
        )
    )

    return jsonify(resp_deleted), HTTPStatus.OK.value


@projects_data_blueprint.route("<data_id>/file")
@swag_from(
    {
        "tags": ["Projects Data"],
        "summary": "Get an input data file for a project",
        "produces": ["application/octet-stream"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to set the data for",
                "required": True,
                "type": "string",
            },
            {
                "in": "path",
                "name": "data_id",
                "description": "ID of the data to get",
                "required": True,
                "type": "string",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The project's data file",
                "content": {"application/octet-stream": {}},
            },
            HTTPStatus.BAD_REQUEST.value: {
                "description": "Information for the error that occurred",
                "schema": ErrorSchema,
            },
            HTTPStatus.NOT_FOUND.value: {
                "description": "Information for the error that occurred",
                "schema": ErrorSchema,
            },
            HTTPStatus.INTERNAL_SERVER_ERROR.value: {
                "description": "Information for the error that occurred",
                "schema": ErrorSchema,
            },
        },
    },
)
def get_data_file(project_id: str, data_id: str):
    """
    Route to get the data file for a given project matching the project_id.
    Raises an HTTPNotFoundError if the project or data is not found in the database.

    :param project_id: the project_id to get the data file for
    :param data_id: the data_id to get the data file for
    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info(
        "getting the data file for project_id {} with data_id {}".format(
            project_id, data_id
        )
    )
    project = get_project_by_id(project_id)
    project_data = get_project_data_by_ids(project_id, data_id)
    project_data.validate_filesystem()
    _LOGGER.info("sending project data file from {}".format(project_data.file_path))

    return send_file(project_data.file_path, mimetype="application/octet-stream")
