"""
Server routes related to the project's model routes
"""

from typing import Tuple
import logging
from http import HTTPStatus
from tempfile import NamedTemporaryFile, gettempdir
import os
import shutil

from marshmallow import ValidationError
from peewee import JOIN
from flask import Blueprint, request, jsonify, Response, send_file
from flasgger import swag_from


from neuralmagicML.onnx.utils import validate_onnx_file
from neuralmagicML.onnx.recal import ModelAnalyzer
from neuralmagicML.server.blueprints.helpers import (
    get_project_by_id,
    get_project_model_by_project_id,
    HTTPNotFoundError,
)
from neuralmagicML.server.blueprints.projects import PROJECTS_ROOT_PATH
from neuralmagicML.server.schemas import (
    data_dump_and_validation,
    ErrorSchema,
    ProjectModelAnalysisSchema,
    ResponseProjectModelAnalysisSchema,
    ResponseProjectModelSchema,
    ResponseProjectModelDeletedSchema,
    CreateUpdateProjectModelSchema,
    SetProjectModelFromSchema,
    DeleteProjectModelSchema,
)
from neuralmagicML.server.models import database, Project, ProjectModel, Job
from neuralmagicML.server.workers import (
    JobWorkerManager,
    ModelFromPathJobWorker,
    ModelFromRepoJobWorker,
)


__all__ = ["PROJECT_MODEL_PATH", "projects_model_blueprint"]


PROJECT_MODEL_PATH = "{}/<project_id>/model".format(PROJECTS_ROOT_PATH)

_LOGGER = logging.getLogger(__name__)

projects_model_blueprint = Blueprint(
    PROJECT_MODEL_PATH, __name__, url_prefix=PROJECT_MODEL_PATH
)


def _add_model_check(project_id: str) -> Project:
    project = get_project_by_id(project_id)
    project_model = get_project_model_by_project_id(project_id, raise_not_found=False)

    if project_model is not None:
        raise ValidationError(
            (
                "A model is already set for the project with id {}, "
                "can only have one model per project"
            ).format(project_id)
        )

    return project


@projects_model_blueprint.route("/upload", methods=["POST"])
@swag_from(
    {
        "tags": ["Projects Model"],
        "summary": "Upload a model file for the project.",
        "description": "ONNX files only currently. "
        "Will overwrite the current model file for the uploaded one on success.",
        "consumes": ["multipart/form-data"],
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to upload the model for",
                "required": True,
                "type": "string",
            },
            {
                "in": "formData",
                "name": "model_file",
                "description": "The ONNX model file",
                "required": True,
                "type": "file",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The saved project's model details",
                "schema": ResponseProjectModelSchema,
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
def upload_model(project_id: str) -> Tuple[Response, int]:
    """
    Route for uploading a model file to a project.
    Raises an HTTPNotFoundError if the project is not found in the database.

    :param project_id: the id of the project to upload the model for
    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info("uploading model for project {}".format(project_id))
    project = _add_model_check(project_id)

    if "model_file" not in request.files:
        _LOGGER.error("missing uploaded file 'model_file'")
        raise ValidationError("missing uploaded file 'model_file'")

    model_file = request.files["model_file"]

    with NamedTemporaryFile() as temp:
        # Verify onnx model is valid and contains opset field
        tempname = os.path.join(gettempdir(), temp.name)
        model_file.save(temp)
        validate_onnx_file(tempname)

        with database.atomic() as transaction:
            try:
                # Create project model
                data = CreateUpdateProjectModelSchema().dump(
                    {"file": "model.onnx", "source": "uploaded"}
                )
                project_model = ProjectModel.create(project=project, **data)
                project_model.setup_filesystem()
                shutil.copy(tempname, project_model.file_path)
                project_model.validate_filesystem()
            except Exception as err:
                _LOGGER.error(
                    "error while creating new project model, rolling back: {}".format(
                        err
                    )
                )
                transaction.rollback()
                raise err

        resp_model = data_dump_and_validation(
            ResponseProjectModelSchema(), {"model": project_model}
        )
        _LOGGER.info("created project model {}".format(resp_model))

        return jsonify(resp_model), HTTPStatus.OK.value


@projects_model_blueprint.route("/upload-from-path", methods=["POST"])
@swag_from(
    {
        "tags": ["Projects Model"],
        "summary": "Set a model file for the project, "
        "from a given path (on server or url).",
        "description": "ONNX files only currently. "
        "Creates a background job to do this, pull the status using the jobs api "
        "for the returned file_source_job_id.",
        "consumes": ["application/json"],
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to set the model for",
                "required": True,
                "type": "string",
            },
            {
                "in": "body",
                "name": "body",
                "description": "The data for the model to add from a path",
                "required": True,
                "schema": SetProjectModelFromSchema,
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The saved project's model details, "
                "contains the id for the job that was kicked off",
                "schema": ResponseProjectModelSchema,
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
def load_model_from_path(project_id: str) -> Tuple[Response, int]:
    """
    Route for loading a model for a project from a given uri path;
    either public url or from the local file system.
    Starts a background job in the JobWorker setup to run.
    The state of the job can be checked after.
    Raises an HTTPNotFoundError if the project is not found in the database.

    :param project_id: the id of the project to load the model for
    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info(
        "loading model from path for project {} for request json {}".format(
            project_id, request.json
        )
    )
    project = _add_model_check(project_id)
    data = SetProjectModelFromSchema().load(request.get_json(force=True))

    with database.atomic() as transaction:
        try:
            project_model = ProjectModel.create(
                project=project, source="downloaded_path", job=None
            )
            job = Job.create(
                project=project,
                type_=ModelFromPathJobWorker.get_type(),
                worker_args=ModelFromPathJobWorker.format_args(
                    model_id=project_model.model_id, uri=data["uri"],
                ),
            )
            project_model.job = job
            project_model.save()
            project_model.setup_filesystem()
            project_model.validate_filesystem()
        except Exception as err:
            _LOGGER.error(
                "error while creating new project model, rolling back: {}".format(err)
            )
            transaction.rollback()
            raise err

    # call into JobWorkerManager to kick off job if it's not already running
    JobWorkerManager().refresh()

    resp_model = data_dump_and_validation(
        ResponseProjectModelSchema(), {"model": project_model}
    )
    _LOGGER.info("created project model from path {}".format(resp_model))

    return jsonify(resp_model), HTTPStatus.OK.value


@projects_model_blueprint.route("/upload-from-repo", methods=["POST"])
@swag_from(
    {
        "tags": ["Projects Model"],
        "summary": "Set a model file for the project, "
        "from the Neural Magic Model Repo.",
        "description": "ONNX files only currently. "
        "Creates a background job to do this, pull the status using the jobs api "
        "for the returned file_source_job_id.",
        "consumes": ["application/json"],
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to set the model for",
                "required": True,
                "type": "string",
            },
            {
                "in": "body",
                "name": "body",
                "description": "The data for the model to add from model repo",
                "required": True,
                "schema": SetProjectModelFromSchema,
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The saved project's model details, "
                "contains the id for the job that was kicked off",
                "schema": ResponseProjectModelSchema,
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
def load_model_from_repo(project_id: str) -> Tuple[Response, int]:
    """
    Route for loading a model for a project from the Neural Magic model repo.
    Starts a background job in the JobWorker setup to run.
    The state of the job can be checked after.
    Raises an HTTPNotFoundError if the project is not found in the database.

    :param project_id: the id of the project to load the model for
    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info(
        "loading model from repo for project {} for request json {}".format(
            project_id, request.json
        )
    )
    project = _add_model_check(project_id)
    data = SetProjectModelFromSchema().load(request.get_json(force=True))

    with database.atomic() as transaction:
        try:
            project_model = ProjectModel.create(
                project=project, source="downloaded_repo", job=None
            )
            job = Job.create(
                project=project,
                type_=ModelFromRepoJobWorker.get_type(),
                worker_args=ModelFromRepoJobWorker.format_args(
                    model_id=project_model.model_id, uri=data["uri"],
                ),
            )
            project_model.job = job
            project_model.save()
            project_model.setup_filesystem()
            project_model.validate_filesystem()
        except Exception as err:
            _LOGGER.error(
                "error while creating new project model, rolling back: {}".format(err)
            )
            transaction.rollback()
            raise err

    # call into JobWorkerManager to kick off job if it's not already running
    JobWorkerManager().refresh()

    resp_model = data_dump_and_validation(
        ResponseProjectModelSchema(), {"model": project_model}
    )
    _LOGGER.info("created project model from repo {}".format(resp_model))

    return jsonify(resp_model), HTTPStatus.OK.value


@projects_model_blueprint.route("/")
@swag_from(
    {
        "tags": ["Projects Model"],
        "summary": "Get the model details for a project",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to set the model for",
                "required": True,
                "type": "string",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The project's model details",
                "schema": ResponseProjectModelSchema,
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
def get_model_details(project_id: str) -> Tuple[Response, int]:
    """
    Route to get the model details for a given project matching the project_id.
    Raises an HTTPNotFoundError if the project is not found in the database
    or a model doesn't exit.

    :param project_id: the project_id to get the model details for
    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info("getting the model details for project_id {}".format(project_id))
    project = get_project_by_id(project_id)
    query = (
        ProjectModel.select(ProjectModel, Job)
        .join_from(ProjectModel, Job, JOIN.LEFT_OUTER)
        .where(ProjectModel.project == project)
    )
    project_model = None

    for res in query:
        project_model = res

    if project_model is None:
        raise HTTPNotFoundError(
            "could not find model for project_id {}".format(project_id)
        )

    resp_model = data_dump_and_validation(
        ResponseProjectModelSchema(), {"model": project_model}
    )
    _LOGGER.info("retrieved project model details {}".format(resp_model))

    return jsonify(resp_model), HTTPStatus.OK.value


@projects_model_blueprint.route("/file")
@swag_from(
    {
        "tags": ["Projects Model"],
        "summary": "Get the model file for a project",
        "produces": ["application/octet-stream"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to set the model for",
                "required": True,
                "type": "string",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The project's model file",
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
def get_model_file(project_id: str) -> Tuple[Response, int]:
    """
    Route to get the model details for a given project matching the project_id.
    Raises an HTTPNotFoundError if the project is not found in the database
    or a model doesn't exit.

    :param project_id: the project_id to get the model details for
    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info("getting the model file for project_id {}".format(project_id))
    project = get_project_by_id(project_id)
    project_model = get_project_model_by_project_id(project_id)
    project_model.validate_filesystem()
    _LOGGER.info("sending project model file from {}".format(project_model.file_path))

    return send_file(project_model.file_path, mimetype="application/octet-stream")


@projects_model_blueprint.route("/", methods=["DELETE"])
@swag_from(
    {
        "tags": ["Projects Model"],
        "summary": "Delete the model for a project",
        "produces": ["application/octet-stream"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to set the model for",
                "required": True,
                "type": "string",
            },
            {
                "in": "query",
                "name": "force",
                "type": "boolean",
                "description": "True to force the deletion and not error out, "
                "False otherwise. Default False",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "Successfully deleted",
                "schema": ResponseProjectModelDeletedSchema,
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
def delete_model(project_id: str) -> Tuple[Response, int]:
    """
    Route to delete the model for a given project matching the project_id.
    Raises an HTTPNotFoundError if the project is not found in the database
    or a model doesn't exit.

    :param project_id: the project_id to delete the model for
    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info("deleting the model for project_id {}".format(project_id))
    args = DeleteProjectModelSchema().load(
        {key: val for key, val in request.args.items()}
    )
    project = get_project_by_id(project_id)
    project_model = get_project_model_by_project_id(project_id)
    model_id = project_model.model_id

    with database.atomic() as transaction:
        try:
            project_model.delete_instance()
            project_model.delete_filesystem()
        except Exception as err:
            _LOGGER.error(
                "error while deleting project model for {}, rolling back: {}".format(
                    project_id, err
                )
            )

            if not args["force"]:
                transaction.rollback()
                raise err

    resp_deleted = data_dump_and_validation(
        ResponseProjectModelDeletedSchema(),
        {"project_id": project_id, "model_id": model_id},
    )
    _LOGGER.info(
        "deleted model for project_id {} and model_id {} from path {}".format(
            project_id, project_model.model_id, project_model.dir_path
        )
    )

    return jsonify(resp_deleted), HTTPStatus.OK.value


@projects_model_blueprint.route("/analysis", methods=["POST"])
@swag_from(
    {
        "tags": ["Projects Model"],
        "summary": "Create an analysis file of the model for the project",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to create a model analysis for",
                "required": True,
                "type": "string",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "Successfully created the analysis",
                "schema": ResponseProjectModelAnalysisSchema,
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
def create_analysis(project_id: str) -> Tuple[Response, int]:
    """
    Route for creating a model analysis for a given project matching the project_id.
    If one exists, will overwrite the previous.
    Raises an HTTPNotFoundError if the project is not found in the database
    or a model doesn't exit.

    :param project_id: the project_id to create the analysis for
    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info("creating analysis for project_id {} model".format(project_id))
    project = get_project_by_id(project_id)
    project_model = get_project_model_by_project_id(project_id)
    project_model.validate_filesystem()
    analyzer = ModelAnalyzer(project_model.file_path)
    analysis = ProjectModelAnalysisSchema().load(analyzer.dict())
    project_model.analysis = analysis
    project_model.save()

    resp_analysis = data_dump_and_validation(
        ResponseProjectModelAnalysisSchema(), {"analysis": analysis}
    )
    _LOGGER.info(
        "analyzed model for project_id {} and model_id {} from path {}".format(
            project_id, project_model.model_id, project_model.file_path
        )
    )

    return jsonify(resp_analysis), HTTPStatus.OK.value


@projects_model_blueprint.route("/analysis")
@swag_from(
    {
        "tags": ["Projects Model"],
        "summary": "Get the analysis file of the model for the project",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to get a model analysis for",
                "required": True,
                "type": "string",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The analysis for the project's model",
                "schema": ResponseProjectModelAnalysisSchema,
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
def get_analysis(project_id: str) -> Tuple[Response, int]:
    """
    Route for getting the model analysis for a given project matching the project_id.
    Raises an HTTPNotFoundError if the project is not found in the database
    or a model doesn't exit.

    :param project_id: the project_id to get the analysis for
    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info("getting model analysis for project_id {}".format(project_id))
    project = get_project_by_id(project_id)
    project_model = get_project_model_by_project_id(project_id)
    analysis = (
        ProjectModelAnalysisSchema().dump(project_model.analysis)
        if project_model.analysis
        else None
    )

    if analysis is None:
        raise ValidationError("analysis must be created first")

    resp_analysis = data_dump_and_validation(
        ResponseProjectModelAnalysisSchema(), {"analysis": analysis}
    )
    _LOGGER.info(
        "retrieved model analysis for project_id {} and model_id {}".format(
            project_id, project_model.model_id
        )
    )

    return jsonify(resp_analysis), HTTPStatus.OK.value


@projects_model_blueprint.route("/analysis", methods=["DELETE"])
@swag_from(
    {
        "tags": ["Projects Model"],
        "summary": "Delete an analysis file of the model for the project",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to delete a model analysis for",
                "required": True,
                "type": "string",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "Successfully deleted the analysis",
                "schema": ResponseProjectModelDeletedSchema,
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
def delete_analysis(project_id: str) -> Tuple[Response, int]:
    """
    Route for deleting the model analysis for a given project matching the project_id.
    Raises an HTTPNotFoundError if the project is not found in the database
    or a model doesn't exit.

    :param project_id: the project_id to delete the analysis for
    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info("deleting analysis for project_id {} model".format(project_id))
    project = get_project_by_id(project_id)
    project_model = get_project_model_by_project_id(project_id)
    project_model.analysis = None
    project_model.save()

    resp_deleted = data_dump_and_validation(
        ResponseProjectModelDeletedSchema(),
        {"project_id": project_id, "model_id": project_model.model_id},
    )
    _LOGGER.info(
        "deleted model analysis for project_id {} and model_id {} from path {}".format(
            project_id, project_model.model_id, project_model.file_path
        )
    )

    return jsonify(resp_deleted), HTTPStatus.OK.value
