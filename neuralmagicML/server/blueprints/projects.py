"""
Server routes related to projects
"""

import logging
from http import HTTPStatus

from flask import Blueprint, request, jsonify
from flasgger import swag_from

from neuralmagicML.server.models import database, Project
from neuralmagicML.server.schemas import (
    data_dump_and_validation,
    ErrorSchema,
    ResponseProjectSchema,
    ResponseProjectExtSchema,
    ResponseProjectsSchema,
    ResponseProjectDeletedSchema,
    SearchProjectsSchema,
    CreateUpdateProjectSchema,
    DeleteProjectSchema,
)
from neuralmagicML.server.blueprints.utils import API_ROOT_PATH, get_project_by_id


__all__ = ["PROJECTS_ROOT_PATH", "projects_blueprint"]


PROJECTS_ROOT_PATH = "{}/projects".format(API_ROOT_PATH)

_LOGGER = logging.getLogger(__name__)

projects_blueprint = Blueprint(
    PROJECTS_ROOT_PATH, __name__, url_prefix=PROJECTS_ROOT_PATH
)


@projects_blueprint.route("/")
@swag_from(
    {
        "tags": ["Projects"],
        "summary": "Get a list of projects currently available",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "query",
                "name": "order_by",
                "type": "string",
                "enum": ["name", "created", "modified"],
                "description": "The field to order the projects by in the response. "
                "Default modified",
            },
            {
                "in": "query",
                "name": "order_desc",
                "type": "boolean",
                "description": "True to order the projects in descending order, "
                "False otherwise. Default True",
            },
            {
                "in": "query",
                "name": "page",
                "type": "integer",
                "description": "The page (one indexed) to get of the projects. "
                "Default 1",
            },
            {
                "in": "query",
                "name": "page_length",
                "type": "integer",
                "description": "The length of the page to get (number of projects). "
                "Default 20",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The requested projects",
                "schema": ResponseProjectsSchema,
            },
            HTTPStatus.BAD_REQUEST.value: {
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
def get_projects():
    """
    Route for getting a list of projects filtered by the flask request args

    :return: a tuple containing (json response, http status code)
    """
    args = {key: val for key, val in request.args.items()}
    _LOGGER.info("getting projects for request args {}".format(args))
    args = SearchProjectsSchema().load(args)
    order_by = getattr(Project, args["order_by"])
    query = (
        Project.select()
        .order_by(order_by if not args["order_desc"] else order_by.desc())
        .paginate(args["page"], args["page_length"])
    )
    projects = [res for res in query]
    resp_projects = data_dump_and_validation(
        ResponseProjectsSchema(), {"projects": projects}
    )
    _LOGGER.info("retrieved {} projects".format(len(resp_projects)))

    return jsonify(resp_projects), HTTPStatus.OK.value


@projects_blueprint.route("/", methods=["POST"])
@swag_from(
    {
        "tags": ["Projects"],
        "summary": "Create a new project",
        "consumes": ["application/json"],
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "body",
                "name": "body",
                "description": "The project to create",
                "required": True,
                "schema": CreateUpdateProjectSchema,
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The created project",
                "schema": ResponseProjectSchema,
            },
            HTTPStatus.BAD_REQUEST.value: {
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
def create_project():
    """
    Route for creating a new project with the given flask request arguments

    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info("creating project for request json {}".format(request.json))
    data = CreateUpdateProjectSchema().load(request.get_json(force=True))
    project = None

    try:
        project = Project.create(**data)
        project.setup_filesystem()
        project.validate_filesystem()
    except Exception as err:
        _LOGGER.error("error while creating new project, rolling back: {}".format(err))
        if project:
            try:
                project.delete_instance()
            except Exception as rollback_err:
                _LOGGER.error(
                    "error while rolling back new project: {}".format(rollback_err)
                )
        raise err

    resp_project = data_dump_and_validation(
        ResponseProjectSchema(), {"project": project}
    )
    _LOGGER.info("created project {}".format(resp_project))

    return jsonify(resp_project), HTTPStatus.OK.value


@projects_blueprint.route("/<project_id>")
@swag_from(
    {
        "tags": ["Projects"],
        "summary": "Get a project",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to return",
                "required": True,
                "type": "string",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The matching project",
                "schema": ResponseProjectExtSchema,
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
def get_project(project_id: str):
    """
    Route for getting a project matching the given project_id.
    Additionally includes attached model and data if available.
    Raises an HTTPNotFoundError if the job is not found in the database.

    :param project_id: the id of the project to get
    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info("getting project {}".format(project_id))
    project = get_project_by_id(project_id)

    # run file system validations to figure out if there are any errors in setup
    project.validate_filesystem()
    project.model = None

    if project.models:
        for model in project.models:
            project.model = model
            model.validate_filesystem()
            break

    if project.data:
        for data in project.data:
            data.validate_filesystem()

    resp_project = data_dump_and_validation(
        ResponseProjectExtSchema(), {"project": project}
    )
    _LOGGER.info("retrieved project {}".format(resp_project))

    return jsonify(resp_project), HTTPStatus.OK.value


@projects_blueprint.route("/<project_id>", methods=["PUT"])
@swag_from(
    {
        "tags": ["Projects"],
        "summary": "Update a project",
        "consumes": ["application/json"],
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to update",
                "required": True,
                "type": "string",
            },
            {
                "in": "body",
                "name": "body",
                "description": "The project data to update with",
                "required": True,
                "schema": CreateUpdateProjectSchema,
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The updated project",
                "schema": ResponseProjectExtSchema,
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
def update_project(project_id: str):
    """
    Route for updating a project with the given flask request arguments.
    Raises an HTTPNotFoundError if the job is not found in the database.

    :param project_id: the id of the project to get
    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info(
        "updating project {} for request json {}".format(project_id, request.json)
    )
    data = CreateUpdateProjectSchema().load(request.get_json(force=True))
    project = get_project_by_id(project_id)

    for key, val in data.items():
        setattr(project, key, val)

    project.save()
    resp_project = data_dump_and_validation(
        ResponseProjectExtSchema(), {"project": project}
    )
    _LOGGER.info("updated project {} with {}".format(project_id, data))

    return jsonify(resp_project), HTTPStatus.OK.value


@projects_blueprint.route("/<project_id>", methods=["DELETE"])
@swag_from(
    {
        "tags": ["Projects"],
        "summary": "Delete a project",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to delete",
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
                "description": "Deleted the project",
                "schema": ResponseProjectDeletedSchema,
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
def delete_project(project_id: str):
    """
    Route for deleting the project with the given project_id.
    Raises an HTTPNotFoundError if the job is not found in the database.

    :param project_id: the id of the project to get
    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info(
        "deleting project {} with request args {}".format(project_id, request.args)
    )
    args = DeleteProjectSchema().load({key: val for key, val in request.args.items()})
    project = get_project_by_id(project_id)

    try:
        Project.delete().where(Project.project_id == project_id).execute()
        project.delete_filesystem()
    except Exception as err:
        _LOGGER.error(
            "error while deleting project {}, rolling back: {}".format(project_id, err)
        )

        if not args["force"]:
            raise err

    resp_deleted = data_dump_and_validation(
        ResponseProjectDeletedSchema(), {"success": True, "project_id": project_id}
    )
    _LOGGER.info("deleted project {}".format(project_id))

    return jsonify(resp_deleted), HTTPStatus.OK.value
