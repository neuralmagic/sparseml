import logging
from http import HTTPStatus

from flask import Blueprint, current_app, request, jsonify
from flasgger import swag_from

from neuralmagicML.server.blueprints.projects import PROJECTS_ROOT_PATH
from neuralmagicML.server.schemas import (
    ErrorSchema,
    ResponseProjectDataSchema,
    ResponseProjectDataDeletedSchema,
    SetProjectDataFromSchema,
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
def upload_data(project_id: str):
    pass


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
def load_data_from_path(project_id: str):
    pass


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
    pass


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
    pass


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
def get_data_single_details(project_id: str, data_id: str):
    pass


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
def delete_data_details(project_id: str, data_id: str):
    pass


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
            HTTPStatus.OK.value: {"description": "The project's data file"},
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
    pass
