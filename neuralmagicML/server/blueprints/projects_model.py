import logging
from http import HTTPStatus

from flask import Blueprint, current_app, request, jsonify
from flasgger import swag_from

from neuralmagicML.server.blueprints.projects import PROJECTS_ROOT_PATH
from neuralmagicML.server.schemas import (
    ErrorSchema,
    ProjectModelAnalysisSchema,
    ResponseProjectModelSchema,
    ResponseProjectModelDeletedSchema,
    SetProjectModelFromSchema,
)


__all__ = ["PROJECT_MODEL_PATH", "projects_model_blueprint"]


PROJECT_MODEL_PATH = "{}/<project_id>/model".format(PROJECTS_ROOT_PATH)

_LOGGER = logging.getLogger(__name__)

projects_model_blueprint = Blueprint(
    PROJECT_MODEL_PATH, __name__, url_prefix=PROJECT_MODEL_PATH
)


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
def upload_model(project_id: str):
    pass


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
def load_model_from_path(project_id: str):
    pass


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
def load_model_from_repo(project_id: str):
    pass


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
def get_model_details(project_id: str):
    pass


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
            HTTPStatus.OK.value: {"description": "The project's model file"},
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
def get_model_file(project_id: str):
    pass


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
def delete_model(project_id: str):
    pass


@projects_model_blueprint.route("/analysis", methods=["POST"])
@swag_from(
    {
        "tags": ["Projects Model"],
        "summary": "Create an analysis file of the model for the project",
        "produces": ["application/octet-stream"],
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
                "schema": ProjectModelAnalysisSchema,
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
def create_analysis(project_id: str):
    pass


@projects_model_blueprint.route("/analysis")
@swag_from(
    {
        "tags": ["Projects Model"],
        "summary": "Get the analysis file of the model for the project",
        "produces": ["application/octet-stream"],
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
                "schema": ProjectModelAnalysisSchema,
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
def get_analysis(project_id: str):
    pass


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
def delete_analysis(project_id: str):
    pass
