import logging
from http import HTTPStatus

from flask import Blueprint, current_app, request, jsonify
from flasgger import swag_from

from neuralmagicML.server.blueprints.projects import PROJECTS_ROOT_PATH
from neuralmagicML.server.schemas import (
    ErrorSchema,
    CreateProjectPerfProfileSchema,
    CreateProjectLossProfileSchema,
    ResponseProjectLossProfileSchema,
    ResponseProjectLossProfilesSchema,
    ResponseProjectPerfProfileSchema,
    ResponseProjectPerfProfilesSchema,
    ResponseProjectProfileDeletedSchema,
)


__all__ = ["PROJECT_PROFILES_PATH", "projects_profiles_blueprint"]


PROJECT_PROFILES_PATH = "{}/<project_id>/profiles/".format(PROJECTS_ROOT_PATH)

_LOGGER = logging.getLogger(__name__)

projects_profiles_blueprint = Blueprint(
    PROJECT_PROFILES_PATH, __name__, url_prefix=PROJECT_PROFILES_PATH
)


@projects_profiles_blueprint.route("/loss")
@swag_from(
    {
        "tags": ["Projects Profiles"],
        "summary": "Get a list of loss profiles in the project",
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
                "description": "The project's loss profiles",
                "schema": ResponseProjectLossProfilesSchema,
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
def get_loss_profiles_metadata(project_id: str):
    pass


@projects_profiles_blueprint.route("/loss", methods=["POST"])
@swag_from(
    {
        "tags": ["Projects Profiles"],
        "summary": "Create/run a new loss profile for the projects model.",
        "description": "Creates a background job to do this, pull the status using "
        "the jobs api for the returned job info.",
        "consumes": ["application/json"],
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to create a profile for",
                "required": True,
                "type": "string",
            },
            {
                "in": "body",
                "name": "body",
                "description": "The profile settings to create with",
                "required": True,
                "schema": CreateProjectLossProfileSchema,
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The created loss profiles",
                "schema": ResponseProjectLossProfileSchema,
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
def create_loss_profile(project_id: str):
    pass


@projects_profiles_blueprint.route("/loss/upload", methods=["POST"])
@swag_from(
    {
        "tags": ["Projects Profiles"],
        "summary": "Upload a new loss profile for the projects model.",
        "consumes": ["multipart/form-data"],
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to upload the profile for",
                "required": True,
                "type": "string",
            },
            {
                "in": "formData",
                "name": "loss_file",
                "description": "The JSON loss profile file",
                "required": True,
                "type": "file",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The created loss profiles metadata",
                "schema": ResponseProjectLossProfileSchema,
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
def upload_loss_profile(project_id: str):
    pass


@projects_profiles_blueprint.route("/loss/<profile_id>")
@swag_from(
    {
        "tags": ["Projects Profiles"],
        "summary": "Get a loss profile metadata for the projects model.",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to create a profile for",
                "required": True,
                "type": "string",
            },
            {
                "in": "path",
                "name": "profile_id",
                "description": "ID of the profile within the project to get",
                "required": True,
                "type": "string",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The requested loss profile",
                "schema": ResponseProjectLossProfileSchema,
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
def get_loss_profile(project_id: str, profile_id: str):
    pass


@projects_profiles_blueprint.route("/loss/<profile_id>", methods=["DELETE"])
@swag_from(
    {
        "tags": ["Projects Profiles"],
        "summary": "Delete a loss profile for the projects model.",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to create a profile for",
                "required": True,
                "type": "string",
            },
            {
                "in": "path",
                "name": "profile_id",
                "description": "ID of the profile within the project to delete",
                "required": True,
                "type": "string",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "Deleted the profile",
                "schema": ResponseProjectProfileDeletedSchema,
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
def delete_loss_profile(project_id: str, profile_id: str):
    pass


@projects_profiles_blueprint.route("/perf")
@swag_from(
    {
        "tags": ["Projects Profiles"],
        "summary": "Get a list of perf profiles in the project",
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
                "description": "The project's perf profiles",
                "schema": ResponseProjectPerfProfilesSchema,
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
def get_perf_profiles_metadata(project_id: str):
    pass


@projects_profiles_blueprint.route("/perf", methods=["POST"])
@swag_from(
    {
        "tags": ["Projects Profiles"],
        "summary": "Create/run a new perf profile for the projects model.",
        "description": "Creates a background job to do this, pull the status using "
        "the jobs api for the returned job info.",
        "consumes": ["application/json"],
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to create a profile for",
                "required": True,
                "type": "string",
            },
            {
                "in": "body",
                "name": "body",
                "description": "The profile settings to create with",
                "required": True,
                "schema": CreateProjectPerfProfileSchema,
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The created perf profiles",
                "schema": ResponseProjectPerfProfileSchema,
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
def create_perf_profile(project_id: str):
    pass


@projects_profiles_blueprint.route("/perf/upload", methods=["POST"])
@swag_from(
    {
        "tags": ["Projects Profiles"],
        "summary": "Upload a new perf profile for the projects model.",
        "consumes": ["multipart/form-data"],
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to upload the profile for",
                "required": True,
                "type": "string",
            },
            {
                "in": "formData",
                "name": "perf_file",
                "description": "The JSON perf profile file",
                "required": True,
                "type": "file",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The created perf profiles metadata",
                "schema": ResponseProjectPerfProfileSchema,
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
def upload_perf_profile(project_id: str):
    pass


@projects_profiles_blueprint.route("/perf/<profile_id>")
@swag_from(
    {
        "tags": ["Projects Profiles"],
        "summary": "Get a perf profile metadata for the projects model.",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to create a profile for",
                "required": True,
                "type": "string",
            },
            {
                "in": "path",
                "name": "profile_id",
                "description": "ID of the profile within the project to get",
                "required": True,
                "type": "string",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The requested perf profile",
                "schema": ResponseProjectPerfProfileSchema,
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
def get_perf_profile(project_id: str, profile_id: str):
    pass


@projects_profiles_blueprint.route("/perf/<profile_id>", methods=["DELETE"])
@swag_from(
    {
        "tags": ["Projects Profiles"],
        "summary": "Delete a perf profile for the projects model.",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to create a profile for",
                "required": True,
                "type": "string",
            },
            {
                "in": "path",
                "name": "profile_id",
                "description": "ID of the profile within the project to delete",
                "required": True,
                "type": "string",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "Deleted the profile",
                "schema": ResponseProjectProfileDeletedSchema,
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
def delete_perf_profile(project_id: str, profile_id: str):
    pass
