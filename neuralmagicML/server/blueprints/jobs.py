import logging
from http import HTTPStatus

from flask import Blueprint, current_app, request, jsonify
from flasgger import swag_from

from neuralmagicML.server.blueprints.helpers import API_ROOT_PATH
from neuralmagicML.server.schemas import (
    JobSchema,
    ResponseJobSchema,
    ResponseJobsSchema,
    ErrorSchema,
    SearchJobsSchema,
)


__all__ = ["JOBS_PATH", "jobs_blueprint"]


JOBS_PATH = "{}/jobs".format(API_ROOT_PATH)

_LOGGER = logging.getLogger(__name__)

jobs_blueprint = Blueprint(JOBS_PATH, __name__, url_prefix=JOBS_PATH)


@jobs_blueprint.route("/")
@swag_from(
    {
        "tags": ["Jobs"],
        "summary": "Get the list of jobs that have ended, running, or pending to run",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "query",
                "name": "order_by",
                "type": "string",
                "enum": ["created", "modified", "status"],
                "description": "The field to order the jobs by in the response. "
                "Default created",
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
                "schema": ResponseJobsSchema,
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
def get_jobs():
    pass


@jobs_blueprint.route("/<job_id>")
@swag_from(
    {
        "tags": ["Jobs"],
        "summary": "Get a job",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "job_id",
                "description": "ID of the job to return",
                "required": True,
                "type": "string",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The matching job",
                "schema": ResponseJobSchema,
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
def get_job(job_id: str):
    pass


@jobs_blueprint.route("/<job_id>/cancel", methods=["POST"])
@swag_from(
    {
        "tags": ["Jobs"],
        "summary": "Cancel a job",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "job_id",
                "description": "ID of the job to return",
                "required": True,
                "type": "string",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The canceled job details",
                "schema": ResponseJobSchema,
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
def cancel_job(job_id: str):
    pass
