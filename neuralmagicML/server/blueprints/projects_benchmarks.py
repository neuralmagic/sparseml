"""
Server routes related to benchmarks
"""

import logging
from http import HTTPStatus

from flask import Blueprint, current_app, request, jsonify
from flasgger import swag_from

from neuralmagicML.server.blueprints.projects import PROJECTS_ROOT_PATH
from neuralmagicML.server.schemas import (
    ErrorSchema,
    CreateProjectBenchmarkSchema,
    ResponseProjectBenchmarkSchema,
    ResponseProjectBenchmarksSchema,
    ResponseProjectBenchmarkDeletedSchema,
)


__all__ = ["PROJECT_BENCHMARK_PATH", "projects_benchmark_blueprint"]


PROJECT_BENCHMARK_PATH = "{}/<project_id>/benchmarks".format(PROJECTS_ROOT_PATH)

_LOGGER = logging.getLogger(__name__)

projects_benchmark_blueprint = Blueprint(
    PROJECT_BENCHMARK_PATH, __name__, url_prefix=PROJECT_BENCHMARK_PATH
)


@projects_benchmark_blueprint.route("/")
@swag_from(
    {
        "tags": ["Projects Benchmarks"],
        "summary": "Get a list of benchmarks in the project",
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
                "description": "The project's benchmarks",
                "schema": ResponseProjectBenchmarksSchema,
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
def get_benchmarks(project_id: str):
    """
    Route for getting a list of benchmarks for a given project
    filtered by the flask request args.
    Raises an HTTPNotFoundError if the project is not found in the database.

    :param project_id: the id of the project to get benchmarks for
    :return: a tuple containing (json response, http status code)
    """
    pass


@projects_benchmark_blueprint.route("/", methods=["POST"])
@swag_from(
    {
        "tags": ["Projects Benchmarks"],
        "summary": "Create/run a new benchmark for the projects model.",
        "description": "Creates a background job to do this, pull the status using "
        "the jobs api for the returned job info.",
        "consumes": ["application/json"],
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to create a benchmark for",
                "required": True,
                "type": "string",
            },
            {
                "in": "body",
                "name": "body",
                "description": "The benchmark settings to create with",
                "required": True,
                "schema": CreateProjectBenchmarkSchema,
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The created benchmark",
                "schema": ResponseProjectBenchmarkSchema,
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
def create_benchmark(project_id: str):
    """
    Route for creating a new benchmark for a given project.
    Raises an HTTPNotFoundError if the project is not found in the database.

    :param project_id: the id of the project to create a benchmark for
    :return: a tuple containing (json response, http status code)
    """
    pass


@projects_benchmark_blueprint.route("/upload", methods=["POST"])
@swag_from(
    {
        "tags": ["Projects Benchmarks"],
        "summary": "Upload a new benchmark for the projects model.",
        "consumes": ["multipart/form-data"],
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to upload the benchmark for",
                "required": True,
                "type": "string",
            },
            {
                "in": "formData",
                "name": "benchmark_file",
                "description": "The JSON benchmark file",
                "required": True,
                "type": "file",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The created benchmark",
                "schema": ResponseProjectBenchmarkSchema,
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
def upload_benchmark(project_id: str):
    """
    Route for creating a new benchmark for a given project from uploaded data.
    Raises an HTTPNotFoundError if the project is not found in the database.

    :param project_id: the id of the project to create a benchmark for
    :return: a tuple containing (json response, http status code)
    """
    pass


@projects_benchmark_blueprint.route("/<benchmark_id>")
@swag_from(
    {
        "tags": ["Projects Benchmarks"],
        "summary": "Get a benchmark for the projects model.",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to create a benchmark for",
                "required": True,
                "type": "string",
            },
            {
                "in": "path",
                "name": "benchmark_id",
                "description": "ID of the benchmark within the project to get",
                "required": True,
                "type": "string",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The requested benchmark",
                "schema": ResponseProjectBenchmarkSchema,
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
def get_benchmark(project_id: str, benchmark_id: str):
    """
    Route for getting a specific benchmark for a given project.
    Raises an HTTPNotFoundError if the project or benchmark are
    not found in the database.

    :param project_id: the id of the project to get the benchmark for
    :param benchmark_id: the id of the benchmark to get
    :return: a tuple containing (json response, http status code)
    """
    pass


@projects_benchmark_blueprint.route("/<benchmark_id>", methods=["DELETE"])
@swag_from(
    {
        "tags": ["Projects Benchmarks"],
        "summary": "Delete a benchmark for the projects model.",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to create a benchmark for",
                "required": True,
                "type": "string",
            },
            {
                "in": "path",
                "name": "benchmark_id",
                "description": "ID of the benchmark within the project to delete",
                "required": True,
                "type": "string",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "Deleted the benchmark",
                "schema": ResponseProjectBenchmarkDeletedSchema,
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
def delete_benchmark(project_id: str, benchmark_id: str):
    """
    Route for deleting a specific benchmark for a given project.
    Raises an HTTPNotFoundError if the project or benchmark are
    not found in the database.

    :param project_id: the id of the project to delete the benchmark for
    :param benchmark_id: the id of the benchmark to delete
    :return: a tuple containing (json response, http status code)
    """
    pass
