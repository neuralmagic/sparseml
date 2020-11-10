"""
Server routes related to benchmarks
"""

from typing import Dict
import datetime
import json
import logging
from http import HTTPStatus

from flask import Blueprint, request, jsonify
from flasgger import swag_from
from marshmallow import ValidationError

from neuralmagicML.onnx.utils import get_ml_sys_info
from neuralmagicML.server.blueprints.projects import PROJECTS_ROOT_PATH
from neuralmagicML.server.blueprints.utils import (
    get_project_benchmark_by_ids,
    get_project_by_id,
)
from neuralmagicML.server.schemas import (
    data_dump_and_validation,
    ErrorSchema,
    CreateProjectBenchmarkSchema,
    ProjectBenchmarkSchema,
    ResponseProjectBenchmarkSchema,
    ResponseProjectBenchmarksSchema,
    ResponseProjectBenchmarkDeletedSchema,
    SearchProjectBenchmarksSchema,
)

from neuralmagicML.server.models import (
    Job,
    ProjectBenchmark,
)

from neuralmagicML.server.workers import (
    JobWorkerManager,
    CreateBenchmarkJobWorker,
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
    args = {key: val for key, val in request.args.items()}
    _LOGGER.info("getting project benchmark for project_id {}".format(project_id))

    args = SearchProjectBenchmarksSchema().load(args)
    query = (
        ProjectBenchmark.select()
        .where(ProjectBenchmark.project_id == project_id)
        .order_by(ProjectBenchmark.created)
        .paginate(args["page"], args["page_length"])
    )
    benchmarks = [res for res in query]
    resp_benchmarks = data_dump_and_validation(
        ResponseProjectBenchmarksSchema(), {"benchmarks": benchmarks}
    )
    _LOGGER.info("retrieved {} benchmarks".format(len(benchmarks)))

    return jsonify(resp_benchmarks), HTTPStatus.OK.value


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
    _LOGGER.info(
        "creating benchmark for project {} for request json {}".format(
            project_id, request.get_json()
        )
    )

    project = get_project_by_id(project_id)

    benchmark_params = CreateProjectBenchmarkSchema().load(request.get_json(force=True))

    model = project.model
    if model is None:
        raise ValidationError(
            (
                "A model has not been set for the project with id {}, "
                "project must set a model before running a benchmark."
            ).format(project_id)
        )

    sys_info = get_ml_sys_info()
    benchmark = None
    job = None

    try:
        benchmark_params["instruction_sets"] = (
            sys_info["available_instructions"]
            if "available_instructions" in sys_info
            else []
        )
        benchmark = ProjectBenchmark.create(
            project=project, source="generated", **benchmark_params
        )

        job = Job.create(
            project_id=project.project_id,
            type_=CreateBenchmarkJobWorker.get_type(),
            worker_args=CreateBenchmarkJobWorker.format_args(
                model_id=model.model_id,
                benchmark_id=benchmark.benchmark_id,
                core_counts=benchmark.core_counts,
                batch_sizes=benchmark.batch_sizes,
                instruction_sets=benchmark.instruction_sets,
                inference_models=benchmark.inference_models,
                warmup_iterations_per_check=benchmark.warmup_iterations_per_check,
                iterations_per_check=benchmark.iterations_per_check,
            ),
        )
        benchmark.job = job
        benchmark.save()
    except Exception as err:
        _LOGGER.error(
            "error while creating new benchmark, rolling back: {}".format(err)
        )
        if benchmark:
            try:
                benchmark.delete_instance()
            except Exception as rollback_err:
                _LOGGER.error(
                    "error while rolling back new benchmark: {}".format(rollback_err)
                )
        if job:
            try:
                job.delete_instance()
            except Exception as rollback_err:
                _LOGGER.error(
                    "error while rolling back new benchmark: {}".format(rollback_err)
                )
        raise err

    JobWorkerManager().refresh()

    resp_benchmark = data_dump_and_validation(
        ResponseProjectBenchmarkSchema(), {"benchmark": benchmark}
    )
    _LOGGER.info("created benchmark and job: {}".format(resp_benchmark))

    return jsonify(resp_benchmark), HTTPStatus.OK.value


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
    _LOGGER.info("uploading benchmark for project {}".format(project_id))

    project = get_project_by_id(project_id)  # validate id

    if "benchmark_file" not in request.files:
        _LOGGER.error("missing uploaded file 'benchmark_file'")
        raise ValidationError("missing uploaded file 'benchmark_file'")

    # read benchmark file
    try:
        benchmark = json.load(request.files["benchmark_file"])  # type: Dict
    except Exception as err:
        _LOGGER.error("error while reading uploaded benchmark file: {}".format(err))
        raise ValidationError(
            "error while reading uploaded benchmark file: {}".format(err)
        )

    benchmark["benchmark_id"] = "<none>"
    benchmark["project_id"] = "<none>"
    benchmark["source"] = "uploaded"
    benchmark["job"] = None
    benchmark["created"] = datetime.datetime.now()

    benchmark = data_dump_and_validation(ProjectBenchmarkSchema(), benchmark)
    del benchmark["benchmark_id"]
    del benchmark["project_id"]
    del benchmark["created"]

    model = project.model
    if model is None:
        raise ValidationError(
            (
                "A model has not been set for the project with id {}, "
                "project must set a model before running a benchmark."
            ).format(project_id)
        )
    benchmark_model = None

    try:
        benchmark_model = ProjectBenchmark.create(project=project, **benchmark)
        resp_benchmark = data_dump_and_validation(
            ResponseProjectBenchmarkSchema(), {"benchmark": benchmark_model}
        )
    except Exception as err:
        _LOGGER.error(
            "error while creating new benchmark, rolling back: {}".format(err)
        )
        try:
            benchmark_model.delete_instance()
        except Exception as rollback_err:
            _LOGGER.error(
                "error while rolling back new benchmark: {}".format(rollback_err)
            )
        raise err

    _LOGGER.info(
        "created benchmark: id: {}, name: {}".format(
            resp_benchmark["benchmark"]["benchmark_id"],
            resp_benchmark["benchmark"]["name"],
        )
    )

    return jsonify(resp_benchmark), HTTPStatus.OK.value


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
    _LOGGER.info(
        "getting benchmark for project {} with id {}".format(project_id, benchmark_id)
    )

    get_project_by_id(project_id)

    benchmark = get_project_benchmark_by_ids(project_id, benchmark_id)

    resp_benchmark = data_dump_and_validation(
        ResponseProjectBenchmarkSchema(), {"benchmark": benchmark}
    )
    _LOGGER.info(
        "found benchmark with benchmark_id {} and project_id: {}".format(
            benchmark_id, project_id
        )
    )

    return jsonify(resp_benchmark), HTTPStatus.OK.value


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
    _LOGGER.info(
        "deleting benchmark for project {} with id {}".format(project_id, benchmark_id)
    )

    get_project_by_id(project_id)

    benchmark = get_project_benchmark_by_ids(project_id, benchmark_id)
    benchmark.delete_instance()

    resp_del = data_dump_and_validation(
        ResponseProjectBenchmarkDeletedSchema(),
        {"success": True, "project_id": project_id, "benchmark_id": benchmark_id},
    )
    _LOGGER.info(
        "deleted benchmark with benchmark_id {} and project_id: {}".format(
            benchmark_id, project_id
        )
    )

    return jsonify(resp_del), HTTPStatus.OK.value
