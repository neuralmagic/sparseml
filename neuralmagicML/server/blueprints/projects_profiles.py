import logging
from http import HTTPStatus

from flask import Blueprint, current_app, request, jsonify
from flasgger import swag_from
import json
from marshmallow import ValidationError

from neuralmagicML.server.blueprints.helpers import (
    HTTPNotFoundError,
    get_project_by_id,
)
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
    SearchProjectsSchema,
)
from neuralmagicML.server.models import (
    database,
    Project,
    ProjectModel,
    Job,
    ProjectLossProfile,
    ProjectPerfProfile,
)
from neuralmagicML.server.workers import (
    JobWorkerManager,
    CreateLossProfileJobWorker,
    CreatePerfProfileJobWorker,
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
def get_loss_profiles(project_id: str):
    _LOGGER.info("getting loss profiles for project {} ".format(project_id))

    get_project_by_id(project_id)  # validate id

    args = SearchProjectsSchema().load({key: val for key, val in request.args.items()})
    loss_profiles_query = (
        ProjectLossProfile.select()
        .where(ProjectLossProfile.project_id == project_id)
        .paginate(args["page"], args["page_length"])
    )
    loss_profiles = [res for res in loss_profiles_query]

    resp_schema = ResponseProjectLossProfilesSchema()
    resp_profiles = resp_schema.dump({"profiles": loss_profiles})
    resp_schema.validate(resp_profiles)
    _LOGGER.info("retrieved {} profiles".format(len(loss_profiles)))

    return jsonify(resp_profiles), HTTPStatus.OK.value


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
    _LOGGER.info(
        "creating loss profile for project {} for request json {}".format(
            project_id, request.json
        )
    )
    project = get_project_by_id(project_id)

    loss_profile_params = CreateProjectLossProfileSchema().load(
        request.get_json(force=True)
    )

    model = ProjectModel.get_or_none(ProjectModel.project == project)
    if model is None:
        raise ValidationError(
            (
                "A model is has not been set for the project with id {}, "
                "project must set a model before running a loss profile."
            ).format(project_id)
        )

    with database.atomic() as transaction:
        try:
            loss_profile = ProjectLossProfile.create(
                project=project, **loss_profile_params
            )
            job = Job.create(
                project=project,
                type_=CreateLossProfileJobWorker.get_type(),
                worker_args=CreateLossProfileJobWorker.format_args(
                    profile_id=loss_profile.profile_id,
                    model_path=model.file_path,
                    name=loss_profile_params["name"],
                    pruning_estimations=loss_profile_params["pruning_estimations"],
                    pruning_estimation_type=loss_profile_params[
                        "pruning_estimation_type"
                    ],
                    pruning_structure=loss_profile_params["pruning_structure"],
                    quantized_estimations=loss_profile_params["quantized_estimations"],
                ),
            )
            loss_profile.job = job
            loss_profile.save()
        except Exception as err:
            _LOGGER.error(
                "error while creating new loss profile, rolling back: {}".format(err)
            )
            transaction.rollback()
            raise err

    # call into JobWorkerManager to kick off job if it's not already running
    JobWorkerManager().refresh()

    resp_schema = ResponseProjectLossProfileSchema()
    resp_profile = resp_schema.dump({"profile": loss_profile})
    resp_schema.validate(resp_profile)
    _LOGGER.info("created loss profile and job: {}".format(resp_profile))

    return jsonify(resp_profile), HTTPStatus.OK.value


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
    _LOGGER.info("uploading loss profile for project {}".format(project_id))

    project = get_project_by_id(project_id)  # validate id

    if "loss_file" not in request.files:
        _LOGGER.error("missing uploaded file 'loss_file'")
        raise ValidationError("missing uploaded file 'loss_file'")

    # read loss analysis file
    try:
        loss_analysis = json.load(request.files["loss_file"])
    except Exception as err:
        _LOGGER.error("error while reading uploaded loss analysis file: {}".format(err))
        raise ValidationError(
            "error while reading uploaded loss analysis file: {}".format(err)
        )

    loss_analysis["project_id"] = project.project_id  # override project_id

    with database.atomic() as transaction:
        try:
            loss_profile = ProjectLossProfile.create(**loss_analysis)
            loss_profile.project = project
            loss_profile.save()
        except Exception as err:
            _LOGGER.error(
                "error while creating new loss profile, rolling back: {}".format(err)
            )
            transaction.rollback()
            raise err

    resp_schema = ResponseProjectLossProfileSchema()
    resp_profile = resp_schema.dump({"profile": loss_profile})
    resp_schema.validate(resp_profile)
    _LOGGER.info(
        "created loss profile: id: {}, name: {}".format(
            resp_profile["profile"]["profile_id"], resp_profile["profile"]["name"]
        )
    )

    return jsonify(resp_profile), HTTPStatus.OK.value


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
    _LOGGER.info(
        "getting loss profile for project {} with id {}".format(project_id, profile_id)
    )

    get_project_by_id(project_id)  # validate id

    # search for loss profile and verify that project_id matches
    loss_profile = ProjectLossProfile.get_or_none(
        ProjectLossProfile.profile_id == profile_id
        and ProjectLossProfile.project_id == project_id
    )
    if loss_profile is None:
        _LOGGER.error(
            "could not find loss profile with profile_id {} and project_id {}".format(
                profile_id, project_id
            )
        )
        raise HTTPNotFoundError(
            "could not find loss profile with profile_id {} and project_id {}".format(
                profile_id, project_id
            )
        )

    resp_schema = ResponseProjectLossProfileSchema()
    resp_profile = resp_schema.dump({"profile": loss_profile})
    _LOGGER.info(
        "found loss profile with profile_id {} and project_id: {}".format(
            profile_id, project_id
        )
    )

    return jsonify(resp_profile), HTTPStatus.OK.value


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
    _LOGGER.info(
        "deleting loss profile for project {} with id {}".format(project_id, profile_id)
    )

    get_project_by_id(project_id)  # validate id

    # search for loss profile and verify that project_id matches
    loss_profile = ProjectLossProfile.get_or_none(
        ProjectLossProfile.profile_id == profile_id
        and ProjectLossProfile.project_id == project_id
    )
    if loss_profile is None:
        _LOGGER.error(
            "could not find loss profile with profile_id {} and project_id {}".format(
                profile_id, project_id
            )
        )
        raise HTTPNotFoundError(
            "could not find loss profile with profile_id {} and project_id {}".format(
                profile_id, project_id
            )
        )

    loss_profile.delete_instance()

    resp_schema = ResponseProjectProfileDeletedSchema()
    resp_del = resp_schema.dump(
        {"success": True, "project_id": project_id, "profile_id": profile_id}
    )
    resp_schema.validate(resp_del)
    _LOGGER.info(
        "deleted loss profile with profile_id {} and project_id: {}".format(
            profile_id, project_id
        )
    )

    return jsonify(resp_del), HTTPStatus.OK.value


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
def get_perf_profiles(project_id: str):
    _LOGGER.info("getting perf profiles for project {} ".format(project_id))

    get_project_by_id(project_id)  # validate id

    args = SearchProjectsSchema().load({key: val for key, val in request.args.items()})
    perf_profiles_query = (
        ProjectPerfProfile.select()
        .where(ProjectPerfProfile.project_id == project_id)
        .paginate(args["page"], args["page_length"])
    )
    perf_profiles = [res for res in perf_profiles_query]

    resp_schema = ResponseProjectPerfProfilesSchema()
    resp_profiles = resp_schema.dump({"profiles": perf_profiles})
    resp_schema.validate(resp_profiles)
    _LOGGER.info("retrieved {} profiles".format(len(perf_profiles)))

    return jsonify(resp_profiles), HTTPStatus.OK.value


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
    _LOGGER.info(
        "creating perf profile for project {} for request json {}".format(
            project_id, request.json
        )
    )
    project = get_project_by_id(project_id)

    perf_profile_params = CreateProjectPerfProfileSchema().load(
        request.get_json(force=True)
    )
    model = ProjectModel.get_or_none(ProjectModel.project == project)
    if model is None:
        raise ValidationError(
            (
                "A model is has not been set for the project with id {}, "
                "project must set a model before running a perf profile."
            ).format(project_id)
        )

    with database.atomic() as transaction:
        try:
            perf_profile = ProjectPerfProfile.create(
                project=project, **perf_profile_params
            )
            job = Job.create(
                project=project,
                type_=CreatePerfProfileJobWorker.get_type(),
                worker_args=CreatePerfProfileJobWorker.format_args(
                    profile_id=perf_profile.profile_id,
                    model_path=model.file_path,
                    name=perf_profile_params["name"],
                    batch_size=perf_profile_params["batch_size"],
                    core_count=perf_profile_params["core_count"],
                    pruning_estimations=perf_profile_params["pruning_estimations"],
                    quantized_estimations=perf_profile_params["quantized_estimations"],
                ),
            )
            perf_profile.job = job
            perf_profile.save()

        except Exception as err:
            _LOGGER.error(
                "error while creating new perf profile, rolling back: {}".format(err)
            )
            transaction.rollback()
            raise err

    # call into JobWorkerManager to kick off job if it's not already running
    JobWorkerManager().refresh()

    resp_schema = ResponseProjectPerfProfileSchema()
    resp_profile = resp_schema.dump({"profile": perf_profile})
    resp_schema.validate(resp_profile)
    _LOGGER.info("created perf profile and job: {}".format(resp_profile))

    return jsonify(resp_profile), HTTPStatus.OK.value


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
    _LOGGER.info("uploading perf profile for project {}".format(project_id))

    project = get_project_by_id(project_id)  # validate id

    if "perf_file" not in request.files:
        _LOGGER.error("missing uploaded file 'perf_file'")
        raise ValidationError("missing uploaded file 'perf_file'")

    # read perf analysis file
    try:
        perf_analysis = json.load(request.files["perf_file"])
    except Exception as err:
        _LOGGER.error("error while reading uploaded perf analysis file: {}".format(err))
        raise ValidationError(
            "error while reading uploaded perf analysis file: {}".format(err)
        )

    perf_analysis["project_id"] = project.project_id

    with database.atomic() as transaction:
        try:
            perf_profile = ProjectPerfProfile.create(**perf_analysis)
            perf_profile.project = project
            perf_profile.save()
        except Exception as err:
            _LOGGER.error(
                "error while creating new perf profile, rolling back: {}".format(err)
            )
            transaction.rollback()
            raise err

    resp_schema = ResponseProjectPerfProfileSchema()
    resp_profile = resp_schema.dump({"profile": perf_profile})
    resp_schema.validate(resp_profile)
    _LOGGER.info(
        "created perf profile: id: {}, name: {}".format(
            resp_profile["profile"]["profile_id"], resp_profile["profile"]["name"]
        )
    )

    return jsonify(resp_profile), HTTPStatus.OK.value


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
    _LOGGER.info(
        "getting perf profile for project {} with id {}".format(project_id, profile_id)
    )

    get_project_by_id(project_id)  # validate id

    # search for perf profile and verify that project_id matches
    perf_profile = ProjectPerfProfile.get_or_none(
        ProjectPerfProfile.profile_id == profile_id
        and ProjectPerfProfile.project_id == project_id
    )

    if perf_profile is None:
        _LOGGER.error(
            "could not find perf profile with profile_id {} and project_id {}".format(
                profile_id, project_id
            )
        )
        raise HTTPNotFoundError(
            "could not find perf profile with profile_id {} and project_id {}".format(
                profile_id, project_id
            )
        )

    resp_schema = ResponseProjectPerfProfileSchema()
    resp_profile = resp_schema.dump({"profile": perf_profile})
    resp_schema.validate(resp_profile)
    _LOGGER.info(
        "found perf profile with profile_id {} and project_id: {}".format(
            profile_id, project_id
        )
    )

    return jsonify(resp_profile), HTTPStatus.OK.value


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
    _LOGGER.info(
        "deleting perf profile for project {} with id {}".format(project_id, profile_id)
    )

    get_project_by_id(project_id)  # validate id

    # search for perf profile and verify that project_id matches
    perf_profile = ProjectPerfProfile.get_or_none(
        ProjectPerfProfile.profile_id == profile_id
        and ProjectPerfProfile.project_id == project_id
    )

    if perf_profile is None:
        _LOGGER.error(
            "could not find perf profile with profile_id {} and project_id {}".format(
                profile_id, project_id
            )
        )
        raise HTTPNotFoundError(
            "could not find perf profile with profile_id {} and project_id {}".format(
                profile_id, project_id
            )
        )

    perf_profile.delete_instance()

    resp_schema = ResponseProjectProfileDeletedSchema()
    resp_del = resp_schema.dump(
        {"success": True, "project_id": project_id, "profile_id": profile_id}
    )
    resp_schema.validate(resp_del)
    _LOGGER.info(
        "deleted perf profile with profile_id {} and project_id: {}".format(
            profile_id, project_id
        )
    )

    return jsonify(resp_del), HTTPStatus.OK.value
