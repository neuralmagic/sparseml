"""
Server routes related to project optimizations and modifiers
"""

import logging
import os
import glob
import re
from http import HTTPStatus

from flask import Blueprint, request, jsonify, send_file, Response
from flasgger import swag_from
from marshmallow import ValidationError
from peewee import JOIN

from neuralmagicML.utils import clean_path
from neuralmagicML.server.blueprints.projects import PROJECTS_ROOT_PATH
from neuralmagicML.server.blueprints.utils import (
    HTTPNotFoundError,
    get_project_by_id,
    optim_validate_and_get_project_by_id,
    get_project_optimizer_by_ids,
    default_epochs_distribution,
    default_pruning_settings,
    get_profiles_by_id,
    optim_trainable_default_nodes,
    optim_lr_sched_default_mods,
    optim_trainable_updater,
    optim_pruning_updater,
    optim_lr_sched_updater,
    optim_updater,
    create_config,
    validate_pruning_nodes,
    PruningModelEvaluator,
    sparse_training_available,
)
from neuralmagicML.server.schemas import (
    data_dump_and_validation,
    ML_FRAMEWORKS,
    ErrorSchema,
    GetProjectOptimizationBestEstimatedResultsSchema,
    CreateProjectOptimizationSchema,
    CreateUpdateProjectOptimizationModifiersPruningSchema,
    CreateUpdateProjectOptimizationModifiersQuantizationSchema,
    CreateUpdateProjectOptimizationModifiersLRScheduleSchema,
    CreateUpdateProjectOptimizationModifiersTrainableSchema,
    UpdateProjectOptimizationSchema,
    SearchProjectOptimizationsSchema,
    ResponseProjectOptimizationFrameworksAvailableSchema,
    ResponseProjectOptimizationFrameworksAvailableSamplesSchema,
    ResponseProjectOptimizationModifiersAvailable,
    ResponseProjectOptimizationModifiersBestEstimated,
    ResponseProjectOptimizationSchema,
    ResponseProjectOptimizationsSchema,
    ResponseProjectOptimizationDeletedSchema,
    ResponseProjectOptimizationModifierDeletedSchema,
)
from neuralmagicML.server.models import (
    database,
    ProjectOptimization,
    ProjectOptimizationModifierTrainable,
    ProjectOptimizationModifierLRSchedule,
    ProjectOptimizationModifierPruning,
    ProjectOptimizationModifierQuantization,
)


__all__ = ["PROJECT_OPTIM_PATH", "projects_optim_blueprint"]


PROJECT_OPTIM_PATH = "{}/<project_id>/optim".format(PROJECTS_ROOT_PATH)

_LOGGER = logging.getLogger(__name__)

projects_optim_blueprint = Blueprint(
    PROJECT_OPTIM_PATH, __name__, url_prefix=PROJECT_OPTIM_PATH
)


@projects_optim_blueprint.route("/")
@swag_from(
    {
        "tags": ["Projects Optimizations"],
        "summary": "Get a list of optimizations in the project",
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
                "description": "The project's optimizations",
                "schema": ResponseProjectOptimizationsSchema,
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
def get_optims(project_id: str):
    """
    Route for getting a list of optims for a given project
    filtered by the flask request args.
    Raises an HTTPNotFoundError if the project is not found in the database.

    :param project_id: the id of the project to get optims for
    :return: a tuple containing (json response, http status code)
    """
    args = {key: val for key, val in request.args.items()}
    _LOGGER.info(
        "getting project optims for project_id {} and request args {}".format(
            project_id, args
        )
    )
    args = SearchProjectOptimizationsSchema().load(args)
    project = optim_validate_and_get_project_by_id(project_id)  # validate id

    query = (
        ProjectOptimization.select(
            ProjectOptimization,
            ProjectOptimizationModifierLRSchedule,
            ProjectOptimizationModifierPruning,
            ProjectOptimizationModifierQuantization,
            ProjectOptimizationModifierTrainable,
        )
        .join_from(
            ProjectOptimization, ProjectOptimizationModifierLRSchedule, JOIN.LEFT_OUTER,
        )
        .join_from(
            ProjectOptimization, ProjectOptimizationModifierPruning, JOIN.LEFT_OUTER,
        )
        .join_from(
            ProjectOptimization,
            ProjectOptimizationModifierQuantization,
            JOIN.LEFT_OUTER,
        )
        .join_from(
            ProjectOptimization, ProjectOptimizationModifierTrainable, JOIN.LEFT_OUTER,
        )
        .where(ProjectOptimization.project_id == project_id)
        .group_by(ProjectOptimization)
        .order_by(ProjectOptimization.created)
        .paginate(args["page"], args["page_length"])
    )

    optimizers = []

    for res in query:
        optimizers.append(res)

    resp_optims = data_dump_and_validation(
        ResponseProjectOptimizationsSchema(), {"optims": optimizers}
    )
    _LOGGER.info("retrieved {} optims".format(len(optimizers)))

    return jsonify(resp_optims), HTTPStatus.OK.value


@projects_optim_blueprint.route("/", methods=["POST"])
@swag_from(
    {
        "tags": ["Projects Optimizations"],
        "summary": "Create a new optimization for the projects model.",
        "description": "To create and automatically fill out specific modifiers, "
        "use the appropriate params in the body.",
        "consumes": ["application/json"],
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to create a optim for",
                "required": True,
                "type": "string",
            },
            {
                "in": "body",
                "name": "body",
                "description": "The optim settings to create with",
                "required": True,
                "schema": CreateProjectOptimizationSchema,
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The created optimization",
                "schema": ResponseProjectOptimizationSchema,
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
def create_optim(project_id: str):
    """
    Route for creating a new optimizer for a given project.
    Raises an HTTPNotFoundError if the project is not found in the database.

    :param project_id: the id of the project to create a benchmark for
    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info(
        "creating optimizer for project_id {} and request json {}".format(
            project_id, request.json
        )
    )
    project = optim_validate_and_get_project_by_id(project_id)
    data = CreateProjectOptimizationSchema().dump(request.get_json(force=True))
    profile_perf, profile_loss = get_profiles_by_id(
        data["profile_perf_id"] if "profile_perf_id" in data else None,
        data["profile_loss_id"] if "profile_loss_id" in data else None,
    )
    optim = None

    try:
        epochs = default_epochs_distribution(project.training_epochs)
        optim = ProjectOptimization.create(project=project)
        optim_updater(
            optim,
            name=data["name"],
            profile_perf=profile_perf,
            profile_loss=profile_loss,
            start_epoch=epochs.start_epoch,
            end_epoch=epochs.end_epoch,
        )
        optim.save()

        if data["add_pruning"]:
            pruning_settings = default_pruning_settings()
            pruning = ProjectOptimizationModifierPruning.create(optim=optim)
            optim_pruning_updater(
                pruning,
                start_epoch=epochs.pruning_start_epoch,
                end_epoch=epochs.pruning_end_epoch,
                update_frequency=epochs.pruning_update_frequency,
                pruning_settings=pruning_settings,
                model=project.model,
                profile_perf=profile_perf,
                profile_loss=profile_loss,
                global_start_epoch=epochs.start_epoch,
                global_end_epoch=epochs.end_epoch,
            )
            pruning.save()

        if data["add_quantization"]:
            # TODO: fill in once quantization is added
            raise ValidationError("add_quantization is currently not supported")

        if data["add_trainable"]:
            training = ProjectOptimizationModifierTrainable.create(optim=optim)
            optim_trainable_updater(
                training,
                project.model.analysis,
                start_epoch=epochs.start_epoch,
                end_epoch=epochs.end_epoch,
                default_trainable=True,
                global_start_epoch=epochs.start_epoch,
                global_end_epoch=epochs.end_epoch,
            )
            training.save()

        if data["add_lr_schedule"]:
            lr_mods = optim_lr_sched_default_mods(
                project.training_lr_init,
                project.training_lr_final,
                epochs.start_epoch,
                epochs.fine_tuning_start_epoch,
                epochs.end_epoch,
            )
            lr_schedule = ProjectOptimizationModifierLRSchedule.create(optim=optim)
            optim_lr_sched_updater(
                lr_schedule,
                lr_mods,
                global_start_epoch=epochs.start_epoch,
                global_end_epoch=epochs.end_epoch,
            )
            lr_schedule.save()
    except Exception as err:
        _LOGGER.error(
            "error while creating new optimization, rolling back: {}".format(err)
        )
        if optim:
            try:
                optim.delete_instance()
            except Exception as rollback_err:
                _LOGGER.error(
                    "error while rolling back new optimization: {}".format(rollback_err)
                )
        raise err

    optims_resp = data_dump_and_validation(
        ResponseProjectOptimizationSchema(), {"optim": optim}
    )

    return jsonify(optims_resp), HTTPStatus.OK.value


@projects_optim_blueprint.route("/frameworks")
@swag_from(
    {
        "tags": ["Projects Optimizations"],
        "summary": "Get available ML frameworks optimization of the projects model.",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to get the available modifiers for",
                "required": True,
                "type": "string",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The available ML frameworks optimization of the model",
                "schema": ResponseProjectOptimizationModifiersAvailable,
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
def get_available_frameworks(project_id: str):
    """
    Route for getting the available ML frameworks for optimization of the projects model

    :param project_id: the project_id to get the available frameworks for
    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info(
        "getting the available frameworks for project_id {}".format(project_id)
    )

    # make sure project exists
    # currently project_id doesn't do anything,
    # but gives us flexibility to edit the frameworks for a projects model in the future
    project = optim_validate_and_get_project_by_id(project_id)

    resp_frameworks = data_dump_and_validation(
        ResponseProjectOptimizationFrameworksAvailableSchema(),
        {"frameworks": ML_FRAMEWORKS},
    )
    _LOGGER.info(
        "retrieved available frameworks for project_id {}: {}".format(
            project_id, resp_frameworks
        )
    )

    return jsonify(resp_frameworks), HTTPStatus.OK.value


@projects_optim_blueprint.route("/frameworks/<framework>/samples")
@swag_from(
    {
        "tags": ["Projects Optimizations"],
        "summary": "Get available ML frameworks optimization of the projects model.",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to get the available modifiers for",
                "required": True,
                "type": "string",
            },
            {
                "in": "path",
                "name": "framework",
                "description": "the ML framework to get available sample code types for",
                "required": True,
                "type": "string",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The available ML frameworks optimization of the model",
                "schema": ResponseProjectOptimizationFrameworksAvailableSamplesSchema,
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
def get_available_frameworks_samples(project_id: str, framework: str):
    """
    Route for getting the available sample code for an ML framework
    for optimization of the projects model

    :param project_id: the project_id to get the available frameworks for
    :param framework: the ML framework to get available sample code types for
    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info(
        (
            "getting the available sample code types "
            "for project_id {} and framework {}"
        ).format(project_id, framework)
    )

    # make sure project exists
    # currently project_id doesn't do anything,
    # but gives us flexibility to edit the frameworks for a projects model in the future
    project = optim_validate_and_get_project_by_id(project_id)

    code_samples_dir = os.path.join(
        os.path.dirname(clean_path(__file__)), "code_samples"
    )

    if framework not in ML_FRAMEWORKS:
        raise HTTPNotFoundError(
            "could not find the given framework of {}".format(framework)
        )

    reg = re.compile("(.+)__(.+)\.py")
    samples = []

    for file in glob.glob(os.path.join(code_samples_dir, "{}*.py".format(framework))):
        split = reg.split(os.path.basename(file))
        found_framework = split[1]
        assert found_framework == framework
        samples.append(split[2])

    resp_samples = data_dump_and_validation(
        ResponseProjectOptimizationFrameworksAvailableSamplesSchema(),
        {"framework": framework, "samples": samples},
    )
    _LOGGER.info(
        "retrieved available samples for project_id {} and framework {}: {}".format(
            project_id, framework, resp_samples
        )
    )

    return jsonify(resp_samples), HTTPStatus.OK.value


@projects_optim_blueprint.route("/frameworks/<framework>/samples/<sample>")
@swag_from(
    {
        "tags": ["Projects Optimizations"],
        "summary": "Get code for optimizing with an optimization "
        "for the projects model.",
        "produces": ["text/plain", "application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to create a optim for",
                "required": True,
                "type": "string",
            },
            {
                "in": "path",
                "name": "framework",
                "description": "The ML framework to get example code for",
                "required": True,
                "type": "string",
            },
            {
                "in": "path",
                "name": "sample",
                "description": "The type of sample code to get",
                "required": True,
                "type": "string",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The requested optimization code",
                "content": {"text/plain": {}},
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
def get_framework_sample(project_id: str, framework: str, sample: str):
    """
    Route for getting sample code for an ML framework
    for optimization of the projects model

    :param project_id: the project_id to get the available frameworks for
    :param framework: the ML framework to get available sample code types for
    :param sample: the type of sample code to get
    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info(
        (
            "getting the sample code for project_id {}, framework {}, and sample {}"
        ).format(project_id, framework, sample)
    )

    # make sure project exists
    # currently project_id doesn't do anything,
    # but gives us flexibility to edit the frameworks for a projects model in the future
    project = optim_validate_and_get_project_by_id(project_id)

    code_samples_dir = os.path.join(
        os.path.dirname(clean_path(__file__)), "code_samples"
    )

    if framework not in ML_FRAMEWORKS:
        raise HTTPNotFoundError(
            "could not find the given framework of {}".format(framework)
        )

    sample_file = os.path.join(code_samples_dir, "{}__{}.py".format(framework, sample))

    if not os.path.exists(sample_file):
        raise HTTPNotFoundError(
            (
                "could not find sample code for project_id {}, "
                "framework {} and sample {}"
            ).format(project_id, framework, sample)
        )

    _LOGGER.info(
        (
            (
                "retrieved available sample code for project_id {}, "
                "framework {}, and sample {} from {}"
            )
        ).format(project_id, framework, sample, sample_file)
    )

    return send_file(sample_file, mimetype="text/plain")


@projects_optim_blueprint.route("/modifiers")
@swag_from(
    {
        "tags": ["Projects Optimizations"],
        "summary": "Get available modifiers for optimization for the projects model.",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to get the available modifiers for",
                "required": True,
                "type": "string",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The available modifiers for optimization of the model",
                "schema": ResponseProjectOptimizationModifiersAvailable,
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
def get_available_modifiers(project_id: str):
    """
    Route for getting the available modifiers for a project

    :param project_id: the project_id to get the available modifiers for
    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info(
        "getting the available frameworks for project_id {}".format(project_id)
    )

    # make sure project exists
    # currently project_id doesn't do anything,
    # but gives us flexibility to edit the frameworks for a projects model in the future
    project = optim_validate_and_get_project_by_id(project_id)

    # TODO: add quantization once available
    modifiers = ["pruning", "lr_schedule"]
    if sparse_training_available(project):
        modifiers.append("trainable")

    resp_modifiers = data_dump_and_validation(
        ResponseProjectOptimizationModifiersAvailable(), {"modifiers": modifiers},
    )

    return jsonify(resp_modifiers), HTTPStatus.OK.value


@projects_optim_blueprint.route("/modifiers/best-estimated")
@swag_from(
    {
        "tags": ["Projects Optimizations"],
        "summary": "Get the best estimated values for applying modifiers "
        "as optimizations for the projects model.",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to create a optim for",
                "required": True,
                "type": "string",
            },
            {
                "in": "query",
                "name": "profile_perf_id",
                "type": "str",
                "description": "id for the performance profile to use to "
                "calculate estimates",
            },
            {
                "in": "query",
                "name": "profile_loss_id",
                "type": "str",
                "description": "id for the loss profile to use to "
                "calculate estimates",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The best estimated values for applying modifiers "
                "as optimizations for the projects model",
                "schema": ResponseProjectOptimizationModifiersBestEstimated,
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
def get_modifiers_estimated_speedup(project_id: str):
    """
    Route for getting the estimated metrics for a project's model after optimizing

    :param project_id: the project_id to get the available frameworks for
    :return: a tuple containing (json response, http status code)
    """
    args = {key: val for key, val in request.args.items()}
    _LOGGER.info(
        "getting best estimated speedup for project_id {} and request args {}".format(
            project_id, args
        )
    )
    args = GetProjectOptimizationBestEstimatedResultsSchema().load(args)
    project = optim_validate_and_get_project_by_id(project_id)
    profile_perf, profile_loss = get_profiles_by_id(
        args["profile_perf_id"], args["profile_loss_id"]
    )

    pruning_settings = default_pruning_settings()
    model = PruningModelEvaluator(
        project.model.analysis,
        profile_perf.analysis if profile_perf is not None else None,
        profile_loss.analysis if profile_loss is not None else None,
    )
    model.eval_baseline(default_pruning_settings().sparsity)
    model.eval_pruning(pruning_settings)
    _, model_res = model.to_dict_values()

    resp_est = data_dump_and_validation(
        ResponseProjectOptimizationModifiersBestEstimated(), model_res
    )
    _LOGGER.info("retrieved best estimated speedup {}".format(resp_est))

    return jsonify(resp_est), HTTPStatus.OK.value


@projects_optim_blueprint.route("/<optim_id>")
@swag_from(
    {
        "tags": ["Projects Optimizations"],
        "summary": "Get a optimization for the projects model.",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to create a optim for",
                "required": True,
                "type": "string",
            },
            {
                "in": "path",
                "name": "optim_id",
                "description": "ID of the optim within the project to get",
                "required": True,
                "type": "string",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The requested optimization",
                "schema": ResponseProjectOptimizationSchema,
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
def get_optim(project_id: str, optim_id: str):
    """
    Route for getting a specific optimization for a given project.
    Raises an HTTPNotFoundError if the project or optimization are
    not found in the database.

    :param project_id: the id of the project to get the optimization for
    :param optim_id: the id of the optimization to get
    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info(
        "getting project optimizer {} for project {}".format(optim_id, project_id)
    )
    project = optim_validate_and_get_project_by_id(project_id)
    optim = get_project_optimizer_by_ids(project_id, optim_id)

    resp_optim = data_dump_and_validation(
        ResponseProjectOptimizationSchema(), {"optim": optim}
    )
    _LOGGER.info("retrieved project optimizer {}".format(project_id))

    return jsonify(resp_optim), HTTPStatus.OK.value


@projects_optim_blueprint.route("/<optim_id>", methods=["PUT"])
@swag_from(
    {
        "tags": ["Projects Optimizations"],
        "summary": "Update an optimization for the projects model.",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to create a optimization for",
                "required": True,
                "type": "string",
            },
            {
                "in": "path",
                "name": "optim_id",
                "description": "ID of the optim within the project to delete",
                "required": True,
                "type": "string",
            },
            {
                "in": "body",
                "name": "body",
                "description": "The optim settings to update with",
                "required": True,
                "schema": UpdateProjectOptimizationSchema,
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The updated optimization",
                "schema": ResponseProjectOptimizationDeletedSchema,
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
def update_optim(project_id: str, optim_id: str):
    """
    Route for updating a specific optimization for a given project.
    Raises an HTTPNotFoundError if the project or optimization are
    not found in the database.

    :param project_id: the id of the project to update the optimization for
    :param optim_id: the id of the optimization to update
    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info(
        "updating project optimizer {} for project {}".format(optim_id, project_id)
    )
    data = UpdateProjectOptimizationSchema().dump(request.get_json(force=True))
    project = optim_validate_and_get_project_by_id(project_id)
    optim = get_project_optimizer_by_ids(project_id, optim_id)
    profiles_set = False

    if "profile_perf_id" in data:
        profile_perf_id = data["profile_perf_id"]
        del data["profile_perf_id"]
        profiles_set = True
    else:
        profile_perf_id = optim.profile_perf_id

    if "profile_loss_id" in data:
        profile_loss_id = data["profile_loss_id"]
        del data["profile_loss_id"]
        profiles_set = True
    else:
        profile_loss_id = optim.profile_loss_id

    profile_perf, profile_loss = get_profiles_by_id(profile_perf_id, profile_loss_id)

    try:
        optim_updater(
            optim, **data, profile_perf=profile_perf, profile_loss=profile_loss
        )
        optim.save()

        if not profiles_set:
            # set to None so we don't update the pruning modifiers if not needed
            profile_perf = None
            profile_loss = None

        global_start_epoch = data["start_epoch"] if "start_epoch" in data else None
        global_end_epoch = data["end_epoch"] if "end_epoch" in data else None

        for pruning in optim.pruning_modifiers:
            optim_pruning_updater(
                pruning,
                model=project.model,
                profile_perf=profile_perf,
                profile_loss=profile_loss,
                global_start_epoch=global_start_epoch,
                global_end_epoch=global_end_epoch,
            )
            pruning.save()

        # todo: add quantization updates when ready
        for trainable in optim.trainable_modifiers:
            optim_trainable_updater(
                trainable,
                project.model.analysis,
                global_start_epoch=global_start_epoch,
                global_end_epoch=global_end_epoch,
            )
            trainable.save()

        for lr_schedule in optim.lr_schedule_modifiers:
            optim_lr_sched_updater(
                lr_schedule,
                global_start_epoch=global_start_epoch,
                global_end_epoch=global_end_epoch,
            )
            lr_schedule.save()
    except Exception as err:
        _LOGGER.error("error while updating optimization".format(err))
        raise err

    resp_optim = data_dump_and_validation(
        ResponseProjectOptimizationSchema(), {"optim": optim}
    )
    _LOGGER.info("updated project optimizer {}".format(resp_optim))

    return get_optim(project_id, optim_id)


@projects_optim_blueprint.route("/<optim_id>", methods=["DELETE"])
@swag_from(
    {
        "tags": ["Projects Optimizations"],
        "summary": "Delete an optimization for the projects model.",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to create a optimization for",
                "required": True,
                "type": "string",
            },
            {
                "in": "path",
                "name": "optim_id",
                "description": "ID of the optim within the project to delete",
                "required": True,
                "type": "string",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "Deleted the optimization",
                "schema": ResponseProjectOptimizationDeletedSchema,
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
def delete_optim(project_id: str, optim_id: str):
    """
    Route for deleting a specific optimization for a given project.
    Raises an HTTPNotFoundError if the project or optimization are
    not found in the database.

    :param project_id: the id of the project to delete the optimization for
    :param optim_id: the id of the optimization to delete
    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info(
        "deleting project optimizer {} for project {}".format(optim_id, project_id)
    )
    project = get_project_by_id(project_id)  # make sure project exists
    optim = get_project_optimizer_by_ids(project_id, optim_id)
    optim.delete_instance()
    resp_deleted = data_dump_and_validation(
        ResponseProjectOptimizationDeletedSchema(),
        {"optim_id": optim_id, "project_id": project_id},
    )

    return jsonify(resp_deleted), HTTPStatus.OK.value


@projects_optim_blueprint.route("/<optim_id>/frameworks/<framework>/config")
@swag_from(
    {
        "tags": ["Projects Optimizations"],
        "summary": "Get an optimization config for the projects model "
        "for given framework.",
        "produces": ["text/yaml", "application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to create a optim for",
                "required": True,
                "type": "string",
            },
            {
                "in": "path",
                "name": "optim_id",
                "description": "ID of the optim within the project to get",
                "required": True,
                "type": "string",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The requested optimization config",
                "content": {"application/yaml": {}},
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
def get_optim_config(project_id: str, optim_id: str, framework: str):
    """
    Route for getting the config yaml represnetation for a project's optimization.
    Raises an HTTPNotFoundError if the project or optimization are
    not found in the database.

    :param project_id: the id of the project to get the config file for
    :param optim_id: the id of the optimization to get the config file for
    :param framework: the ML framework to get the config file for
    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info(
        "getting project optimizer config {} for project {} for framework {}".format(
            optim_id, project_id, framework
        )
    )
    project = optim_validate_and_get_project_by_id(project_id)
    optim = get_project_optimizer_by_ids(project_id, optim_id)
    config_string = create_config(project, optim, framework)
    _LOGGER.info(
        "retrieved project optimizer config {} for project {} for framework {}".format(
            optim_id, project_id, framework
        )
    )

    return Response(config_string, mimetype="text/yaml"), HTTPStatus.OK.value


@projects_optim_blueprint.route(
    "/<optim_id>/modifiers/<modifier_id>", methods=["DELETE"]
)
@swag_from(
    {
        "tags": ["Projects Optimizations"],
        "summary": "Delete an optimization's modifier for the projects model.",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to create a optimization for",
                "required": True,
                "type": "string",
            },
            {
                "in": "path",
                "name": "optim_id",
                "description": "ID of the optim within the project to delete",
                "required": True,
                "type": "string",
            },
            {
                "in": "path",
                "name": "modifier_id",
                "description": "ID of the optimization's modifier within "
                "the project to delete",
                "required": True,
                "type": "string",
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "Deleted the optimization's modifier",
                "schema": ResponseProjectOptimizationModifierDeletedSchema,
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
def delete_optim_modifier(project_id: str, optim_id: str, modifier_id: str):
    """
    Route for deleting a modifier in a specific optimization for a given project.
    Raises an HTTPNotFoundError if the project or optim are
    not found in the database.

    :param project_id: the id of the project to delete the modifier for
    :param optim_id: the id of the optim to delete the modifier for
    :param modifier_id: the id of the modifier to delete
    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info("deleting modifier {} for optimizer {}".format(modifier_id, optim_id))
    mods = [
        ProjectOptimizationModifierLRSchedule.get_or_none(
            ProjectOptimizationModifierLRSchedule.modifier_id == modifier_id
        ),
        ProjectOptimizationModifierPruning.get_or_none(
            ProjectOptimizationModifierPruning.modifier_id == modifier_id
        ),
        ProjectOptimizationModifierQuantization.get_or_none(
            ProjectOptimizationModifierQuantization.modifier_id == modifier_id
        ),
        ProjectOptimizationModifierTrainable.get_or_none(
            ProjectOptimizationModifierTrainable.modifier_id == modifier_id
        ),
    ]
    mod_for_deletion = None
    for mod in mods:
        if mod is not None:
            mod_for_deletion = mod
            break
    if mod_for_deletion is None:
        _LOGGER.error(
            "could not find optimization modifier with modifier_id {}".format(
                modifier_id
            )
        )
        raise HTTPNotFoundError(
            "could not find optimization modifier with modifier_id {}".format(
                modifier_id
            )
        )

    mod_for_deletion.delete_instance()
    resp_deleted = data_dump_and_validation(
        ResponseProjectOptimizationModifierDeletedSchema(),
        {"project_id": project_id, "optim_id": optim_id, "modifier_id": modifier_id},
    )
    _LOGGER.info("deleted modifier {} for optimizer {}".format(modifier_id, optim_id))

    return jsonify(resp_deleted), HTTPStatus.OK.value


@projects_optim_blueprint.route("/<optim_id>/modifiers/pruning", methods=["POST"])
@swag_from(
    {
        "tags": ["Projects Optimizations"],
        "summary": "Create a pruning modifier for a project model optimizations.",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to create a optimization for",
                "required": True,
                "type": "string",
            },
            {
                "in": "path",
                "name": "optim_id",
                "description": "ID of the optim within the project to create for",
                "required": True,
                "type": "string",
            },
            {
                "in": "body",
                "name": "body",
                "description": "The modifier settings to create with",
                "required": True,
                "schema": CreateUpdateProjectOptimizationModifiersPruningSchema,
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The updated optimization containing created modifier",
                "schema": ResponseProjectOptimizationSchema,
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
def create_optim_modifier_pruning(project_id: str, optim_id: str):
    """
    Route for creating a new pruning modifier for a given project optim.
    Raises an HTTPNotFoundError if the project or optim are not found in the database.

    :param project_id: the id of the project to create a pruning modifier for
    :param optim_id: the id of the optim to create a pruning modifier for
    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info(
        "creating a pruning modifier for optim {} and project {} with data {}".format(
            optim_id, project_id, request.json
        )
    )
    project = optim_validate_and_get_project_by_id(project_id)
    optim = get_project_optimizer_by_ids(project_id, optim_id)
    profile_perf, profile_loss = get_profiles_by_id(
        optim.profile_perf_id, optim.profile_loss_id
    )
    data = CreateUpdateProjectOptimizationModifiersPruningSchema().load(
        request.get_json(force=True)
    )

    if "nodes" in data and data["nodes"]:
        validate_pruning_nodes(project, data["nodes"])

    default_epochs = default_epochs_distribution(project.training_epochs)
    default_pruning = default_pruning_settings()

    if "balance_perf_loss" not in data:
        data["balance_perf_loss"] = default_pruning.balance_perf_loss

    if "start_epoch" not in data:
        data["start_epoch"] = default_epochs.pruning_start_epoch

    if "end_epoch" not in data:
        data["end_epoch"] = default_epochs.pruning_end_epoch

    if "update_frequency" not in data:
        data["update_frequency"] = default_epochs.pruning_update_frequency

    if "mask_type" not in data:
        data["mask_type"] = default_pruning.mask_type

    if "nodes" not in data and "sparsity" not in data:
        data["sparsity"] = default_pruning.sparsity

    pruning = None

    try:
        pruning = ProjectOptimizationModifierPruning.create(optim=optim)
        optim_pruning_updater(
            pruning,
            **data,
            model=project.model,
            profile_perf=profile_perf,
            profile_loss=profile_loss,
        )
        pruning.save()

        # update optim in case bounds for new modifier went outside it
        optim_updater(
            optim, mod_start_epoch=pruning.start_epoch, mod_end_epoch=pruning.end_epoch,
        )
        optim.save()
    except Exception as err:
        _LOGGER.error(
            "error while creating pruning modifier, rolling back: {}".format(err)
        )
        if pruning:
            try:
                pruning.delete_instance()
            except Exception as rollback_err:
                _LOGGER.error(
                    "error while rolling back new pruning: {}".format(rollback_err)
                )
        raise err

    _LOGGER.info(
        "created a pruning modifier with id {} for optim {} and project {}".format(
            pruning.modifier_id, optim_id, project_id
        )
    )

    return get_optim(project_id, optim_id)


@projects_optim_blueprint.route(
    "/<optim_id>/modifiers/<modifier_id>/pruning", methods=["PUT"]
)
@swag_from(
    {
        "tags": ["Projects Optimizations"],
        "summary": "Update a pruning modifier for a project model optimizations.",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to create a optimization for",
                "required": True,
                "type": "string",
            },
            {
                "in": "path",
                "name": "optim_id",
                "description": "ID of the optim within the project to delete",
                "required": True,
                "type": "string",
            },
            {
                "in": "path",
                "name": "modifier_id",
                "description": "ID of the optimization's modifier to update",
                "required": True,
                "type": "string",
            },
            {
                "in": "body",
                "name": "body",
                "description": "The optim settings to update with",
                "required": True,
                "schema": CreateUpdateProjectOptimizationModifiersPruningSchema,
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The updated optimization containing updated modifier",
                "schema": ResponseProjectOptimizationSchema,
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
def update_optim_modifier_pruning(project_id: str, optim_id: str, modifier_id: str):
    """
    Route for updating a pruning modifier for a given project optim.
    Raises an HTTPNotFoundError if the project, optim, or modifier
    are not found in the database.

    :param project_id: the id of the project to update a pruning modifier for
    :param optim_id: the id of the optim to update a pruning modifier for
    :param modifier_id: the id of the modifier to update
    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info(
        (
            "updating a pruning modifier with id {} for optim {} "
            "and project {} with data {}"
        ).format(modifier_id, optim_id, project_id, request.json)
    )
    project = optim_validate_and_get_project_by_id(project_id)
    optim = get_project_optimizer_by_ids(project_id, optim_id)
    pruning = ProjectOptimizationModifierPruning.get_or_none(
        ProjectOptimizationModifierPruning.modifier_id == modifier_id
    )
    if pruning is None:
        _LOGGER.error(
            "could not find pruning modifier {} for project {} with optim_id {}".format(
                modifier_id, project_id, optim_id
            )
        )
        raise HTTPNotFoundError(
            "could not find pruning modifier {} for project {} with optim_id {}".format(
                modifier_id, project_id, optim_id
            )
        )
    profile_perf, profile_loss = get_profiles_by_id(
        optim.profile_perf_id, optim.profile_loss_id
    )
    data = CreateUpdateProjectOptimizationModifiersPruningSchema().load(
        request.get_json(force=True)
    )

    if "nodes" in data and data["nodes"]:
        validate_pruning_nodes(project, data["nodes"])

    try:
        optim_pruning_updater(
            pruning,
            **data,
            model=project.model,
            profile_perf=profile_perf,
            profile_loss=profile_loss,
        )
        pruning.save()

        # update optim in case bounds for new modifier went outside it
        optim_updater(
            optim, mod_start_epoch=pruning.start_epoch, mod_end_epoch=pruning.end_epoch,
        )
        optim.save()
    except Exception as err:
        _LOGGER.error("error while updating pruning modifier".format(err))
        raise err

    _LOGGER.info(
        "updated pruning modifier with id {} for optim {} and project {}".format(
            pruning.modifier_id, optim_id, project_id
        )
    )

    return get_optim(project_id, optim_id)


@projects_optim_blueprint.route("/<optim_id>/modifiers/quantization", methods=["POST"])
@swag_from(
    {
        "tags": ["Projects Optimizations"],
        "summary": "Create a quantization modifier for a project model optimizations.",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to create a optimization for",
                "required": True,
                "type": "string",
            },
            {
                "in": "path",
                "name": "optim_id",
                "description": "ID of the optim within the project to create for",
                "required": True,
                "type": "string",
            },
            {
                "in": "body",
                "name": "body",
                "description": "The modifier settings to create with",
                "required": True,
                "schema": CreateUpdateProjectOptimizationModifiersQuantizationSchema,
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The updated optimization containing created modifier",
                "schema": ResponseProjectOptimizationSchema,
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
def create_optim_modifier_quantization(project_id: str, optim_id: str):
    raise NotImplementedError()


@projects_optim_blueprint.route(
    "/<optim_id>/modifiers/<modifier_id>/quantization", methods=["PUT"]
)
@swag_from(
    {
        "tags": ["Projects Optimizations"],
        "summary": "Update a quantization modifier for a project model optimizations.",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to create a optimization for",
                "required": True,
                "type": "string",
            },
            {
                "in": "path",
                "name": "optim_id",
                "description": "ID of the optim within the project to delete",
                "required": True,
                "type": "string",
            },
            {
                "in": "path",
                "name": "modifier_id",
                "description": "ID of the optimization's modifier to update",
                "required": True,
                "type": "string",
            },
            {
                "in": "body",
                "name": "body",
                "description": "The optim settings to update with",
                "required": True,
                "schema": CreateUpdateProjectOptimizationModifiersQuantizationSchema,
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The updated optimization containing updated modifier",
                "schema": ResponseProjectOptimizationSchema,
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
def update_optim_modifier_quantization(
    project_id: str, optim_id: str, modifier_id: str
):
    raise NotImplementedError()


@projects_optim_blueprint.route("/<optim_id>/modifiers/lr-schedule", methods=["POST"])
@swag_from(
    {
        "tags": ["Projects Optimizations"],
        "summary": "Create a lr-schedule modifier for a project model optimizations.",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to create a optimization for",
                "required": True,
                "type": "string",
            },
            {
                "in": "path",
                "name": "optim_id",
                "description": "ID of the optim within the project to create for",
                "required": True,
                "type": "string",
            },
            {
                "in": "body",
                "name": "body",
                "description": "The modifier settings to create with",
                "required": True,
                "schema": CreateUpdateProjectOptimizationModifiersLRScheduleSchema,
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The updated optimization containing created modifier",
                "schema": ResponseProjectOptimizationSchema,
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
def create_optim_modifier_lr_schedule(project_id: str, optim_id: str):
    """
    Route for creating a new lr schedule modifier for a given project optim.
    Raises an HTTPNotFoundError if the project or optim are not found in the database.

    :param project_id: the id of the project to create an lr schedule modifier for
    :param optim_id: the id of the optim to create an lr schedule modifier for
    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info(
        (
            "creating an lr schedule modifier for optim {} "
            "and project {} with data {}"
        ).format(optim_id, project_id, request.json)
    )
    project = optim_validate_and_get_project_by_id(project_id)
    optim = get_project_optimizer_by_ids(project_id, optim_id)
    data = CreateUpdateProjectOptimizationModifiersLRScheduleSchema().load(
        request.get_json(force=True)
    )

    if "lr_mods" not in data:
        epochs = default_epochs_distribution(project.training_epochs)
        data["lr_mods"] = optim_lr_sched_default_mods(
            project.training_lr_init,
            project.training_lr_final,
            epochs.start_epoch,
            epochs.fine_tuning_start_epoch,
            epochs.end_epoch,
        )

    lr_sched = None

    try:
        lr_sched = ProjectOptimizationModifierLRSchedule.create(optim=optim)
        optim_lr_sched_updater(
            lr_sched, lr_mods=data["lr_mods"] if data["lr_mods"] else [],
        )
        lr_sched.save()

        # update optim in case bounds for new modifier went outside it
        optim_updater(
            optim,
            mod_start_epoch=lr_sched.start_epoch,
            mod_end_epoch=lr_sched.end_epoch,
        )
        optim.save()
    except Exception as err:
        _LOGGER.error(
            "error while creating lr_sched modifier, rolling back: {}".format(err)
        )
        if lr_sched:
            try:
                lr_sched.delete_instance()
            except Exception as rollback_err:
                _LOGGER.error(
                    "error while rolling back new lr schedule: {}".format(rollback_err)
                )
        raise err

    _LOGGER.info(
        "created an lr schedule modifier with id {} for optim {} and project {}".format(
            lr_sched.modifier_id, optim_id, project_id
        )
    )

    return get_optim(project_id, optim_id)


@projects_optim_blueprint.route(
    "/<optim_id>/modifiers/<modifier_id>/lr-schedule", methods=["PUT"]
)
@swag_from(
    {
        "tags": ["Projects Optimizations"],
        "summary": "Update a lr-schedule modifier for a project model optimizations.",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to create a optimization for",
                "required": True,
                "type": "string",
            },
            {
                "in": "path",
                "name": "optim_id",
                "description": "ID of the optim within the project to delete",
                "required": True,
                "type": "string",
            },
            {
                "in": "path",
                "name": "modifier_id",
                "description": "ID of the optimization's modifier to update",
                "required": True,
                "type": "string",
            },
            {
                "in": "body",
                "name": "body",
                "description": "The optim settings to update with",
                "required": True,
                "schema": CreateUpdateProjectOptimizationModifiersLRScheduleSchema,
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The updated optimization containing updated modifier",
                "schema": ResponseProjectOptimizationSchema,
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
def update_optim_modifier_lr_schedule(project_id: str, optim_id: str, modifier_id: str):
    """
    Route for updating an lr schedule modifier for a given project optim.
    Raises an HTTPNotFoundError if the project, optim, or modifier
    are not found in the database.

    :param project_id: the id of the project to update an lr schedule modifier for
    :param optim_id: the id of the optim to update an lr schedule modifier for
    :param modifier_id: the id of the modifier to update
    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info(
        (
            "updating an lr schedule modifier with id {} for optim {} "
            "and project {} with data {}"
        ).format(modifier_id, optim_id, project_id, request.json)
    )
    project = optim_validate_and_get_project_by_id(project_id)
    optim = get_project_optimizer_by_ids(project_id, optim_id)
    lr_sched = ProjectOptimizationModifierLRSchedule.get_or_none(
        ProjectOptimizationModifierLRSchedule.modifier_id == modifier_id
    )
    if lr_sched is None:
        _LOGGER.error(
            (
                "could not find lr schedule modifier {} "
                "for project {} with optim_id {}"
            ).format(modifier_id, project_id, optim_id)
        )
        raise HTTPNotFoundError(
            (
                "could not find lr schedule modifier {} "
                "for project {} with optim_id {}"
            ).format(modifier_id, project_id, optim_id)
        )
    data = CreateUpdateProjectOptimizationModifiersLRScheduleSchema().load(
        request.get_json(force=True)
    )

    try:
        optim_lr_sched_updater(
            lr_sched,
            lr_mods=[]
            if "lr_mods" not in data or not data["lr_mods"]
            else data["lr_mods"],
        )
        lr_sched.save()

        # update optim in case bounds for new modifier went outside it
        optim_updater(
            optim,
            mod_start_epoch=lr_sched.start_epoch,
            mod_end_epoch=lr_sched.end_epoch,
        )
        optim.save()
    except Exception as err:
        _LOGGER.error("error while updating lr_sched modifier".format(err))
        raise err

    _LOGGER.info(
        "updated pruning modifier with id {} for optim {} and project {}".format(
            lr_sched.modifier_id, optim_id, project_id
        )
    )

    return get_optim(project_id, optim_id)


@projects_optim_blueprint.route("/<optim_id>/modifiers/trainable", methods=["POST"])
@swag_from(
    {
        "tags": ["Projects Optimizations"],
        "summary": "Create a trainable modifier for a project model optimizations.",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to create a optimization for",
                "required": True,
                "type": "string",
            },
            {
                "in": "path",
                "name": "optim_id",
                "description": "ID of the optim within the project to create for",
                "required": True,
                "type": "string",
            },
            {
                "in": "body",
                "name": "body",
                "description": "The modifier settings to create with",
                "required": True,
                "schema": CreateUpdateProjectOptimizationModifiersTrainableSchema,
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The updated optimization containing created modifier",
                "schema": ResponseProjectOptimizationSchema,
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
def create_optim_modifier_trainable(project_id: str, optim_id: str):
    """
    Route for creating a new trainable modifier for a given project optim.
    Raises an HTTPNotFoundError if the project or the optim are not found in the database.

    :param project_id: the id of the project to create a trainable modifier for
    :param optim_id: the id of the optim to create a trainable modifier for
    :return: a tuple containing (json response, http status code)
    """

    _LOGGER.info(
        (
            "creating an trainable modifier for optim {} " "and project {} with data {}"
        ).format(optim_id, project_id, request.json)
    )
    project = optim_validate_and_get_project_by_id(project_id)
    optim = get_project_optimizer_by_ids(project_id, optim_id)
    data = CreateUpdateProjectOptimizationModifiersTrainableSchema().load(
        request.get_json(force=True)
    )
    trainable = None

    try:
        trainable = ProjectOptimizationModifierTrainable.create(optim=optim)

        optim_trainable_updater(
            trainable,
            project.model.analysis,
            **data,
            global_start_epoch=optim.start_epoch,
            global_end_epoch=optim.end_epoch,
        )
        trainable.save()

        # update optim in case bounds for new modifier went outside it
        optim_updater(
            optim,
            mod_start_epoch=trainable.start_epoch,
            mod_end_epoch=trainable.end_epoch,
        )
        optim.save()
    except Exception as err:
        _LOGGER.error(
            "error while creating new trainable modifier, rolling back: {}".format(err)
        )
        if trainable:
            try:
                trainable.delete_instance()
            except Exception as rollback_err:
                _LOGGER.error(
                    "error while rolling back new trainable: {}".format(rollback_err)
                )
        raise err

    _LOGGER.info(
        "created a training modifier with id {} for optim {} and project {}".format(
            trainable.modifier_id, optim_id, project_id
        )
    )

    return get_optim(project_id, optim_id)


@projects_optim_blueprint.route(
    "/<optim_id>/modifiers/<modifier_id>/trainable", methods=["PUT"]
)
@swag_from(
    {
        "tags": ["Projects Optimizations"],
        "summary": "Update a trainable modifier for a project model optimizations.",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "path",
                "name": "project_id",
                "description": "ID of the project to create a optimization for",
                "required": True,
                "type": "string",
            },
            {
                "in": "path",
                "name": "optim_id",
                "description": "ID of the optim within the project to delete",
                "required": True,
                "type": "string",
            },
            {
                "in": "path",
                "name": "modifier_id",
                "description": "ID of the optimization's modifier to update",
                "required": True,
                "type": "string",
            },
            {
                "in": "body",
                "name": "body",
                "description": "The optim settings to update with",
                "required": True,
                "schema": CreateUpdateProjectOptimizationModifiersTrainableSchema,
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The updated optimization containing updated modifier",
                "schema": ResponseProjectOptimizationSchema,
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
def update_optim_modifier_trainable(project_id: str, optim_id: str, modifier_id: str):
    """
    Route for updating a trainable modifier for a given project optim.
    Raises an HTTPNotFoundError if the project, optim, or modifier
    are not found in the database.

    :param project_id: the id of the project to update a trainable modifier for
    :param optim_id: the id of the optim to update a trainable modifier for
    :param modifier_id: the id of the modifier to update
    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info(
        (
            "updating a trainable modifier with id {} for optim {} "
            "and project {} with data {}"
        ).format(modifier_id, optim_id, project_id, request.json)
    )
    project = optim_validate_and_get_project_by_id(project_id)
    optim = get_project_optimizer_by_ids(project_id, optim_id)
    trainable = ProjectOptimizationModifierTrainable.get_or_none(
        ProjectOptimizationModifierTrainable.modifier_id == modifier_id
    )
    if trainable is None:
        _LOGGER.error(
            "could not find trainable modifier {} for project {} with optim_id {}".format(
                modifier_id, project_id, optim_id
            )
        )
        raise HTTPNotFoundError(
            "could not find trainable modifier {} for project {} with optim_id {}".format(
                modifier_id, project_id, optim_id
            )
        )
    data = CreateUpdateProjectOptimizationModifiersTrainableSchema().load(
        request.get_json(force=True)
    )

    try:
        optim_trainable_updater(
            trainable,
            project.model.analysis,
            **data,
            global_start_epoch=optim.start_epoch,
            global_end_epoch=optim.end_epoch,
        )
        trainable.save()

        # update optim in case bounds for new modifier went outside it
        optim_updater(
            optim,
            mod_start_epoch=trainable.start_epoch,
            mod_end_epoch=trainable.end_epoch,
        )
        optim.save()
    except Exception as err:
        _LOGGER.error("error while creating trainable modifer".format(err))
        raise err

    _LOGGER.info(
        "updaed training modifier with id {} for optim {} and project {}".format(
            trainable.modifier_id, optim_id, project_id
        )
    )

    return get_optim(project_id, optim_id)
