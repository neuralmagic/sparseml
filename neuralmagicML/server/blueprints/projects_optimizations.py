import logging
import os
import glob
import re
from http import HTTPStatus

from flask import Blueprint, current_app, request, jsonify, send_file
from flasgger import swag_from
from peewee import JOIN

from neuralmagicML.utils import clean_path
from neuralmagicML.onnx.recal import ModelAnalyzer
from neuralmagicML.server.blueprints.projects import PROJECTS_ROOT_PATH
from neuralmagicML.server.blueprints.helpers import (
    HTTPNotFoundError,
    get_project_by_id,
    get_project_model_by_project_id,
    get_project_optimizer_by_ids,
)
from neuralmagicML.server.schemas import (
    ML_FRAMEWORKS,
    ErrorSchema,
    ProjectOptimizationSchema,
    CreateProjectOptimizationSchema,
    CreateUpdateProjectOptimizationModifiersPruningSchema,
    CreateUpdateProjectOptimizationModifiersQuantizationSchema,
    CreateUpdateProjectOptimizationModifiersLRScheduleSchema,
    ProjectOptimizationModifierLRSchema,
    CreateUpdateProjectOptimizationModifiersTrainableSchema,
    UpdateProjectOptimizationSchema,
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


def _get_epochs_from_lr_mods(lr_mods: ProjectOptimizationModifierLRSchema):
    start_epoch = None
    end_epoch = None
    for lr_mod in lr_mods:
        if start_epoch is None or lr_mod["start_epoch"] < start_epoch:
            start_epoch = lr_mod["start_epoch"]
        if end_epoch is None or lr_mod["end_epoch"] > end_epoch:
            end_epoch = lr_mod["end_epoch"]

    return start_epoch, end_epoch


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
    :param project_id: project_id for optimizers
    :return: List of project optimizers with project_id
    """
    _LOGGER.info("getting all project optimizers for project {}".format(project_id))
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
        .where(ProjectOptimization.project_id == project_id,)
        .group_by(ProjectOptimization)
    )

    optimizers = []

    for res in query:
        optimizers.append(res)

    response = ResponseProjectOptimizationsSchema().dump({"optims": optimizers})

    return jsonify(response), HTTPStatus.OK.value


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
    Creates a project optimizer for given project with project_id
    :param project_id: project_id for optimizer
    :return: Created optimizer
    """
    project = get_project_by_id(project_id)

    data = CreateProjectOptimizationSchema().dump(request.get_json(force=True))

    training_epochs = project.training_epochs
    start_epoch = 0
    stabilization_epochs = 1
    pruning_epochs = int(training_epochs / 3)
    fine_tuning_epochs = int(training_epochs / 4)
    end_epoch = stabilization_epochs + pruning_epochs + fine_tuning_epochs

    model = get_project_model_by_project_id(project_id)
    node_ids = [node.id_ for node in ModelAnalyzer(model.file_path).nodes]
    with database.atomic() as transaction:
        try:
            optim = ProjectOptimization.create(
                start_epoch=start_epoch, end_epoch=end_epoch, project=project, **data
            )
            if data["add_trainable"]:
                ProjectOptimizationModifierTrainable.create(
                    start_epoch=start_epoch,
                    end_epoch=end_epoch,
                    optim=optim,
                    nodes=[
                        {"node_id": node_id, "trainable": True} for node_id in node_ids
                    ],
                )
            if data["add_pruning"]:
                pruning_start_epochs = stabilization_epochs
                ProjectOptimizationModifierPruning.create(
                    start_epoch=pruning_start_epochs,
                    end_epoch=pruning_start_epochs + pruning_epochs,
                    optim=optim,
                    update_frequency=1,
                    sparsity=0.85,
                    nodes=[
                        {"node_id": node_id, "sparsity": 0.85} for node_id in node_ids
                    ],
                )
            if data["add_quantization"]:
                ProjectOptimizationModifierQuantization.create(
                    start_epoch=start_epoch,
                    end_epoch=end_epoch,
                    optim=optim,
                    nodes=[{"node_id": node_id} for node_id in node_ids],
                )
            if data["add_lr_schedule"]:
                ProjectOptimizationModifierLRSchedule.create(
                    start_epoch=start_epoch, end_epoch=end_epoch, optim=optim,
                )

        except Exception as err:
            _LOGGER.error(
                "error while creating new optimizer, rolling back: {}".format(err)
            )
            transaction.rollback()
            raise err

    response = ResponseProjectOptimizationSchema().dump({"optims": optim})

    return jsonify(response), HTTPStatus.OK.value


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
    project = get_project_by_id(project_id)

    resp_schema = ResponseProjectOptimizationFrameworksAvailableSchema()
    resp_frameworks = resp_schema.dump({"frameworks": ML_FRAMEWORKS})
    resp_schema.validate(resp_frameworks)

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
    project = get_project_by_id(project_id)

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

    resp_schema = ResponseProjectOptimizationFrameworksAvailableSamplesSchema()
    resp_samples = resp_schema.dump({"framework": framework, "samples": samples})
    resp_schema.validate(resp_samples)

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
    project = get_project_by_id(project_id)

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
    pass


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
    pass


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
    :param project_id: project_id for optimizer
    :param optim_id: optim_id for optimizer
    :return: Project optimizers with matching project_id and optim_id
    """
    _LOGGER.info(
        "getting project optimizer {} for project {}".format(optim_id, project_id)
    )
    optim = get_project_optimizer_by_ids(project_id, optim_id)

    response = ResponseProjectOptimizationSchema().dump({"optims": optim})
    _LOGGER.info("retrieved project optimizer {}".format(project_id))
    return jsonify(response), HTTPStatus.OK.value


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
    Updates a project optimizer
    :param project_id: project_id for optimizer
    :param optim_id: optim_id for optimizer
    :return: Updated project optimizers with matching project_id and optim_id
    """
    _LOGGER.info(
        "updating project optimizer {} for project {}".format(optim_id, project_id)
    )
    data = UpdateProjectOptimizationSchema().dump(request.get_json(force=True))
    optim = get_project_optimizer_by_ids(project_id, optim_id)

    for key, val in data.items():
        setattr(optim, key, val)

    optim.save()
    response = ResponseProjectOptimizationSchema().dump({"optims": optim})
    _LOGGER.info("updated project optimizer {}".format(response))
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
    _LOGGER.info(
        "deleting project optimizer {} for project {}".format(optim_id, project_id)
    )

    optim = get_project_optimizer_by_ids(project_id, optim_id)

    with database.atomic() as transaction:
        try:
            optim.delete_instance()
        except Exception as err:
            _LOGGER.error("error while deleting optim, rolling back: {}".format(err))
            transaction.rollback()
            raise err

    response = ResponseProjectOptimizationDeletedSchema().dump(
        {"optim_id": optim_id, "project_id": project_id}
    )

    return jsonify(response), HTTPStatus.OK.value


@projects_optim_blueprint.route("/<optim_id>/config")
@swag_from(
    {
        "tags": ["Projects Optimizations"],
        "summary": "Get an optimization config for the projects model.",
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
def get_optim_config(project_id: str, optim_id: str):
    pass


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

    response = ResponseProjectOptimizationModifierDeletedSchema().dump(
        {"project_id": project_id, "optim_id": optim_id}
    )

    return jsonify(response), HTTPStatus.OK.value


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
    optim = get_project_optimizer_by_ids(project_id, optim_id)

    data = CreateUpdateProjectOptimizationModifiersPruningSchema().dump(
        request.get_json(force=True)
    )

    with database.atomic() as transaction:
        try:
            ProjectOptimizationModifierPruning.create(
                optim=optim,
                est_recovery=None,
                est_perf_gain=None,
                est_time=None,
                est_time_baseline=None,
                **data
            )
        except Exception as err:
            _LOGGER.error(
                "error while creating pruning modifier, rolling back: {}".format(err)
            )
            transaction.rollback()
            raise err

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
    pruning_mod = ProjectOptimizationModifierPruning.get_or_none(
        ProjectOptimizationModifierPruning.modifier_id == modifier_id
    )

    if pruning_mod is None:
        _LOGGER.error(
            "could not find pruning modifier with modifier_id {}".format(modifier_id)
        )
        raise HTTPNotFoundError(
            "could not find pruning modifier with modifier_id {}".format(modifier_id)
        )

    data = CreateUpdateProjectOptimizationModifiersPruningSchema().dump(
        request.get_json(force=True)
    )

    for key, val in data.items():
        setattr(pruning_mod, key, val)

    setattr(pruning_mod, "est_recovery", None)
    setattr(pruning_mod, "est_perf_gain", None)
    setattr(pruning_mod, "est_time", None)
    setattr(pruning_mod, "est_time_basline", None)

    pruning_mod.save()
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
    pass


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
    pass


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
    optim = get_project_optimizer_by_ids(project_id, optim_id)

    data = CreateUpdateProjectOptimizationModifiersLRScheduleSchema().dump(
        request.get_json(force=True)
    )

    with database.atomic() as transaction:
        try:
            start_epoch, end_epoch = _get_epochs_from_lr_mods(data["lr_mods"])
            ProjectOptimizationModifierLRSchedule.create(
                optim=optim, start_epoch=start_epoch, end_epoch=end_epoch, **data
            )
        except Exception as err:
            _LOGGER.error(
                "error while creating lr schedule modifier, rolling back: {}".format(
                    err
                )
            )
            transaction.rollback()
            raise err

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
    pruning_mod = ProjectOptimizationModifierLRSchedule.get_or_none(
        ProjectOptimizationModifierLRSchedule.modifier_id == modifier_id
    )

    if pruning_mod is None:
        _LOGGER.error(
            "could not find lr schedule modifier with modifier_id {}".format(
                modifier_id
            )
        )
        raise HTTPNotFoundError(
            "could not find lr schedule modifier with modifier_id {}".format(
                modifier_id
            )
        )

    data = CreateUpdateProjectOptimizationModifiersLRScheduleSchema().dump(
        request.get_json(force=True)
    )

    for key, val in data.items():
        if key == "lr_mods":
            start_epoch, end_epoch = _get_epochs_from_lr_mods(val)
            setattr(pruning_mod, "start_epoch", start_epoch)
            setattr(pruning_mod, "end_epoch", end_epoch)
        setattr(pruning_mod, key, val)

    pruning_mod.save()
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
    pass


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
    pass
