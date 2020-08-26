import logging
from http import HTTPStatus

from flask import Blueprint, current_app, request, jsonify
from flasgger import swag_from
from peewee import JOIN

from neuralmagicML.onnx.recal import ModelAnalyzer

from neuralmagicML.server.blueprints.projects import PROJECTS_ROOT_PATH
from neuralmagicML.server.blueprints.helpers import (
    HTTPNotFoundError,
    get_project_by_id,
    get_project_model_by_project_id,
    get_project_optimizer_by_ids,
)
from neuralmagicML.server.schemas import (
    ErrorSchema,
    ProjectOptimizationSchema,
    CreateProjectOptimizationSchema,
    CreateUpdateProjectOptimizationModifiersPruningSchema,
    CreateUpdateProjectOptimizationModifiersQuantizationSchema,
    CreateUpdateProjectOptimizationModifiersLRScheduleSchema,
    CreateUpdateProjectOptimizationModifiersTrainableSchema,
    UpdateProjectOptimizationSchema,
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

    data = request.get_json(force=True)
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
                "error while creating new project, rolling back: {}".format(err)
            )
            transaction.rollback()
            raise err

    response = ResponseProjectOptimizationSchema().dump({"optims": optim})

    return jsonify(response), HTTPStatus.OK.value


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
    _LOGGER.info("retrieved project optimizer {}".format(response))
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
            _LOGGER.error(
                "error while creating new project, rolling back: {}".format(err)
            )
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


@projects_optim_blueprint.route("/<optim_id>/code/<framework>/<code_type>")
@swag_from(
    {
        "tags": ["Projects Optimizations"],
        "summary": "Get code for optimizing with an optimization "
        "for the projects model.",
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
            {
                "in": "path",
                "name": "framework",
                "description": "The ML framework to get example code for",
                "required": True,
                "type": "string",
            },
            {
                "in": "path",
                "name": "code_type",
                "description": "The type of code to get",
                "required": True,
                "type": "string",
                "enum": ["integration", "training"],
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The requested optimization code",
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
def get_optim_code(project_id: str, optim_id: str, framework: str, code_type: str):
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
    pass


@projects_optim_blueprint.route(
    "/<optim_id>/modifiers/<modifier_id>/pruning", methods=["POST"]
)
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
def create_optim_modifier_pruning(project_id: str, optim_id: str, modifier_id: str):
    pass


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
    pass


@projects_optim_blueprint.route(
    "/<optim_id>/modifiers/<modifier_id>/quantization", methods=["POST"]
)
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
def create_optim_modifier_quantization(
    project_id: str, optim_id: str, modifier_id: str
):
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


@projects_optim_blueprint.route(
    "/<optim_id>/modifiers/<modifier_id>/lr-schedule", methods=["POST"]
)
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
def create_optim_modifier_lr_schedule(project_id: str, optim_id: str, modifier_id: str):
    pass


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
    pass


@projects_optim_blueprint.route(
    "/<optim_id>/modifiers/<modifier_id>/trainable", methods=["POST"]
)
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
def create_optim_modifier_trainable(project_id: str, optim_id: str, modifier_id: str):
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
