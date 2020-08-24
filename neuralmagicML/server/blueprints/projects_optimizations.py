import logging
from http import HTTPStatus

from flask import Blueprint, current_app, request, jsonify
from flasgger import swag_from

from neuralmagicML.server.blueprints.projects import PROJECTS_ROOT_PATH
from neuralmagicML.server.schemas import (
    ErrorSchema,
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
    pass


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
    pass


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
    pass


@projects_optim_blueprint.route("/<optim_id>", methods=["UPDATE"])
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
    pass


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
    pass


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
