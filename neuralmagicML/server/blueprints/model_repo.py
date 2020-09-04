"""
Server routes related to the model repo routes
"""

import logging
from http import HTTPStatus

from flask import Blueprint, request, jsonify
from flasgger import swag_from

from neuralmagicML.server.blueprints.utils import API_ROOT_PATH
from neuralmagicML.server.schemas import (
    SearchModelRepoModels,
    ResponseModelRepoModels,
    ErrorSchema,
)


__all__ = ["MODEL_REPO_PATH", "model_repo_blueprint"]


MODEL_REPO_PATH = "{}/model-repo".format(API_ROOT_PATH)

_LOGGER = logging.getLogger(__name__)

model_repo_blueprint = Blueprint(MODEL_REPO_PATH, __name__, url_prefix=MODEL_REPO_PATH)


@model_repo_blueprint.route("/", methods=["POST"])
@swag_from(
    {
        "tags": ["Model Repo"],
        "summary": "Get a potentially filtered list of models available "
        "in the model repo",
        "produces": ["application/json"],
        "parameters": [
            {
                "in": "body",
                "name": "body",
                "description": "The filter criteria to search for model repos",
                "required": True,
                "schema": SearchModelRepoModels,
            },
        ],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The requested models from the model repo",
                "schema": ResponseModelRepoModels,
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
def get_models():
    """
    Route for getting a potentially filtered list of models in the model repo

    :return: a tuple containing (json response, http status code)
    """
    raise NotImplementedError()
