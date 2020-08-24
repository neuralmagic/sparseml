import logging
from http import HTTPStatus

from flask import Blueprint, current_app, request, jsonify
from flasgger import swag_from

from neuralmagicML.server.blueprints.helpers import API_ROOT_PATH
from neuralmagicML.server.schemas import ResponseSystemInfo, ErrorSchema


__all__ = ["SYSTEM_PATH", "system_blueprint"]


SYSTEM_PATH = "{}/system".format(API_ROOT_PATH)

_LOGGER = logging.getLogger(__name__)

system_blueprint = Blueprint(SYSTEM_PATH, __name__, url_prefix=SYSTEM_PATH)


@system_blueprint.route("/info")
@swag_from(
    {
        "tags": ["System"],
        "summary": "Get system specs and other hardware info",
        "produces": ["application/json"],
        "parameters": [],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The info for the current system the server is on",
                "schema": ResponseSystemInfo,
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
def info():
    pass


@system_blueprint.route("/validate", methods=["POST"])
@swag_from(
    {
        "tags": ["System"],
        "summary": "Validate that the system is setup correctly to run. "
        "For example, make sure neuralmagic and neuralmagicML are accessible",
        "produces": ["application/json"],
        "parameters": [],
        "responses": {
            HTTPStatus.OK.value: {"description": "System is setup correctly"},
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
def validate():
    pass
