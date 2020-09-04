"""
Server routes related to the system
"""

import logging
from http import HTTPStatus

from flask import Blueprint, jsonify
from flasgger import swag_from

from neuralmagicML.onnx.utils import get_ml_sys_info, ml_engines_errors
from neuralmagicML.server.blueprints.utils import API_ROOT_PATH
from neuralmagicML.server.schemas import (
    data_dump_and_validation,
    ResponseSystemInfo,
    ErrorSchema,
)


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
    """
    Route for getting the info describing the current system the server is running on

    :return: a tuple containing (json response, http status code)
    """
    _LOGGER.info("getting system info")
    sys_info = get_ml_sys_info()
    resp_info = data_dump_and_validation(ResponseSystemInfo(), {"info": sys_info})
    _LOGGER.info("retrieved system info {}".format(resp_info))

    return jsonify(resp_info), HTTPStatus.OK.value


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
    """
    Route for validating the current system the server is running on,
    neuralmagic and onnxruntime must be installed to validate successfully

    :return: a tuple containing (response, http status code)
    """
    _LOGGER.info("validating system")
    errors = ml_engines_errors()

    for key, err in errors.items():
        if err is not None:
            raise Exception("error on import for {}: {}".format(key, err))

    _LOGGER.info("validated system")

    return "", HTTPStatus.OK.value
