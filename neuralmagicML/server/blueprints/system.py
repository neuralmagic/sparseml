"""
Server routes related to the system
"""

import logging
import socket
import psutil
from http import HTTPStatus

from flask import Blueprint, jsonify
from flasgger import swag_from

from neuralmagicML.onnx.utils import available_engines
from neuralmagicML.server.blueprints.helpers import API_ROOT_PATH
from neuralmagicML.server.schemas import (
    data_dump_and_validation,
    ResponseSystemInfo,
    ErrorSchema,
)

try:
    import neuralmagic

    nm_import_exception = None
except Exception as nm_err:
    neuralmagic = None
    nm_import_exception = nm_err

try:
    import onnxruntime

    ort_import_execption = None
except Exception as ort_err:
    onnxruntime = None
    ort_import_execption = ort_err


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

    sys_info = {
        "available_engines": available_engines(),
        "ip_address": socket.gethostbyname(socket.gethostname()),
    }

    # get sys info from neuralmagic.cpu
    if neuralmagic is not None:
        nm_info = neuralmagic.cpu.cpu_architecture()
        nm_info = {k.lower(): v for k, v in nm_info.items()}  # standardize case
        sys_info.update(nm_info)  # add to main dict

        available_instructions = neuralmagic.cpu.VALID_VECTOR_EXTENSIONS
        available_instructions = [ins.upper() for ins in available_instructions]
        sys_info["available_instructions"] = available_instructions

        _LOGGER.info("retrieved system info using neuralmagic.cpu")
    else:
        _LOGGER.info("retrieved basic system info")
        sys_info["cores_per_socket"] = psutil.cpu_count(logical=False)

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

    if nm_import_exception is not None:
        raise nm_import_exception

    if ort_import_execption is not None:
        raise ort_import_execption

    _LOGGER.info("validated system")

    return "", HTTPStatus.OK.value
