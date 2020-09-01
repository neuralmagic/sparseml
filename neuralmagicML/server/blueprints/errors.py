"""
Flask blueprint setup for handling errors for the server application
"""

import logging
from http import HTTPStatus

from marshmallow import ValidationError
from werkzeug.exceptions import BadRequest
from flask import Blueprint, jsonify

from neuralmagicML.server.blueprints.helpers import HTTPNotFoundError
from neuralmagicML.server.schemas import data_dump_and_validation, ErrorSchema

__all__ = ["errors_blueprint"]


_LOGGER = logging.getLogger(__name__)

errors_blueprint = Blueprint("errors", __name__)


@errors_blueprint.app_errorhandler(ValidationError)
@errors_blueprint.app_errorhandler(BadRequest)
def handle_client_error(error: ValidationError):
    """
    Handle an error that occurred in the flask app that should return a 400 response

    :param error: the error that occurred
    :return: a tuple containing (json response, http status code [400])
    """
    _LOGGER.error("handling client error, returning 400 status: {}".format(error))
    resp_error = data_dump_and_validation(
        ErrorSchema(),
        {"error_type": error.__class__.__name__, "error_message": str(error)},
    )

    return jsonify(resp_error), HTTPStatus.BAD_REQUEST


@errors_blueprint.app_errorhandler(HTTPNotFoundError)
def handle_client_error(error: HTTPNotFoundError):
    """
    Handle an error that occurred in the flask app that should return a 404 response

    :param error: the error that occurred
    :return: a tuple containing (json response, http status code [404])
    """
    _LOGGER.error("handling not found error, returning 404 status: {}".format(error))
    resp_error = data_dump_and_validation(
        ErrorSchema(),
        {"error_type": error.__class__.__name__, "error_message": str(error)},
    )

    return jsonify(resp_error), HTTPStatus.NOT_FOUND


@errors_blueprint.app_errorhandler(Exception)
def handle_unexpected_error(error: Exception):
    """
    Handle an error that occurred in the flask app that was not expected.
    Will return as a 500 representing a server error, 500

    :param error: the error that occurred
    :return: a tuple containing (json response, http status code [500])
    """
    _LOGGER.error("handling unexpected error, returning 500 status: {}".format(error))
    resp_error = data_dump_and_validation(
        ErrorSchema(),
        {"error_type": error.__class__.__name__, "error_message": str(error)},
    )

    return jsonify(resp_error), HTTPStatus.INTERNAL_SERVER_ERROR
