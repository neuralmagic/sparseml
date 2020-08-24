"""
Flask blueprint setup for handling errors for the server application
"""

import logging
from http import HTTPStatus

from marshmallow import ValidationError
from werkzeug.exceptions import BadRequest
from flask import Blueprint, jsonify

from neuralmagicML.server.blueprints.helpers import HTTPNotFoundError
from neuralmagicML.server.schemas import ErrorSchema

__all__ = ["errors_blueprint"]


_LOGGER = logging.getLogger(__name__)

errors_blueprint = Blueprint("errors", __name__)


@errors_blueprint.app_errorhandler(ValidationError)
@errors_blueprint.app_errorhandler(BadRequest)
def handle_client_error(error: ValidationError):
    _LOGGER.error("handling client error, returning 400 status: {}".format(error))
    error_obj = {"error_type": error.__class__.__name__, "error_message": str(error)}
    response = ErrorSchema().dump(error_obj)

    return jsonify(response), HTTPStatus.BAD_REQUEST


@errors_blueprint.app_errorhandler(HTTPNotFoundError)
def handle_client_error(error: HTTPNotFoundError):
    _LOGGER.error("handling not found error, returning 404 status: {}".format(error))
    error_obj = {"error_type": error.__class__.__name__, "error_message": str(error)}
    response = ErrorSchema().dump(error_obj)

    return jsonify(response), HTTPStatus.NOT_FOUND


@errors_blueprint.app_errorhandler(Exception)
def handle_unexpected_error(error: Exception):
    _LOGGER.error("handling unexpected error, returning 500 status: {}".format(error))
    error_obj = {"error_type": error.__class__.__name__, "error_message": str(error)}
    response = ErrorSchema().dump(error_obj)

    return jsonify(response), HTTPStatus.INTERNAL_SERVER_ERROR
