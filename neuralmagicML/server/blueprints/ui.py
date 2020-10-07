"""
Flask blueprint setup for serving UI files for the server application
"""

import logging
from http import HTTPStatus

from flask import Blueprint, current_app, send_from_directory
from flasgger import swag_from

from neuralmagicML.server.schemas import ErrorSchema

__all__ = ["ui_blueprint"]


_LOGGER = logging.getLogger(__name__)

ui_blueprint = Blueprint("ui", __name__, url_prefix="/")


@ui_blueprint.route("/", defaults={"path": ""})
@ui_blueprint.route("/index.html")
@ui_blueprint.route("/<path:path>")
@swag_from(
    {
        "tags": ["UI"],
        "summary": "Get the index.html file for the UI to render",
        "produces": ["text/html", "application/json"],
        "parameters": [],
        "responses": {
            HTTPStatus.OK.value: {
                "description": "The index html file",
                "content": {"text/html": {}},
            },
            HTTPStatus.INTERNAL_SERVER_ERROR.value: {
                "description": "Information for the error that occurred",
                "schema": ErrorSchema,
            },
        },
    },
)
def index(path: str):
    """
    Route for getting the root index.html file for the UI

    :param path: the path requested for a UI file to render,
        note this is a catch all, so may improperly catch other misspelled routes
        and therefore fail to render them
    :return: response containing the main index.html file for the ui,
        uses send_from_directory. Targets "/" and "/index.html"
    """
    _LOGGER.info(
        "sending index.html file at {} from {}".format(
            path, current_app.config["UI_PATH"]
        )
    )
    return send_from_directory(current_app.config["UI_PATH"], "index.html")
