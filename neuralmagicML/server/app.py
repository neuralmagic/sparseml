from typing import Any
import os
import pkg_resources
import logging
import argparse

from flask import Flask
from flask_cors import CORS
from playhouse.flask_utils import FlaskDB
from flasgger import Swagger

from neuralmagicML.utils import create_dirs, convert_to_bool, clean_path
from neuralmagicML.server.log import set_server_logging_level
from neuralmagicML.server.blueprints import (
    errors_blueprint,
    jobs_blueprint,
    projects_blueprint,
    projects_benchmark_blueprint,
    projects_data_blueprint,
    projects_model_blueprint,
    projects_optim_blueprint,
    projects_profiles_blueprint,
    system_blueprint,
    ui_blueprint,
)
from neuralmagicML.server.models import (
    database,
    storage,
    Job,
    Project,
    ProjectModel,
    ProjectData,
)


__all__ = ["run"]

_LOGGER = logging.getLogger(__name__)


def _validate_working_dir(working_dir: str) -> str:
    if not working_dir:
        working_dir = os.getenv("NM_SERVER_WORKING_DIR", "")

    if not working_dir:
        working_dir = os.path.join("~", "nm_server")

    working_dir = clean_path(working_dir)

    try:
        create_dirs(working_dir)
    except Exception as err:
        raise RuntimeError(
            (
                "Error while trying to create neuralmagicML.server "
                "working_dir at {}: {}"
            ).format(working_dir, err)
        )

    return working_dir


def _setup_logging(logging_level: str):
    try:
        logging_level = getattr(logging, logging_level)
    except Exception as err:
        _LOGGER.error(
            "error setting logging level to {}: {}".format(logging_level, err)
        )

    set_server_logging_level(logging_level)


def _database_setup(app: Flask, working_dir: str):
    storage.init(working_dir)
    db_path = os.path.join(working_dir, "db.sqlite")
    database.init(db_path, max_connections=10, stale_timeout=300, timeout=0)
    FlaskDB(app, database)

    database.connect()
    database.create_tables(models=[Job, Project, ProjectModel, ProjectData], safe=True)


def _blueprints_setup(app: Flask):
    app.register_blueprint(errors_blueprint)
    app.register_blueprint(jobs_blueprint)
    app.register_blueprint(projects_blueprint)
    app.register_blueprint(projects_benchmark_blueprint)
    app.register_blueprint(projects_data_blueprint)
    app.register_blueprint(projects_model_blueprint)
    app.register_blueprint(projects_optim_blueprint)
    app.register_blueprint(projects_profiles_blueprint)
    app.register_blueprint(system_blueprint)
    app.register_blueprint(ui_blueprint)


def _api_docs_setup(app: Flask):
    try:
        dist = pkg_resources.get_distribution("neuralmagicML")
        version = dist.version
    except Exception as err:
        _LOGGER.error("error while getting neuralmagicML version: {}".format(err))
        version = None

    Swagger(app)


def run(working_dir: str, host: str, port: int, debug: bool, logging_level: str):
    working_dir = _validate_working_dir(working_dir)
    _setup_logging(logging_level)

    app = Flask("neuralmagicML.server")
    CORS(app)

    _database_setup(app, working_dir)
    _blueprints_setup(app)
    _api_docs_setup(app)

    app.run(host=host, port=port, debug=debug, threaded=True)


def parse_args() -> Any:
    parser = argparse.ArgumentParser(description="neuralmagicML.server")
    parser.add_argument(
        "--working-dir",
        default=None,
        type=str,
        help="The path to the working directory to store state in, "
        "defaults to ~/nm_server",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        type=str,
        help="The host path to launch the server on",
    )
    parser.add_argument(
        "--port", default=5543, type=int, help="The local port to launch the server on"
    )
    parser.add_argument(
        "--debug",
        default=False,
        type=convert_to_bool,
        help="True to run in debug mode, False otherwise",
    )
    parser.add_argument(
        "--logging-level",
        default="INFO",
        type=str,
        help="The logging level to report at",
    )

    return parser.parse_args()


if __name__ == "__main__":
    ARGS = parse_args()
    run(ARGS.working_dir, ARGS.host, ARGS.port, ARGS.debug, ARGS.logging_level)
