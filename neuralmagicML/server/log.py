"""
Root logging file for the server application to handle standard logging setups
"""

import logging

from neuralmagicML.log import _create_console_stream, DEFAULT_LOG_LEVEL

__all__ = ["set_server_logging_level", "get_nm_server_logger"]


NM_SERVER_LOGGER = logging.getLogger("neuralmagicML.server")
NM_SERVER_LOGGER.setLevel(DEFAULT_LOG_LEVEL)
NM_SERVER_LOGGER.addHandler(
    _create_console_stream(
        DEFAULT_LOG_LEVEL,
        "%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
)


def set_server_logging_level(level: int):
    """
    Set the logging level for the NM_SERVER logger along with all
    loggers created in the neuralmagicML.server namespace

    :param level: the log level to set; ex: logging.INFO
    """
    NM_SERVER_LOGGER.setLevel(level)
    for hand in NM_SERVER_LOGGER.handlers:
        hand.setLevel(level)


def get_nm_server_logger() -> logging.Logger:
    """
    :return: the logger used for the neuralmagicML.server root package that all
        other loggers in that namespace are created from
    """
    return NM_SERVER_LOGGER
