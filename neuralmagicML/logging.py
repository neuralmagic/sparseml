"""
Root logging file to handle standard logging setups for the package
"""

import logging


__all__ = ["set_logging_level", "get_nm_root_logger", "get_main_logger"]


def _create_console_stream(level: int, format_: str, datefmt: str):
    stream = logging.StreamHandler()
    stream.setLevel(level)
    formatter = logging.Formatter(format_, datefmt)
    stream.setFormatter(formatter)

    return stream


DEFAULT_LOG_LEVEL = logging.INFO

NM_ROOT_LOGGER = logging.getLogger("neuralmagicML")
NM_ROOT_LOGGER.setLevel(DEFAULT_LOG_LEVEL)
NM_ROOT_LOGGER.addHandler(
    _create_console_stream(
        DEFAULT_LOG_LEVEL,
        "%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
)

MAIN_LOGGER = logging.getLogger("__main__")
MAIN_LOGGER.setLevel(DEFAULT_LOG_LEVEL)
MAIN_LOGGER.addHandler(
    _create_console_stream(
        DEFAULT_LOG_LEVEL,
        "%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
)


def set_logging_level(level: int):
    NM_ROOT_LOGGER.setLevel(level)
    for hand in NM_ROOT_LOGGER.handlers:
        hand.setLevel(level)

    MAIN_LOGGER.setLevel(level)
    for hand in MAIN_LOGGER.handlers:
        hand.setLevel(level)


def get_nm_root_logger() -> logging.Logger:
    return NM_ROOT_LOGGER


def get_main_logger() -> logging.Logger:
    return MAIN_LOGGER
