"""
Setup a color logger
=============

Example:
---------

.. code-block:: python

    from neuralib.util.color_logging import setup_clogger

    logger = setup_clogger()

    logger.debug("a debug message")
    logger.info("an info message")
    logger.warning("a warning message")
    logger.error("an error message")
    logger.critical("a critical message")

    # custom logging
    LOGGING_IO_LEVEL = 11
    LOGGING_IO_NAME = 'IO'

    logger.log(LOGGING_IO_LEVEL, 'io information message')

"""
from __future__ import annotations

import logging

__all__ = [
    'LOGGING_IO_LEVEL',
    #
    'setup_logger',
    'setup_clogger'
]

# custom
LOGGING_IO_LEVEL = 11
LOGGING_IO_NAME = 'IO'


def setup_logger() -> logging.Logger:
    """basic built-in logger"""
    _format = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(format=_format, level=logging.INFO)
    return logging.getLogger()


def setup_clogger(level: int | str = 10,
                  caller_name: str | None = None) -> logging.Logger:
    """
    Return a logger with a default ColoredFormatter.

    :param level: default level = 10 (DEBUG level)
    :param caller_name: show the name <Path name, Script name, ...> if needed
    :return:
    """
    from colorlog import ColoredFormatter

    logging.addLevelName(LOGGING_IO_LEVEL, LOGGING_IO_NAME)

    if caller_name is None:
        _format = "%(log_color)s %(levelname)-8s%(reset)s %(log_color)s[%(asctime)s] %(light_green)s %(message)s"
    else:
        _format = "%(log_color)s %(levelname)-8s%(reset)s %(log_color)s[%(asctime)s] [%(name)s] %(light_green)s %(message)s"

    formatter = ColoredFormatter(
        _format,
        datefmt=None,
        reset=True,
        log_colors={
            "DEBUG": "white",
            'IO': "purple",
            "INFO": "cyan",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red",
        },
    )

    logger = logging.getLogger(caller_name)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    #
    if isinstance(level, str) and level not in formatter.log_colors.keys():
        raise ValueError(f'invalid level string: {level}')
    logger.setLevel(level)

    return logger


