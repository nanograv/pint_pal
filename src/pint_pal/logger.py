from loguru import logger
import pint.logging
from typing import Union
import sys

def setup(level: Union[str, int] = "INFO", use_stdout: bool = True) -> None:
    """
    Configure the Loguru logger with a style suitable for notebook use.
    Unlike PINT's default logging configuration, this does not use a separate
    thread to "enqueue" messages, which makes it less suitable for logging from
    multiple threads at once, but should alleviate an issue where log messages
    appear in the notebook cell where the call to `setup()` is made, rather
    than in the cell where they originated.

    Parameters
    ----------
    level: str or int
        The minimum level of messages to log. Can be specified as a string
        or as an integer. The mapping (specified by loguru) is as follows:
        "DEBUG"=10, "INFO"=20, "ERROR"=30, "WARNING"=40, "CRITICAL"=50.
    use_stdout: bool
        If `True` (the default), only messages of "WARNING" level or above
        will be sent to stderr; others will be sent to stdout. Otherwise,
        all messages will be sent to stderr. The default behavior is convenient
        in notebooks, where stderr output is printed with a colored background.
        In a shell, it is preferable to send all messages to stderr so that
        logging output can be redirected separately from stdout.
    """
    log_format = "<level>{level}</level> ({name}:{line}): {message}"
    stderr_filter = pint.logging.LogFilter()

    def stdout_filter(record):
        info_or_below = record["level"].no < logger.level("WARNING").no
        return info_or_below and stderr_filter(record)

    logger.remove()
    if use_stdout:
        logger.add(
            sys.stdout,
            level=level,
            format=log_format,
            filter=stdout_filter,
        )
        logger.add(
            sys.stderr,
            level="WARNING",
            format=log_format,
            filter=stderr_filter,
        )
    else:
        logger.add(
            sys.stderr,
            level=level,
            format=log_format,
            filter=stderr_filter,
        )
