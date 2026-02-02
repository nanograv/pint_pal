from loguru import logger
import pint.logging
from typing import Union, Optional, IO
from collections.abc import Callable
from pathlib import Path
import sys
import time
import warnings

def log_format(record):
    level = record['level']
    name = record['name']
    line = record['line']
    message = record['message']
    if 'extra' in record and record['extra']:
        # This is a captured warning, find the original source
        name = record['extra']['mod_name']
        line = record['extra']['lineno']
    if name is None:
        name = "__main__"
    return f"<level>{level}</level> ({name}:{line}): {message}\n"

logfile_id = None

def add_sink(
    file: Union[IO, str, Path] = sys.stderr,
    level: Union[str, int] = "INFO",
    filter: Optional[Callable] = pint.logging.LogFilter(),
) -> None:
    """
    Add a sink (i.e., a file where log messages will be written) to the logger,
    using a format consistent with that used by the default configuration.

    Parameters
    ----------
    file: str, Path, or file-like object
        The location where log messages will be written. A string or `Path`
        will be interpreted as a file name.
    level: str or int
        The minimum level of messages to log. Can be specified as a string
        or as an integer. The mapping (determined by loguru) is as follows:
        "DEBUG"=10, "INFO"=20, "ERROR"=30, "WARNING"=40, "CRITICAL"=50.
    filter: callable, optional
        Filter function to apply to messages. By default, uses the filter
        provided by PINT (an instance of `pint.logging.LogFilter`).
    """
    return logger.add(
        file,
        level=level,
        format=log_format,
        filter=filter,
    )

def setup(
    level: Union[str, int] = "INFO",
    use_stdout: bool = True,
    capture_warnings: bool = True,
) -> None:
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
        or as an integer. The mapping (determined by loguru) is as follows:
        "DEBUG"=10, "INFO"=20, "ERROR"=30, "WARNING"=40, "CRITICAL"=50.
    use_stdout: bool, default True
        If `True` (the default), only messages of "WARNING" level or above
        will be sent to stderr; others will be sent to stdout. Otherwise,
        all messages will be sent to stderr. The default behavior is convenient
        in notebooks, where stderr output is printed with a colored background.
        In a shell, it is preferable to send all messages to stderr so that
        logging output can be redirected separately from stdout.
    capture_warnings: bool, default True
        Whether to re-route warnings created by the `warnings` module through
        the Loguru logger.
    """
    stderr_filter = pint.logging.LogFilter()

    def stdout_filter(record):
        info_or_below = record["level"].no < logger.level("WARNING").no
        return info_or_below and stderr_filter(record)

    logger.remove()
    if use_stdout:
        add_sink(sys.stdout, level=level, filter=stdout_filter)
        add_sink(sys.stderr, level="WARNING", filter=stderr_filter)
    else:
        add_sink(sys.stderr, level=level, filter=stderr_filter)
    if capture_warnings:
        log_warnings()

def log_to_file(
    job_name: str,
    base_dir: Union[str, Path] = ".",
) -> None:
    """
    Enable logging to an automatically generated log file.

    Parameters
    ----------
    job_name: str
        Used as the base of the log file name.
    base_dir: str or Path
        Path in which the log file will be created.
    """
    timestamp = time.strftime('%Y-%m-%d_%H%M%S')
    logfile_name = Path(base_dir) / f"{job_name}.{timestamp}.log"

    # Start a new log file every time you reload the yaml
    if logfile_id is not None:
        logger.remove(logfile_id)

    logfile_id = add_sink(logfile_name)

_showwarning_orig = None
def _showwarning(message, category, filename, lineno, file=None, line=None):
    # Look through sys.modules to find the module object corresponding
    # to the source file and extract the module name from it.
    mod_name = None
    for name, mod in list(sys.modules.items()):
        try:
            # Believe it or not this can fail in some cases:
            # https://github.com/astropy/astropy/issues/2671
            path = Path(getattr(mod, '__file__', ''))
        except Exception:
            continue
        if path == Path(filename).stem:
            mod_name = mod.__name__
            break

    logger.warning(
        f"{message}\n",
        filename=filename,
        mod_name=mod_name,
        lineno=lineno,
    )

def log_warnings():
    """
    Route warnings raised by the `warnings` module through Loguru.
    """
    global _showwarning_orig
    if _showwarning_orig is None:
        _showwarning_orig = warnings.showwarning
        warnings.showwarning = _showwarning
