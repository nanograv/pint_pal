import contextlib
from io import StringIO

from loguru import logger
import pint_pal.logger

def print_log_messages():
    """
    Print a series of log messages at different levels for testing purposes.
    """
    logger.debug("Debug message")
    logger.info("Information message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

def test_stderr_logging():
    """
    Check that logging to stderr works correctly. Capture output by redirecting
    stderr to an in-memory StringIO object, and check that messages are printed
    at the correct logging level.
    """
    buffer = StringIO()

    with contextlib.redirect_stderr(buffer):
        pint_pal.logger.setup(level="INFO", use_stdout=False)
        print_log_messages()

    buffer.seek(0)
    messages = buffer.readlines()
    assert len(messages) == 4
    assert "INFO" in messages[0]

def test_combo_logging():
    """
    Check that logging to a combination of stdout and stderr works correctly.
    Capture output by redirecting both input streams to in-memory StringIO
    objects, and check that the correct messages have been printed to each.
    """
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()

    with (contextlib.redirect_stdout(stdout_buffer),
          contextlib.redirect_stderr(stderr_buffer)):
        pint_pal.logger.setup(level="DEBUG", use_stdout=True)
        print_log_messages()

    stdout_buffer.seek(0)
    stdout_messages = stdout_buffer.readlines()
    assert len(stdout_messages) == 2
    assert "DEBUG" in stdout_messages[0]

    stderr_buffer.seek(0)
    stderr_messages = stderr_buffer.readlines()
    assert len(stderr_messages) == 3
    assert "WARNING" in stderr_messages[0]
