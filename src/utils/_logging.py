import logging
import sys


def setup_logger(level=logging.INFO, format_string=None):
    if format_string is None:
        format_string = "%(asctime)s - %(levelname)s - %(message)s"

    logger_ = logging.getLogger()
    logger_.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)

    logger_.addHandler(handler)

    return logger_


logger = setup_logger()
