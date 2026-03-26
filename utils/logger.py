# logger.py

"""Logging utilities."""


def get_logger(name='miqas'):
    import logging
    logger = logging.getLogger(name)
    return logger
