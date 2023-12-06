import logging


def get_logger(name):
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    if not logger.handlers:  # Check if handlers already exist
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger
