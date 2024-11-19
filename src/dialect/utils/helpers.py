import os
import logging


def check_file_exists(fn):
    """
    Checks if the specified file exists.

    @param fn: Path to the file to validate.
    Raises FileNotFoundError if the file does not exist.
    """
    logging.info(f"Validating input file: {fn}")
    if not os.path.exists(fn):
        raise FileNotFoundError(f"File not found: {fn}")
