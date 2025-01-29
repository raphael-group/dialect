"""TODO: Add docstring."""

import logging
from typing import Any

VERBOSE_LEVEL = 15
logging.addLevelName(VERBOSE_LEVEL, "VERBOSE")


def verbose(
    self: logging.Logger,
    message: str,
    *args: str,
    **kwargs: dict[str, Any],
) -> None:
    """TODO: Add docstring."""
    if self.isEnabledFor(VERBOSE_LEVEL):
        self._log(VERBOSE_LEVEL, message, args, **kwargs)


logging.Logger.verbose = verbose


def configure_logging(verbose: bool = False) -> None:
    """TODO: Add docstring."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(VERBOSE_LEVEL if verbose else logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)

    def module_verbose(message: str, *args: str, **kwargs: dict[str, Any]) -> None:
        if verbose:
            root_logger._log(VERBOSE_LEVEL, message, args, **kwargs)

    logging.verbose = module_verbose
