import logging

VERBOSE_LEVEL = 15
logging.addLevelName(VERBOSE_LEVEL, "VERBOSE")


def verbose(self, message, *args, **kwargs):
    """Log a message with the custom VERBOSE level."""
    if self.isEnabledFor(VERBOSE_LEVEL):
        self._log(VERBOSE_LEVEL, message, args, **kwargs)


# Attach the verbose method to logging.Logger
logging.Logger.verbose = verbose


def configure_logging(verbose=False):
    """
    Configures logging with optional verbose mode.

    :param verbose: Enable verbose logging if True, default is False.
    """
    # Reset logging configuration
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

    # Ensure verbose method is accessible globally
    def module_verbose(message, *args, **kwargs):
        if verbose:
            root_logger._log(VERBOSE_LEVEL, message, args, **kwargs)

    logging.verbose = module_verbose
