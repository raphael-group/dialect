import logging

VERBOSE_LEVEL = 15  # Between INFO (20) and DEBUG (10)
logging.addLevelName(VERBOSE_LEVEL, "VERBOSE")


def verbose(self, message, *args, **kwargs):
    """
    Logs a message with the custom VERBOSE level.
    """
    if self.isEnabledFor(VERBOSE_LEVEL):
        self._log(VERBOSE_LEVEL, message, args, **kwargs)


logging.Logger.verbose = verbose


def configure_logging(verbose=False):
    """
    Configure logging with an optional verbose mode.

    :param verbose: Enable verbose logging if True. Defaults to False.
    :type verbose: bool
    """
    logging.basicConfig(
        level=VERBOSE_LEVEL if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if verbose:
        logging.getLogger().setLevel(VERBOSE_LEVEL)
