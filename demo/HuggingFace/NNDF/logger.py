import logging

G_LOGGER = logging.getLogger("OSS")
G_LOGGER.DEBUG = logging.DEBUG
G_LOGGER.INFO = logging.INFO
G_LOGGER.WARNING = logging.WARNING
G_LOGGER.ERROR = logging.ERROR

formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s")
stream = logging.StreamHandler()
stream.setFormatter(formatter)
G_LOGGER.addHandler(stream)
