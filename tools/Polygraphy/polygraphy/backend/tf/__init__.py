from polygraphy.backend.tf.loader import *
from polygraphy.backend.tf.runner import *


def register_logger_callback():
    from polygraphy.logger import G_LOGGER

    def set_tf_logging_level(sev):
        import os
        from polygraphy import mod

        tf = mod.lazy_import("tensorflow", version="<2.0")
        if not mod.has_mod(tf):
            return

        if sev > G_LOGGER.WARNING:
            tf_sev = tf.compat.v1.logging.ERROR
            tf_logging_level = "3"
        elif sev > G_LOGGER.INFO:
            tf_sev = tf.compat.v1.logging.WARN
            tf_logging_level = "2"
        elif sev > G_LOGGER.VERBOSE:
            tf_sev = tf.compat.v1.logging.INFO
            tf_logging_level = "1"
        else:
            tf_sev = tf.compat.v1.logging.DEBUG
            tf_logging_level = "0"

        tf.compat.v1.logging.set_verbosity(tf_sev)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = tf_logging_level

    G_LOGGER.register_callback(set_tf_logging_level)  # Will be registered when this backend is imported.


register_logger_callback()
