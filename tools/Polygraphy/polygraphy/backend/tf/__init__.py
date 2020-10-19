from polygraphy.backend.tf.loader import OptimizeGraph, GraphFromKeras, GraphFromFrozen, GraphFromCkpt, UseTfTrt, ModifyGraph, SaveGraph, CreateConfig, SessionFromGraph
from polygraphy.backend.tf.runner import TfRunner


def register_callback():
    from polygraphy.logger.logger import G_LOGGER

    def set_tf_logging_level(sev):
        import os
        import tensorflow as tf

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
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = tf_logging_level

    G_LOGGER.register_callback(set_tf_logging_level) # Will be registered when this runner is imported.

register_callback()
