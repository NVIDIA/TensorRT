from polygraphy.backend.trt.algorithm_selector import *
from polygraphy.backend.trt.calibrator import *
from polygraphy.backend.trt.config import *
from polygraphy.backend.trt.loader import *
from polygraphy.backend.trt.profile import *
from polygraphy.backend.trt.runner import *
from polygraphy.backend.trt.util import *


def register_logger_callback():
    from polygraphy.logger import G_LOGGER

    def set_trt_logging_level(severity_trie):
        from polygraphy import mod
        import os

        trt = mod.lazy_import("tensorrt")

        if not mod.has_mod("tensorrt"):
            return

        if mod.version(trt.__version__) >= mod.version("8.0"):
            # For TensorRT 8.0 and newer, we use a custom logger implementation
            # which redirects all messages into Polygraphy's logger for better integration.
            # Thus, this callback is unnecessary.
            return

        sev = severity_trie.get(G_LOGGER.module_path(os.path.dirname(__file__)))
        if sev >= G_LOGGER.CRITICAL:
            get_trt_logger().min_severity = trt.Logger.INTERNAL_ERROR
        elif sev >= G_LOGGER.ERROR:
            get_trt_logger().min_severity = trt.Logger.ERROR
        elif sev >= G_LOGGER.INFO:
            get_trt_logger().min_severity = trt.Logger.WARNING
        elif sev >= G_LOGGER.VERBOSE:
            get_trt_logger().min_severity = trt.Logger.INFO
        else:
            get_trt_logger().min_severity = trt.Logger.VERBOSE

    G_LOGGER.register_callback(set_trt_logging_level)  # Will be registered when this backend is imported.


register_logger_callback()
