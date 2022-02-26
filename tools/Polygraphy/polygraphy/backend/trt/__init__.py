from polygraphy.backend.trt.algorithm_selector import *
from polygraphy.backend.trt.calibrator import *
from polygraphy.backend.trt.loader import *
from polygraphy.backend.trt.profile import *
from polygraphy.backend.trt.runner import *
from polygraphy.backend.trt.util import *


def register_logger_callback():
    from polygraphy.logger import G_LOGGER

    def set_trt_logging_level(sev):
        from polygraphy import mod

        trt = mod.lazy_import("tensorrt")
        if not mod.has_mod(trt):
            return

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
