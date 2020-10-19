from polygraphy.backend.trt.loader import EngineFromBytes, NetworkFromOnnxBytes, NetworkFromOnnxPath, ModifyNetwork, Profile, CreateConfig, EngineFromNetwork, SaveEngine, LoadPlugins, CreateNetwork
from polygraphy.backend.trt.runner import TrtRunner
from polygraphy.backend.trt.calibrator import Calibrator
from polygraphy.backend.trt.util import TRT_LOGGER


def register_callback():
    from polygraphy.logger.logger import G_LOGGER

    def set_trt_logging_level(sev):
        import tensorrt as trt

        if sev >= G_LOGGER.CRITICAL:
            TRT_LOGGER.min_severity = trt.Logger.INTERNAL_ERROR
        elif sev >= G_LOGGER.ERROR:
            TRT_LOGGER.min_severity = trt.Logger.ERROR
        elif sev >= G_LOGGER.INFO:
            TRT_LOGGER.min_severity = trt.Logger.WARNING
        elif sev >= G_LOGGER.VERBOSE:
            TRT_LOGGER.min_severity = trt.Logger.INFO
        else:
            TRT_LOGGER.min_severity = trt.Logger.VERBOSE

    G_LOGGER.register_callback(set_trt_logging_level) # Will be registered when this runner is imported.

register_callback()
