from polygraphy.backend.onnx.loader import OnnxFromPath, OnnxFromTfGraph, ModifyOnnx, SaveOnnx, BytesFromOnnx
from polygraphy.backend.onnx.runner import OnnxTfRunner


def register_callback():
    from polygraphy.logger.logger import G_LOGGER

    def set_onnx_logging_level(sev):
        import warnings
        if sev >= G_LOGGER.INFO:
            warnings.filterwarnings("ignore")
        else:
            warnings.filterwarnings("default")

    G_LOGGER.register_callback(set_onnx_logging_level)

register_callback()
