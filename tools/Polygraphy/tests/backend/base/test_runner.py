import numpy as np
import pytest
from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnxBytes
from polygraphy.common import PolygraphyException
from tests.models.meta import ONNX_MODELS


def test_infer_raises_if_runner_inactive():
    runner = OnnxrtRunner(SessionFromOnnxBytes(ONNX_MODELS["identity"].loader))
    feed_dict = {"x": np.ones((1, 1, 2, 2), dtype=np.float32)}

    with pytest.raises(PolygraphyException):
        runner.infer(feed_dict)
