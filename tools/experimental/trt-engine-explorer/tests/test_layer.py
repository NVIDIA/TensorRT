from .util import test_engine1_prefix_path
from trex import import_graph_file, Layer, Activation
import pytest

class TestLayer:
    def test_initialization(self):
        graph_file = f'{test_engine1_prefix_path}.graph.json'
        raw_layers, _ = import_graph_file(graph_file)
        layer = Layer(raw_layers[0])

        assert layer.name == 'QuantizeLinear_2'
        assert layer.type == 'Reformat'
        assert layer.subtype == 'Reformat'
        assert isinstance(layer.inputs[0], (Activation))
        assert isinstance(layer.outputs[0], (Activation))
        assert layer.outputs_size_bytes == 150528
        assert layer.precision == 'FP32'
        assert layer.inputs_size_bytes == 602112
        assert layer.total_io_size_bytes == 752640
        assert layer.total_footprint_bytes == 752640
