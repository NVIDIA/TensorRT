
import numpy as np
from polygraphy.tools.util.args import parse_meta


class TestParseMeta(object):
    def test_parse_shape_only(self):
        meta_args = ["input0,1x3x224x224"]
        meta = parse_meta(meta_args, includes_dtype=False)
        assert meta["input0"].shape == (1, 3, 224, 224)
        assert meta["input0"].dtype is None


    def test_parse_shape_single_dim(self):
        meta_args = ["input0,1"]
        meta = parse_meta(meta_args, includes_dtype=False)
        assert meta["input0"].shape == (1, )


    def test_parse_dtype_only(self):
        meta_args = ["input0,float32"]
        meta = parse_meta(meta_args, includes_shape=False)
        assert meta["input0"].shape is None
        assert meta["input0"].dtype == np.float32


    def test_parse_shape_dtype(self):
        meta_args = ["input0,1x3x224x224,float32"]
        meta = parse_meta(meta_args)
        assert meta["input0"].shape == (1, 3, 224, 224)
        assert meta["input0"].dtype == np.float32


    def test_parse_shape_dtype_auto(self):
        meta_args = ["input0,auto,auto"]
        meta = parse_meta(meta_args)
        assert meta["input0"].shape is None
        assert meta["input0"].dtype is None


    def test_parse_shape_with_dim_param_single_quote(self):
        meta_args = ["input0,'batch'x3x224x224"]
        meta = parse_meta(meta_args, includes_dtype=False)
        assert meta["input0"].shape == ("batch", 3, 224, 224)


    def test_parse_shape_with_dim_param_double_quote(self):
        meta_args = ['input0,"batch"x3x224x224']
        meta = parse_meta(meta_args, includes_dtype=False)
        assert meta["input0"].shape == ("batch", 3, 224, 224)


    def test_parse_shape_with_dim_param_including_x(self):
        meta_args = ["input0,'batchx'x3x224x224"]
        meta = parse_meta(meta_args, includes_dtype=False)
        assert meta["input0"].shape == ("batchx", 3, 224, 224)
