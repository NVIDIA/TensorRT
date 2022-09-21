from .util import plan
from trex import summary_dict, EnginePlan
import pytest
import pandas as pd

def test_summary_dict(plan):
    d = summary_dict(plan)
    assert d["Inputs"] == "input1: [1, 3, 224, 224]xFP32 NCHW"
    assert d["Average time"] == "0.470 ms"
    assert d["Layers"] == "72"
    assert d["Weights"] == "3.3 MB"
    assert d["Activations"] == "15.9 MB"

class TestEnginePlan:
    def test_df(self, plan):
        df = plan.df
        assert isinstance(df, (pd.DataFrame))
