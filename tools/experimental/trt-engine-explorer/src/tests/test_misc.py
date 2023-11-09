from .util import plan
from trex import group_count, group_sum_attr
import pytest
import pandas as pd

def test_group_count(plan):
    gc = group_count(plan.df, "type").set_index("type")
    gc_exp = plan.df.groupby(["type"]).size()
    assert (gc.loc['Convolution'] == gc_exp.loc['Convolution']).all()
    assert (gc.loc['Pooling'] == gc_exp.loc['Pooling']).all()
    assert (gc.loc['Reformat'] == gc_exp.loc['Reformat']).all()

def test_group_sum_attr(plan):
    gsa = group_sum_attr(plan.df,"type", "latency.avg_time").set_index("type")
    gsa_exp = plan.df.groupby(["type"]).sum()[["latency.avg_time"]]
    assert (gsa.loc['Convolution'] == gsa_exp.loc['Convolution']).all()
    assert (gsa.loc['Pooling'] == gsa_exp.loc['Pooling']).all()
    assert (gsa.loc['Reformat'] == gsa_exp.loc['Reformat']).all()
