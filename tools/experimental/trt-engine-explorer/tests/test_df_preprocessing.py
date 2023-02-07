from .util import plan
from trex import change_col_order, drop_columns
import pytest

def test_change_col_order(plan):
    new_col_order = change_col_order(plan.df).columns

    def check_col_order(new_col_order):
        common_cols = list(('Name', 'type', 'Inputs', 'Outputs', 
        'latency.avg_time', 'latency.pct_time', 'total_footprint_bytes',
        'tactic'))

        uncommon_col_seen = False
        for col in new_col_order:
            if col in common_cols and uncommon_col_seen:
                return False
            if col not in common_cols:
                uncommon_col_seen = True
        return True

    assert check_col_order(new_col_order)

def test_drop_columns(plan):
    df = plan.df
    cols_before = df.columns
    assert len(cols_before) > 0
    col_to_drop = cols_before[0]
    drop_columns(df, [col_to_drop])
    cols_after = df.columns
    assert col_to_drop not in cols_after
