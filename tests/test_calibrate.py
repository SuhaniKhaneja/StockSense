import numpy as np
import pandas as pd

from stocksense.calibrate import ALPHA, apply_offset, coverage, evaluate, upper_offset


def _wide_interval_sample(n, seed):
    """Demand-like outcomes with an interval that is deliberately far too wide."""
    rng = np.random.default_rng(seed)
    y = rng.gamma(2.0, 3.0, n) * (rng.random(n) < 0.6)
    p = np.full(n, 5.0)
    lo = np.zeros(n)
    hi = np.full(n, 40.0)
    return y, p, lo, hi


def test_upper_offset_pulls_coverage_toward_target():
    y, p, lo, hi = _wide_interval_sample(6000, 0)
    assert coverage(y, lo, hi) > 0.97                       # starts far too wide
    q = upper_offset(y, lo, hi)
    assert q < 0                                             # so the upper bound shrinks
    y2, p2, lo2, hi2 = _wide_interval_sample(6000, 1)        # fresh, held-out sample
    lo3, hi3 = apply_offset(p2, lo2, hi2, q)
    assert abs(coverage(y2, lo3, hi3) - (1 - ALPHA)) < 0.05


def _mixed_volume_sample(n, seed):
    """Products whose volume differs by 200x, with intervals that are too wide for all of them."""
    r = np.random.default_rng(seed)
    scale = np.exp(r.uniform(0, np.log(200), n))
    y = r.gamma(2.0, scale / 2) * (r.random(n) < 0.6)
    return y, 0.6 * scale, np.zeros(n), 3.0 * scale + 1


def test_offset_scales_with_interval_width():
    y, p, lo, hi = _mixed_volume_sample(20000, 1)
    q = upper_offset(y, lo, hi)
    y2, p2, lo2, hi2 = _mixed_volume_sample(20000, 2)
    lo3, hi3 = apply_offset(p2, lo2, hi2, q)
    assert coverage(y2, lo2, hi2) > 0.97
    assert abs(coverage(y2, lo3, hi3) - 0.8) < 0.03
    assert (hi3 - lo3).mean() < 0.6 * (hi2 - lo2).mean()    # a fixed shift in units cannot do this


def test_apply_offset_keeps_point_inside_and_nonnegative():
    p = np.array([5.0, 0.0, 9.0])
    lo = np.array([0.0, 0.0, 2.0])
    hi = np.array([8.0, 1.0, 12.0])
    lo2, hi2 = apply_offset(p, lo, hi, q=-6.0)
    assert (lo2 >= 0).all() and (lo2 <= p).all() and (hi2 >= p).all()
    assert np.allclose(lo2, lo)                              # lower bound is never moved


def test_zero_demand_days_stay_covered():
    y = np.zeros(100)
    p = np.zeros(100)
    lo, hi = np.zeros(100), np.full(100, 10.0)
    lo2, hi2 = apply_offset(p, lo, hi, upper_offset(y, lo, hi))
    assert coverage(y, lo2, hi2) == 1.0


def test_evaluate_uses_only_older_folds_and_improves_coverage():
    rows = []
    for fold in range(4):
        y, p, lo, hi = _wide_interval_sample(3000, 10 + fold)
        rows.append(pd.DataFrame({"model": "Gradient boosting", "fold": fold, "y": y, "p": p, "lo": lo, "hi": hi}))
    table = evaluate(pd.concat(rows))
    assert list(table["fold"].astype(str)) == ["0", "1", "2", "ALL"]   # oldest fold is calibration-only
    allrow = table[table["fold"].astype(str) == "ALL"].iloc[0]
    assert abs(allrow["coverage_after"] - 0.8) < abs(allrow["coverage_before"] - 0.8)
    assert allrow["width_after"] < allrow["width_before"]
