"""E1 sürekli işaretli artık: yalnız geçmiş, işaret ve birleşim kontrolleri."""

import numpy as np
import pandas as pd

from src.anomaly_e1_sustained import scores_for, sustained


def make_signals(z):
    """z̄ dizisinden spike/drop/A3 sinyalleri (z̄ = spike − drop, A3 = |z̄|)."""
    z = pd.Series(z, index=pd.date_range("2009-06-01", periods=len(z), freq="h"), dtype=float)
    return pd.DataFrame({"A3": z.abs(), "spike": z.clip(lower=0), "drop": (-z).clip(lower=0)})


def test_sustained_uses_only_past_values():
    """t+1 ve sonrası değişince u_t değişmez; ilk w−1 saat NaN."""
    rng = np.random.default_rng(0)
    z = rng.normal(size=50)
    before = sustained(make_signals(z), 6)
    changed = z.copy()
    changed[30:] += 10
    after = sustained(make_signals(changed), 6)
    pd.testing.assert_series_equal(before.iloc[:30], after.iloc[:30])
    assert before.iloc[:5].isna().all() and before.iloc[5:].notna().all()


def test_sign_is_averaged_before_absolute_value():
    """Zıt yönlü hatalar birbirini götürür; sürekli negatif sapma da yakalanır."""
    alternating = sustained(make_signals([1, -1] * 6), 6)
    assert np.allclose(alternating.dropna(), 0.0)
    negative = sustained(make_signals([-2.0] * 8), 6)
    assert np.allclose(negative.dropna(), 2.0)


def test_e1_score_is_max_of_scaled_parts():
    signals = make_signals([1.0] * 8)
    scales = {"A3": 2.0, "u6": 0.5}
    result = scores_for(signals, scales, 6)
    assert np.allclose(result["S0-yerel"], 0.5)
    assert np.allclose(result["E1-w6"].dropna(), 2.0)       # max(1/2, 1/0,5)
    assert result["E1-w6"].iloc[:5].isna().all()            # eksik pencere → NaN (skipna=False)
