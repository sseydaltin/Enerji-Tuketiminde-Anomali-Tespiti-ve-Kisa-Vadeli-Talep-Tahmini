"""Aşama 3 kısa olay sinyalleri testleri: işaret, geçmiş bilgisi, ölçek, alarm birleştirme."""

import numpy as np
import pandas as pd

from src.anomaly_short_event import (
    EXPERIMENTS,
    change_scores,
    change_table,
    combined_score,
    signal_frame,
    weight_grid,
)
from src.models.residual_detector import residual_std_table
from src.utils.anomaly_metrics import event_f1_max_threshold, fill_alarm_gaps

TARGET = "Global_active_power"


def make_frame(hours=24 * 21, seed=0):
    """Saatlik sentetik tablo (yalnız hedef ve is_imputed)."""
    index = pd.date_range("2009-01-05", periods=hours, freq="h")
    rng = np.random.default_rng(seed)
    return pd.DataFrame({TARGET: 1.0 + rng.random(hours), "is_imputed": np.zeros(hours, dtype="int8")},
                        index=index)


def make_predictions(frame, offset):
    """Her ufukta y − offset tahmini (artık = +offset) ve birim std tabloları."""
    predictions = pd.DataFrame({h: frame[TARGET] - offset for h in range(1, 25)}, index=frame.index)
    unit = residual_std_table(frame, frame[TARGET] - 0.0, TARGET).assign(std=1.0)
    return predictions, {h: unit for h in range(1, 25)}


def test_signed_residual_separates_spike_and_drop():
    """Pozitif artık yalnız spike'a, negatif artık yalnız drop'a gider; A3 = |artık| ortalaması."""
    frame = make_frame()
    change_tab = change_table(frame, TARGET)
    predictions, std = make_predictions(frame, offset=2.0)            # y − ŷ = +2
    signals = signal_frame(frame, predictions, std, change_tab, TARGET)
    assert np.allclose(signals["spike"], 2.0) and np.allclose(signals["drop"], 0.0)
    assert np.allclose(signals["A3"], 2.0)
    predictions, std = make_predictions(frame, offset=-3.0)           # y − ŷ = −3
    signals = signal_frame(frame, predictions, std, change_tab, TARGET)
    assert np.allclose(signals["drop"], 3.0) and np.allclose(signals["spike"], 0.0)
    assert np.allclose(signals["A3"], 3.0)


def test_change_signal_uses_only_past_values():
    """t+1 ve sonrasını değiştirmek t'deki değişim skorunu değiştirmez (gelecek bilgisi yok)."""
    frame = make_frame()
    table = change_table(frame, TARGET)
    before = change_scores(frame, table, TARGET)
    changed = frame.copy()
    changed.iloc[200:, 0] += 5.0
    after = change_scores(changed, table, TARGET)
    pd.testing.assert_series_equal(before.iloc[:200], after.iloc[:200])
    assert np.isnan(before.iloc[0])                                   # ilk saatin geçmişi yok


def test_change_table_excludes_imputed_hours():
    """Doldurulmuş saat (ve ondan sonraki saatin farkı) tabloya girmez."""
    frame = make_frame()
    frame.iloc[50, 1] = 1                                             # 50. saat doldurulmuş
    dirty = frame.copy()
    dirty.iloc[50, 0] = 100.0                                         # değeri tabloyu etkilememeli
    pd.testing.assert_frame_equal(change_table(dirty, TARGET), change_table(frame, TARGET))


def test_s0_combined_score_is_scaled_a3_only():
    """S0'da birleşik skor yalnız A3 / q_A3; ek sinyaller etkisiz."""
    frame = make_frame()
    signals = pd.DataFrame({"A3": [1.0, 2.0], "spike": [9.0, 9.0], "drop": [9.0, 9.0],
                            "change": [9.0, 9.0]}, index=frame.index[:2])
    scales = {"A3": 2.0, "spike": 1.0, "drop": 1.0, "change": 1.0}
    assert np.allclose(combined_score(signals, scales, {}), [0.5, 1.0])
    assert np.allclose(combined_score(signals, scales, {"drop": 0.5}), [4.5, 4.5])
    assert weight_grid("S0") == [{}]
    assert len(weight_grid("S5")) == 5 ** len(EXPERIMENTS["S5"])


def test_fill_alarm_gaps_only_between_alarms():
    mask = np.array([0, 1, 0, 1, 0, 0, 1, 0], dtype=bool)
    assert fill_alarm_gaps(mask, 0).tolist() == mask.tolist()
    assert fill_alarm_gaps(mask, 1).astype(int).tolist() == [0, 1, 1, 1, 0, 0, 1, 0]
    assert fill_alarm_gaps(mask, 2).astype(int).tolist() == [0, 1, 1, 1, 1, 1, 1, 0]
    assert fill_alarm_gaps(mask, 2) is not mask                       # girdi değişmez
    assert mask.astype(int).tolist() == [0, 1, 0, 1, 0, 0, 1, 0]


def test_event_threshold_default_gap_keeps_old_behavior():
    """max_gap=0 eski event_f1_max_threshold davranışıdır."""
    rng = np.random.default_rng(1)
    types = np.array(["none"] * 200, dtype=object)
    types[50:53] = "spike"
    scores = rng.random(200)
    scores[50:53] += 2
    copies = [(types, scores, np.ones(200, dtype=bool))]
    quantiles = np.linspace(0.7, 0.999, 50)
    assert event_f1_max_threshold(copies, quantiles) == event_f1_max_threshold(copies, quantiles, max_gap=0)
