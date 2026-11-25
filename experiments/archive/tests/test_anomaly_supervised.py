"""Gözetimli dedektör: özellikler yalnız geçmişe bakar; yasak bilgi (tip, ham kW, alt sayaç) yok."""

import numpy as np
import pandas as pd

from src.anomaly_short_event import change_table
from src.anomaly_supervised import FEATURES, PERIODS, build_features, diverse_settings
from src.models.residual_detector import residual_std_table

T = "Global_active_power"


def make_frame(days=40):
    index = pd.date_range("2009-01-05", periods=24 * days, freq="h")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({T: 1.0 + rng.random(len(index)), "is_imputed": 0,
                          "is_weekend": (index.dayofweek >= 5).astype(int), "is_holiday": 0,
                          "hour_sin": np.sin(2 * np.pi * index.hour / 24),
                          "hour_cos": np.cos(2 * np.pi * index.hour / 24)}, index=index)
    return frame


def features_for(frame):
    predictions = pd.DataFrame({h: frame[T].shift(h) for h in range(1, 25)}, index=frame.index)  # yalnız geçmiş
    unit = residual_std_table(frame, frame[T] * 0, T).assign(std=1.0)
    floor = pd.Series(0.1, index=pd.MultiIndex.from_product([range(24), [0, 1]]))
    return build_features(frame, predictions, {h: unit for h in range(1, 25)}, change_table(frame, T),
                          floor, 0.15, T)


def test_features_use_only_past_values():
    frame = make_frame()
    t = 24 * 30 + 5
    before = features_for(frame)
    changed = frame.copy()
    changed.iloc[t + 1:, 0] += 50.0
    after = features_for(changed)
    pd.testing.assert_series_equal(before.iloc[t], after.iloc[t])


def test_no_forbidden_inputs_and_periods_end_before_evaluation():
    assert not {"Global_active_power", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "other_energy",
                "anomaly_type", "is_anomaly", "month_sin", "month_cos"} & set(FEATURES)
    assert pd.Timestamp(PERIODS["winter_dev"]["inner"][1]) < pd.Timestamp("2009-10-01")
    assert pd.Timestamp(PERIODS["summer_dev"]["inner"][1]) < pd.Timestamp("2009-06-01")
    # Fit dönemi ile iç doğrulama arasında en az 7 gün boşluk.
    for periods in PERIODS.values():
        gap = pd.Timestamp(periods["inner"][0]) - pd.Timestamp(periods["fit"][1])
        assert gap >= pd.Timedelta(days=7)


def test_training_injection_is_diversified_not_evaluation_constant():
    values = {diverse_settings({}, i)["drop"]["value_kw"] for i in range(8)}
    assert len(values) == 4 and 0.05 in values
