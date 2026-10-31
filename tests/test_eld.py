"""ELD veri adımı: müşteri seçimi, yaz saati düzeltmesi, saatlik toplama, config birleştirme."""

import numpy as np
import pandas as pd
import pytest

from src.data.eld import active_clients, deep_merge, dst_change_days, eld_config, fix_dst, to_hourly
from src.data.load import load_config


def quarter_series(start, end, value=100.0):
    """Aralık sonu damgalı sabit 15 dakikalık seri."""
    index = pd.date_range(pd.Timestamp(start) + pd.Timedelta(minutes=15), end, freq="15min")
    return pd.Series(value, index=index, dtype=np.float64)


def test_dst_days_are_last_sundays():
    march, october = dst_change_days([2012, 2013, 2014])
    assert [d.date().isoformat() for d in march] == ["2012-03-25", "2013-03-31", "2014-03-30"]
    assert [d.date().isoformat() for d in october] == ["2012-10-28", "2013-10-27", "2014-10-26"]


def test_fix_dst_interpolates_march_zeros_and_halves_october_double_hour():
    total = quarter_series("2012-03-01", "2012-11-30")
    stamps = ["01:00", "01:15", "01:30", "01:45"]
    total.loc[[pd.Timestamp(f"2012-03-25 {s}") for s in stamps]] = 0.0
    total.loc[pd.Timestamp("2012-03-25 02:00")] = 200.0          # ara değer eğimi görünsün
    total.loc[[pd.Timestamp(f"2012-10-28 {s}") for s in stamps]] = 200.0

    fixed, changed, report = fix_dst(total, [2012])
    march = fixed.loc["2012-03-25 01:00":"2012-03-25 01:45"].to_numpy()
    np.testing.assert_allclose(march, [120.0, 140.0, 160.0, 180.0])   # 100 → 200 doğrusal
    np.testing.assert_allclose(fixed.loc["2012-10-28 01:00":"2012-10-28 01:45"], 100.0)
    assert changed.sum() == 8 and len(report) == 2
    assert total.loc["2012-10-28 01:00"] == 200.0                     # girdi değişmedi


def test_fix_dst_raises_when_expected_pattern_is_missing():
    total = quarter_series("2012-03-01", "2012-11-30")                # hiç kalıntı yok
    with pytest.raises(ValueError, match="Mart"):
        fix_dst(total, [2012])


def test_hourly_mean_uses_interval_end_stamps_and_flags_fixed_hours():
    total = pd.Series([1.0, 2.0, 3.0, 4.0, 10.0, 10.0, 10.0, 10.0],
                      index=pd.date_range("2012-01-01 00:15", periods=8, freq="15min"))
    changed = pd.Series(False, index=total.index)
    changed.iloc[4] = True                                            # 01:15 damgası → saat 01
    hourly = to_hourly(total, changed)
    # Saat 00 = (00:00, 01:00] = damgalar 00:15, 00:30, 00:45, 01:00.
    assert hourly.loc["2012-01-01 00:00", "load"] == pytest.approx(2.5)
    assert hourly.loc["2012-01-01 01:00", "load"] == pytest.approx(10.0)
    assert hourly["is_imputed"].tolist() == [0, 1]


def test_active_clients_excludes_late_starters():
    index = pd.date_range("2011-12-31 23:00", periods=8, freq="15min")
    raw = pd.DataFrame({"early": 1.0, "late": [0.0] * 5 + [3.0] * 3}, index=index, dtype=np.float32)
    assert list(active_clients(raw, "2012-01-01 00:00")) == ["early"]


def test_eld_config_keeps_household_hyperparameters_and_eld_splits():
    config = eld_config("configs/optuna_seq2seq_best.yaml")
    household = load_config("configs/optuna_seq2seq_best.yaml")["forecast"]
    assert config["forecast"] == household                            # hiperparametreler aynen
    assert config["split"]["test"] == {"start": "2014-07-01", "end": "2014-12-31"}
    assert config["preprocess"]["target"] == "total_load_kw"
    assert config["paths"]["experiments_file"] != load_config()["paths"]["experiments_file"]
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}
    assert set(config["split"]) == {"train", "val", "test"}            # household val_dev sızmadı
    assert config["anomaly"]["residual"]["ensemble"] == {"ELD-F10-opt": 0.8, "ELD-F5-opt": 0.2}
    assert config["protocol"]["scenarios"]["main"]["injection"]["spike"]["sigma_low"] == 3.0


def test_eld_splits_are_chronological_and_disjoint():
    config = eld_config()
    ends = [pd.Timestamp(config["split"][name]["end"]) for name in ("train", "val")]
    starts = [pd.Timestamp(config["split"][name]["start"]) for name in ("val", "test")]
    for end, start in zip(ends, starts):
        assert end + pd.Timedelta(days=1) == start
