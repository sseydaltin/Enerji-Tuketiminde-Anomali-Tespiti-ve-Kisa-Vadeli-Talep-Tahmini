"""A3 + taban (v2): taban kuralı, zaman boşluğu, doldurulmuş saat, OR birleşimi, ortak maske ve demo tutarlılığı."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.anomaly_floor_channel import FLOOR_QUANTILE, floor_flags
from src.anomaly_v2 import BASE_METHOD, V2_METHOD, train_floor, v2_score

FLOOR = 0.155
ROOT = Path(__file__).resolve().parents[1]


def hours(n, start="2010-04-01"):
    return pd.date_range(start, periods=n, freq="h")


def test_two_consecutive_low_hours_alarm_single_low_hour_does_not():
    values = np.array([1.0, 0.05, 1.0, 0.05, 0.05, 0.05, 1.0])
    flags = floor_flags(values, FLOOR, index=hours(7), imputed=np.zeros(7, dtype=bool))
    # Tek düşük saat (1) alarm vermez; 3–5 ardışık düşük: alarm 4. ve 5. saatte (ilk düşük saatte değil).
    assert flags.tolist() == [False, False, False, False, True, True, False]


def test_time_gap_breaks_consecutive_hours():
    index = pd.DatetimeIndex(["2010-04-01 00:00", "2010-04-01 01:00", "2010-04-01 03:00", "2010-04-01 04:00"])
    values = np.array([0.05, 0.05, 0.05, 1.0])
    flags = floor_flags(values, FLOOR, index=index)
    assert flags.tolist() == [False, True, False, False]     # 01:00 → 03:00 arası boşluk: ardışık değil
    assert floor_flags(values, FLOOR).tolist() == [False, True, True, False]   # korumasız (zip) davranış


def test_imputed_hour_is_not_counted_as_low():
    values = np.array([0.05, 0.05, 0.05])
    imputed = np.array([False, True, False])
    flags = floor_flags(values, FLOOR, index=hours(3), imputed=imputed)
    assert flags.tolist() == [False, False, False]


def test_v2_alarm_is_or_of_a3_and_floor_and_uses_same_mask():
    a3 = np.array([5.0, 1.0, np.nan, 1.0, 1.0])               # yumuşatılmış A3 skoru
    values = np.array([1.0, 1.0, 0.05, 0.05, 1.0])
    threshold = 3.0
    score, flags = v2_score(a3, threshold, values, FLOOR, hours(5), np.zeros(5, dtype=bool))
    with np.errstate(invalid="ignore"):
        expected = (a3 >= threshold) | flags
    assert ((score >= 1.0) == expected).all()
    assert (score >= 1.0).tolist() == [True, False, False, True, False]
    # Ortak değerlendirme maskesi: A3 skoru olmayan saat iki yöntemde de değerlendirilmez.
    valid = np.isfinite(a3)
    assert ((score >= 1.0) & valid).tolist() == [True, False, False, True, False]


def test_train_floor_is_0005_quantile_of_non_imputed_train():
    train = pd.DataFrame({"y": np.arange(1, 1001, dtype=float) / 1000, "is_imputed": 0}, index=hours(1000))
    train.loc[train.index[:5], "is_imputed"] = 1               # en düşük 5 değer doldurulmuş
    assert FLOOR_QUANTILE == 0.005
    expected = train.loc[train["is_imputed"] == 0, "y"].quantile(0.005)
    assert train_floor(train, "y") == pytest.approx(expected)
    assert train_floor(train, "y") < 0.02                     # 0.5 (medyan) değil


@pytest.mark.skipif(not (ROOT / "models/F10-opt_best.pt").is_file()
                    or not (ROOT / "data/processed/hourly.parquet").is_file(),
                    reason="yerel veri veya F10-opt ağırlığı yok")
def test_demo_v2_flags_match_shared_rule_on_test():
    """Demo, deney betiğiyle aynı v2_score'u kullanır: alarmlar A3 alarmı VEYA taban bayrağı."""
    from app.anomaly_services import (
        analyze_test_range,
        anomaly_demo_paths,
        checkpoint_signature,
        flagged_hours,
        load_anomaly_calibration,
    )
    from app.services import load_f13_models, load_household_data

    config, data_path, models_dir, tables_dir, features = anomaly_demo_paths()
    frame = load_household_data(str(data_path))
    calibration = load_anomaly_calibration(str(tables_dir))
    models = load_f13_models(str(models_dir), features)
    start, end = pd.Timestamp("2010-04-05"), pd.Timestamp("2010-11-20 23:00")
    result = analyze_test_range(frame, config, start, end, True, "drop", "main", 2, calibration,
                                checkpoint_signature(models_dir), models, v2=True)
    assert result["floor_kw"] == pytest.approx(0.155, abs=0.01)      # train'den hesaplanır
    a3 = flagged_hours(result, BASE_METHOD)
    v2 = flagged_hours(result, V2_METHOD)
    injected = result["injected_kw"]
    flags = floor_flags(injected.to_numpy(), result["floor_kw"], index=injected.index,
                        imputed=result["imputed"].to_numpy())
    valid = result["valid"].to_numpy()
    # Aralık başındaki saat de bir önceki saate bakabildiği için ilk saat hariç karşılaştırılır.
    assert ((a3 | (flags & valid))[1:] == v2[1:]).all()
    assert v2.sum() >= a3.sum()
    # v2 seçilmediğinde sonuç eskisi gibi kalır.
    plain = analyze_test_range(frame, config, start, end, True, "drop", "main", 2, calibration,
                               checkpoint_signature(models_dir), models)
    assert V2_METHOD not in plain["scores"].columns and plain["thresholds"] == calibration["thresholds"]
