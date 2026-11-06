"""A6 deneyinin kuralları: geçmişe bakma, test kümesini kullanmama, seçim ölçütü."""

import numpy as np
import pandas as pd
import pytest

from src.anomaly_dual import (
    DUAL,
    DUAL_FLOOR,
    STAGES,
    dual_alarm,
    floor_alarm,
    floor_threshold,
    pooled_event_f1,
    run_hold,
    select_settings,
)


def test_floor_alarm_starts_at_kth_hour_and_needs_consecutive_hours():
    """Alarm k. saatte başlar; araya normal saat girerse sayaç sıfırlanır."""
    values = [1.0, 0.1, 0.1, 0.1, 1.0, 0.1, 1.0]
    assert floor_alarm(values, 0.2, 2).tolist() == [False, False, True, True, False, False, False]
    assert floor_alarm(values, 0.2, 1).tolist() == [False, True, True, True, False, True, False]


def test_floor_alarm_does_not_use_future_values():
    """Gelecek değerler değişince geçmiş kararlar değişmemeli (sızıntı yok)."""
    values = np.array([0.1, 0.1, 1.0, 0.1, 0.1, 0.1])
    changed = values.copy()
    changed[3:] = 5.0
    assert floor_alarm(values, 0.2, 2)[:3].tolist() == floor_alarm(changed, 0.2, 2)[:3].tolist()


def test_floor_threshold_uses_only_clean_train_hours():
    """Doldurulmuş saatler taban eşiğini etkilememeli."""
    train = pd.DataFrame({"y": [1.0, 2.0, 3.0, 0.0], "is_imputed": [0, 0, 0, 1]})
    assert floor_threshold(train, "y", 50) == pytest.approx(2.0)


def test_dual_alarm_is_or_of_smoothed_and_raw_and_ignores_nan():
    """Yumuşatılmış ≥ T1 veya ham ≥ T2; NaN alarm üretmez."""
    smoothed = np.array([4.0, 1.0, 1.0, np.nan])
    raw = np.array([0.0, 9.0, 1.0, np.nan])
    assert dual_alarm(smoothed, raw, 3.0, 5.0).tolist() == [True, True, False, False]


def test_pooled_event_f1_matches_hand_calculation():
    """1 olay yakalandı, 1 olay kaçtı, 1 yanlış blok: R=0,5, P=0,5, F1=0,5."""
    types = np.array(["spike", "none", "spike", "none", "none"])
    flags = np.array([True, False, False, False, True])
    assert pooled_event_f1([(types, flags)]) == pytest.approx(0.5)


def test_selection_prefers_fewer_alarms_on_ties_and_finds_the_useful_threshold():
    """Ham skor olayı ayırıyorsa T2 onu seçmeli; eşitlikte yüksek T2 ve büyük k."""
    types = np.array(["none", "spike", "none", "none", "none", "none"])
    copy = {
        "types": types,
        "base_smoothed": np.zeros(6),
        "base_raw": np.array([1.0, 8.0, 1.0, 2.0, 1.0, 1.0]),
        "values": np.ones(6),
        "base_valid": np.ones(6, dtype=bool),
    }
    chosen = select_settings([copy], 3.0, 0.2, np.array([1.5, 4.0, 8.0]), [1, 2])
    assert chosen[DUAL]["t2"] == 8.0
    assert chosen[DUAL]["f1"] == pytest.approx(1.0)
    assert chosen[DUAL_FLOOR]["k"] == 2


def test_experiment_has_no_path_to_test_split():
    """Aşamalar yalnız doğrulama parçalarıdır."""
    assert set(STAGES.values()) == {"val_dev", "val_hold"}


def test_hold_refuses_second_measurement(tmp_path):
    """val_hold sonucu varken ölçüm tekrar çalışmamalı."""
    (tmp_path / "a6_val_hold_per_seed.csv").write_text("x\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="bir kez"):
        run_hold({}, tmp_path)
