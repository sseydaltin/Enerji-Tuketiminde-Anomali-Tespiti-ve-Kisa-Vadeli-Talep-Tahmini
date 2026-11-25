"""Taban kanalı: yalnız t ve t−1; OR birleşimi; enjeksiyon değeri kullanılmaz."""

import numpy as np

from src.anomaly_floor_channel import combined, floor_flags


def test_floor_needs_two_consecutive_low_hours_and_only_past():
    values = np.array([1.0, 0.1, 1.0, 0.1, 0.1, 0.1, 1.0])
    assert floor_flags(values, 0.15).tolist() == [False, False, False, False, True, True, False]
    changed = values.copy()
    changed[5:] = 5.0                                  # gelecek değişse de t=4 kararı aynı
    assert floor_flags(changed, 0.15)[4]


def test_combined_alarm_is_or_of_channels():
    a = np.array([2.0, 0.1, np.nan, 0.1])
    flags = np.array([False, True, True, False])
    imputed = np.array([False, False, False, True])
    score, valid = combined(a, flags, imputed)
    assert ((score >= 1.0) & valid).tolist() == [True, True, True, False]


def test_experimental_config_matches_script_constants():
    """configs/experimental_s0_floor.yaml, betiğin kullandığı değerlerle aynı olmalı (deneysel etiketiyle)."""
    from pathlib import Path

    import yaml

    from src.anomaly_candidates import S0_SCALED_THRESHOLD
    from src.anomaly_floor_channel import FLOOR_QUANTILE, SENSITIVITY_QUANTILE

    path = Path(__file__).resolve().parents[1] / "configs/experimental_s0_floor.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))["experimental_s0_floor"]
    assert config["status"].startswith("deneysel")
    assert config["short_channel"]["threshold_scaled"] == S0_SCALED_THRESHOLD
    assert config["floor_channel"]["floor_quantile"] == FLOOR_QUANTILE
    assert config["floor_channel"]["consecutive_hours"] == 2
    assert config["sensitivity_drop_quantile"] == SENSITIVITY_QUANTILE
