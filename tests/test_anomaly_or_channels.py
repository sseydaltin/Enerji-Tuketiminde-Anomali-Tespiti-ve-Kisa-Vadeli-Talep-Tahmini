"""OR kanal birleşimi: r = max(a/T1, c/T2) ≥ 1 ⇔ (a ≥ T1) VEYA (c ≥ T2); NaN kanal diğerini engellemez."""

import numpy as np

from src.anomaly_or_channels import added_false_alarms, attribution, or_score


def test_or_score_equals_or_of_channel_decisions():
    a = np.array([2.0, 0.5, np.nan, 0.5, 1.0])
    c = np.array([1.0, 9.0, 9.0, np.nan, 1.0])
    imputed = np.array([False, False, False, False, True])
    score, valid = or_score(a, c, imputed, 1.0, 8.0)
    with np.errstate(invalid="ignore"):
        expected = ((a >= 1.0) | (c >= 8.0)) & ~imputed
    assert ((score >= 1.0) & valid).tolist() == expected.tolist() == [True, True, True, False, False]


def test_attribution_and_added_false_alarms():
    types = np.array(["none", "spike", "none", "pattern", "pattern", "none", "none"], dtype=object)
    flag_a = np.array([0, 1, 0, 0, 0, 0, 0], dtype=bool)
    flag_c = np.array([0, 1, 0, 1, 0, 0, 1], dtype=bool)
    caught = {r["type"]: r["caught_by"] for r in attribution("k", 1, types, flag_a, flag_c)}
    assert caught == {"spike": "ikisi", "pattern": "yalnız C1"}
    added = added_false_alarms("k", 1, types, flag_a | flag_c, flag_a, np.zeros(7, dtype=bool))
    assert added["or_false_blocks"] == 1 and added["added_vs_s0"] == 1 and added["added_in_background"] == 0
