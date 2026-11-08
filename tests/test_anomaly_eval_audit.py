"""Değerlendirme denetimi: yanlış blokların arka plan (enjeksiyonsuz) alarmlarıyla ayrıştırılması."""

import numpy as np

from src.anomaly_eval_audit import alarm_mask, copy_row


def test_false_blocks_split_into_background_and_new():
    """Yanlış bloklardan enjeksiyonsuz kopyada da alarm verenler 'arka plan' sayılır; metrik değişmez."""
    types = np.array(["none"] * 30, dtype=object)
    types[5:7] = "spike"
    flagged = np.zeros(30, dtype=bool)
    flagged[[5, 15, 25]] = True                     # 1 doğru blok + 2 yanlış blok
    background = np.zeros(30, dtype=bool)
    background[15] = True                           # 15 enjeksiyonsuz kopyada da alarm
    row = copy_row(types, flagged, background)
    assert row["false_alarm_blocks"] == 2
    assert row["false_blocks_in_background"] == 1 and row["false_blocks_new"] == 1
    assert row["block_precision"] == row["count_precision"] == 1 / 3


def test_alarm_mask_never_flags_invalid_hours():
    score = np.array([5.0, np.nan, 5.0, 0.0, 5.0])
    valid = np.array([True, False, True, True, False])
    assert alarm_mask(score, valid, 1.0, 2).tolist() == [True, False, True, False, False]
