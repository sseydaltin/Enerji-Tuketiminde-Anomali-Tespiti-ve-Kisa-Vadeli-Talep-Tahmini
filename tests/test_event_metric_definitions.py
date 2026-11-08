"""Olay metriğinin iki tanımı için küçük, elle hesaplanabilir kontroller.

blok-precision (event_metrics): doğru blok / tüm alarm blokları.
olay-eşleşmeli (event_matched_metrics): yakalanan olay / (yakalanan olay + yanlış blok).
"""

import numpy as np
import pytest

from src.utils.anomaly_metrics import event_matched_metrics, event_metrics


def series(n, events=(), alarms=()):
    """types [n] ve predicted [n]: events = [(tip, başlangıç, bitiş)], alarms = [(başlangıç, bitiş)]."""
    types = np.array(["none"] * n, dtype=object)
    for kind, start, end in events:
        types[start:end] = kind
    predicted = np.zeros(n, dtype=bool)
    for start, end in alarms:
        predicted[start:end] = True
    return types, predicted


def test_fragmented_alarms_inside_one_event():
    """Tek olay (10 saat) içinde 3 parçalı alarm + 1 yanlış alarm."""
    types, predicted = series(40, [("level_shift", 5, 15)], [(5, 6), (8, 9), (12, 13), (30, 31)])
    block = event_metrics(types, predicted)
    matched = event_matched_metrics(types, predicted)
    assert block["event_recall"] == 1.0
    assert block["event_precision"] == pytest.approx(3 / 4)       # 3 doğru blok / 4 blok
    assert matched["matched_precision"] == pytest.approx(1 / 2)   # 1 olay / (1 olay + 1 yanlış)
    assert matched["false_alarm_blocks"] == 1


def test_merging_fragments_lowers_block_precision_only():
    """Aynı alarmlar birleştirilince blok-precision düşer, olay-eşleşmeli değişmez."""
    types, fragmented = series(40, [("level_shift", 5, 15)], [(5, 6), (8, 9), (12, 13), (30, 31)])
    _, merged = series(40, [("level_shift", 5, 15)], [(5, 13), (30, 31)])
    assert event_metrics(types, merged)["event_precision"] == pytest.approx(1 / 2)
    assert event_metrics(types, fragmented)["event_precision"] > event_metrics(types, merged)["event_precision"]
    assert (event_matched_metrics(types, merged)["matched_f1"]
            == pytest.approx(event_matched_metrics(types, fragmented)["matched_f1"]))


def test_one_alarm_covering_two_events():
    """Tek alarm bloğu iki olayı kapsar: iki tanımda da iki olay yakalanmış; blok 1 doğru, TP 2."""
    types, predicted = series(30, [("spike", 5, 6), ("drop", 8, 10)], [(4, 11)])
    block = event_metrics(types, predicted)
    matched = event_matched_metrics(types, predicted)
    assert block["event_recall"] == 1.0 and block["n_pred_blocks"] == 1
    assert block["event_precision"] == 1.0
    assert matched["detected_events"] == 2 and matched["matched_precision"] == 1.0


def test_missed_event_and_false_alarm():
    """İki olaydan biri kaçırılır, bir yanlış alarm var."""
    types, predicted = series(50, [("spike", 5, 6), ("drop", 20, 24)], [(5, 6), (40, 42)])
    block = event_metrics(types, predicted)
    matched = event_matched_metrics(types, predicted)
    assert block["event_recall"] == 0.5
    assert block["event_recall_spike"] == 1.0 and block["event_recall_drop"] == 0.0
    assert block["event_precision"] == 0.5 and matched["matched_precision"] == 0.5
    assert block["event_f1"] == pytest.approx(0.5) and matched["matched_f1"] == pytest.approx(0.5)


def test_no_alarms_and_no_events():
    types, predicted = series(20, [("spike", 3, 4)], [])
    assert event_metrics(types, predicted)["event_f1"] == 0.0
    assert event_matched_metrics(types, predicted)["matched_f1"] == 0.0
    types, predicted = series(20, [], [(3, 5)])
    assert np.isnan(event_metrics(types, predicted)["event_recall"])
    assert event_matched_metrics(types, predicted)["false_alarm_blocks"] == 1
