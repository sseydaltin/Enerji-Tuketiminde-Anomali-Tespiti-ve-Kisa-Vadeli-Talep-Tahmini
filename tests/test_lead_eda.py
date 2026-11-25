"""LEAD EDA hesapları: olaylar eksik saatte bölünür, sabit seriler doğru sayılır, kapsama doğru."""

import pandas as pd

from src.datasets.lead.eda import anomaly_events, coverage, gap_neighbors, length_bins, reading_flags


def _frame(rows):
    """[(bina, saat_ofseti, okuma, etiket)] → temiz tablo biçimi."""
    base = pd.Timestamp("2016-01-01")
    df = pd.DataFrame(rows, columns=["building_id", "h", "meter_reading", "anomaly"])
    df["timestamp"] = base + pd.to_timedelta(df["h"], unit="h")
    return df.drop(columns="h").sort_values(["building_id", "timestamp"]).reset_index(drop=True)


def test_events_split_on_gap_and_building():
    """Ardışık etiketli saatler tek olay; eksik saat ve bina değişimi olayı böler."""
    df = _frame([(1, 0, 1, 1), (1, 1, 1, 1), (1, 3, 1, 1),        # 0–1 olay, 2 eksik, 3 yeni olay
                 (1, 4, 1, 0), (2, 5, 1, 1), (2, 6, 1, 1)])         # bina 2: yeni olay
    events = anomaly_events(df)
    assert list(events["length_hours"]) == [2, 1, 2]
    assert list(events["building_id"]) == [1, 1, 2]


def test_frozen_runs():
    """Aynı değer ≥3 ardışık saat sabit sayılır; boşluk ve bina değişimi seriyi böler; sıfır ayrıca işaretlenir."""
    df = _frame([(1, 0, 5, 0), (1, 1, 5, 0), (1, 2, 5, 0),        # 3'lü seri → sabit
                 (1, 3, 0, 1), (1, 4, 0, 1),                       # 2'li sıfır serisi → sabit değil
                 (1, 6, 0, 1),                                     # boşluk sonrası tek sıfır
                 (2, 7, 5, 0)])                                    # başka bina, aynı değer
    flags = reading_flags(df, frozen_min_run=3)
    assert list(flags["is_frozen"]) == [True, True, True, False, False, False, False]
    assert list(flags["run_length"]) == [3, 3, 3, 2, 2, 1, 1]
    assert list(flags["is_zero"]) == [False, False, False, True, True, True, False]


def test_coverage_counts_missing_hours():
    """Eksik saat = genel ilk–son saat ızgarası − gözlem."""
    df = _frame([(1, 0, 1, 0), (1, 1, 1, 0), (1, 3, 1, 0), (2, 0, 1, 0)])
    cov = coverage(df).set_index("building_id")
    assert cov.loc[1, "expected_hours"] == 4
    assert cov.loc[1, "missing_hours"] == 1 and cov.loc[2, "missing_hours"] == 3


def test_length_bins():
    """Uzunluk kutuları sınırları dahil eder."""
    bins = length_bins(pd.Series([1, 2, 3, 5, 6, 24, 25, 169]))
    assert list(bins.astype(str)) == ["1", "2", "3–5", "3–5", "6–12", "13–24", "25–72", ">168"]


def test_gap_neighbors():
    """Eksik saatin iki yanındaki satırlar işaretlenir; ızgaranın genel başı/sonu boşluk sayılmaz."""
    df = _frame([(1, 0, 1, 0), (1, 1, 1, 0), (1, 3, 1, 0), (1, 4, 1, 0),   # saat 2 eksik
                 (2, 1, 1, 0), (2, 2, 1, 0)])                              # bina 2: saat 0, 3, 4 eksik
    assert list(gap_neighbors(df)) == [False, True, True, False, True, True]
