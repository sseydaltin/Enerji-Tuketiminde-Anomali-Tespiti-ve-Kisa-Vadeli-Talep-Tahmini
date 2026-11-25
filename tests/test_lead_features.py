"""LEAD öznitelikleri, bölmeler ve değerlendirme etiketleri (protokol §3–§7, §15)."""

import numpy as np
import pandas as pd

from src.datasets.lead.evaluate import evaluate, f1_max_threshold
from src.datasets.lead.features import (BEHAVIOR, CALENDAR, FEATURE_SETS, FORBIDDEN_NAME_PARTS, FULL,
                                        SHORTCUT_COLUMNS, evaluation_labels, full_grid, tabular_features)
from src.datasets.lead.splits import building_split, site_split


def _clean(n_hours=400, buildings=(1, 2), missing=((1, 100),), seed=0):
    """Temiz tablo biçiminde küçük sentetik veri; bazı saatler eksik, bazıları anomali."""
    rng = np.random.default_rng(seed)
    rows = []
    for b in buildings:
        for h in range(n_hours):
            if (b, h) in missing:
                continue
            value = 10 + 5 * np.sin(2 * np.pi * h / 24) + rng.normal()
            label = 1 if h in (50, 51, 101, 200, 201, 202) else 0
            if h in (200, 201, 202):
                value = 3.0                                      # sabit seri (anomali)
            rows.append((b, pd.Timestamp("2016-01-01") + pd.Timedelta(hours=h), value, label))
    return pd.DataFrame(rows, columns=["building_id", "timestamp", "meter_reading", "anomaly"])


def test_no_absolute_date_features():
    """Hiçbir öznitelik setinde mutlak tarih türevi yok; takvim yalnız döngüsel (güncelleme 1)."""
    for name, columns in FEATURE_SETS.items():
        for column in columns:
            assert not any(part in column.lower() for part in FORBIDDEN_NAME_PARTS), (name, column)
    calendar_like = [c for c in FULL if any(k in c for k in ("hour", "dow", "month", "week", "day"))]
    assert set(calendar_like) == set(CALENDAR)
    X = tabular_features(full_grid(_clean()), "full")
    assert list(X.columns) == FULL                               # üretilen tablo da listeyle aynı


def test_behavior_set_excludes_shortcuts():
    """Behavior setinde eksiklik ve sabitlik sütunları yok; NaN üzerinden eksiklik bilgisi de yok."""
    assert not set(SHORTCUT_COLUMNS) & set(BEHAVIOR)
    X = tabular_features(full_grid(_clean()), "behavior")
    grid = full_grid(_clean())
    observed = (grid["observed"] == 1).to_numpy()
    lag_cols = [c for c in X.columns if c.startswith("lag_")]
    # ilk 168 saat dışında gözlenen saatlerde gecikmeler NaN değil (ileri doldurulmuş seri)
    late = observed & (grid.groupby("building_id").cumcount().to_numpy() >= 168)
    assert not X.loc[late, lag_cols].isna().any().any()


def test_features_are_causal():
    """t'den SONRAKİ okumaları değiştirmek t ve öncesindeki öznitelikleri değiştirmez."""
    clean = _clean()
    cut = pd.Timestamp("2016-01-01") + pd.Timedelta(hours=250)
    changed = clean.copy()
    changed.loc[changed["timestamp"] > cut, "meter_reading"] *= 7.0
    for feature_set in FEATURE_SETS:
        a = tabular_features(full_grid(clean), feature_set)
        b = tabular_features(full_grid(changed), feature_set)
        keep = (full_grid(clean)["timestamp"] <= cut).to_numpy()
        pd.testing.assert_frame_equal(a[keep], b[keep])


def test_building_split_disjoint_and_reproducible():
    """Parçalar kesişmiyor, toplam korunuyor, aynı seed aynı bölmeyi veriyor."""
    buildings = np.arange(200)
    split = building_split(buildings, 42)
    parts = [set(split[p]) for p in ("train", "val", "test")]
    assert [len(p) for p in parts] == [120, 40, 40]
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
    assert split == building_split(buildings, 42) and split != building_split(buildings, 7)


def test_site_split_keeps_sites_whole():
    """Bir sitenin bütün binaları tek parçada."""
    sites = pd.DataFrame({"building_id": np.arange(30), "site_id": np.repeat(np.arange(6), 5)})
    split = site_split(sites, 7)
    part_of = {b: p for p in split for b in split[p]}
    for _, group in sites.groupby("site_id"):
        assert len({part_of[b] for b in group["building_id"]}) == 1
    assert sorted(part_of) == list(range(30))


def test_evaluation_labels_and_metrics():
    """Her anomali saati tek alt türde; öncelik gap_edge > frozen > other; metrik tablosu tüm dilimleri içeriyor."""
    clean = _clean().sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    labels = evaluation_labels(clean)
    hours = (clean["timestamp"] - pd.Timestamp("2016-01-01")) / pd.Timedelta(hours=1)
    b1 = clean["building_id"] == 1
    assert labels.loc[b1 & (hours == 101), "subtype"].item() == "gap_edge"   # saat 100 eksik
    assert labels.loc[b1 & (hours == 200), "subtype"].item() == "frozen"
    assert labels.loc[b1 & (hours == 50), "subtype"].item() == "other"
    assert (labels.loc[clean["anomaly"] == 0, "subtype"] == "normal").all()
    assert labels.loc[clean["anomaly"] == 1, "shared"].notna().all()        # her iki binada aynı saatler
    y = clean["anomaly"].to_numpy()
    table = evaluate(y, y.astype(float), f1_max_threshold(y, y.astype(float)), labels)
    assert set(table["dilim"]) == {"genel", "gap_edge", "frozen", "other", "paylasilan", "tekil"}
    assert np.isclose(table.set_index("dilim").loc["genel", "f1"], 1.0)
