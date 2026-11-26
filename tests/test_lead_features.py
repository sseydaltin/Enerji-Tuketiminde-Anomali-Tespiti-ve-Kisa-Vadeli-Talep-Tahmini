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


# ----------------------------------------------------------------------------- L-FE grupları (§17)
from src.datasets.lead.features import (FE_BEHAVIOR_DROP, FE_GROUPS, fe_columns, fe_group,  # noqa: E402
                                        normalized_residual)

FE_CAUSAL_GROUPS = ("G0", "G1", "G2", "G3", "G4", "G5")


def _fe_inputs(clean):
    """G6 için bina → kod, G7 için yapay skor (ızgara sırasında)."""
    grid = full_grid(clean)
    return grid, {1: 0, 2: 3}, np.arange(len(grid), dtype=float)


def test_fe_group_columns_match_lists():
    """Her grubun ürettiği sütunlar listeyle aynı; mutlak tarih ve site_id yok; behavior'da kısayollar yok."""
    grid, use, stacked = _fe_inputs(_clean())
    for variant in ("full", "behavior"):
        for group in ("G0",) + tuple(FE_GROUPS):
            X = fe_group(grid, group, variant, primary_use=use, stacked=stacked)
            assert list(X.columns) == fe_columns([group], variant), (group, variant)
            for column in X.columns:
                assert not any(part in column.lower() for part in FORBIDDEN_NAME_PARTS), column
                assert "site" not in column.lower(), column
            if variant == "behavior":
                assert not set(X.columns) & set(FE_BEHAVIOR_DROP)


def test_fe_groups_are_causal():
    """Her grup ve varyant için: t'den sonraki okumaları (ve eksikliği) değiştirmek ≤ t özniteliklerini değiştirmez."""
    clean = _clean(n_hours=500)
    cut = pd.Timestamp("2016-01-01") + pd.Timedelta(hours=300)
    changed = clean.copy()
    later = changed["timestamp"] > cut
    changed.loc[later, "meter_reading"] *= 3.0
    changed = changed.drop(changed[later].index[::7])            # gelecekte yeni eksik saatler de
    keep = (full_grid(clean)["timestamp"] <= cut).to_numpy()
    for variant in ("full", "behavior"):
        for group in FE_CAUSAL_GROUPS:
            a = fe_group(full_grid(clean), group, variant)
            b = fe_group(full_grid(changed).reindex(full_grid(clean).index), group, variant) \
                if False else fe_group(_regrid(changed, clean), group, variant)
            pd.testing.assert_frame_equal(a[keep], b[keep], obj=f"{group}/{variant}")


def _regrid(changed, clean):
    """changed'in ızgarasını clean'in saat aralığına hizalar (son saat silinmişse ızgara kısalmasın)."""
    grid = full_grid(clean)[["building_id", "timestamp"]]
    return grid.merge(changed, on=["building_id", "timestamp"], how="left").assign(
        observed=lambda d: d["meter_reading"].notna().astype("int8"))


def test_g5_missing_context_values():
    """Saat 100 eksik (bina 1): 101'de son boşluktan beri 1 saat, boşluk uzunluğu 1, son 24 saatte 1 eksik."""
    grid = full_grid(_clean())
    X = fe_group(grid, "G5", "full")
    b1 = (grid["building_id"] == 1).to_numpy()
    hour = ((grid["timestamp"] - pd.Timestamp("2016-01-01")) / pd.Timedelta(hours=1)).to_numpy()
    row = X[b1 & (hour == 101)].iloc[0]
    assert row["gap_age_h"] == 1 and row["last_gap_len"] == 1 and row["missing_24h"] == 1
    assert np.isnan(X[b1 & (hour == 50)].iloc[0]["gap_age_h"])     # öncesinde boşluk yok


def test_g1_unchanged_and_profile_flag():
    """Sabit seri (200–202) değişmeden geçen süreyi 1, 2, 3 yapar; profil ilk haftalarda kısa bayraklı."""
    grid = full_grid(_clean())
    X = fe_group(grid, "G1", "full")
    b1 = (grid["building_id"] == 1).to_numpy()
    hour = ((grid["timestamp"] - pd.Timestamp("2016-01-01")) / pd.Timedelta(hours=1)).to_numpy()
    assert [X.loc[b1 & (hour == h), "unchanged_h"].item() for h in (200, 201, 202)] == [1, 2, 3]
    P = fe_group(grid, "G4", "full")
    assert P.loc[b1 & (hour == 10), "profile_short"].item() == 1
    assert np.isnan(P.loc[b1 & (hour == 10), "profile_z"].item())


def test_normalized_residual_is_causal():
    """G7 ölçeği yalnız geçmiş hatalardan: t'den sonraki hataları değiştirmek ≤ t skorunu değiştirmez."""
    rng = np.random.default_rng(0)
    errors = rng.random((400, 3))
    changed = errors.copy()
    changed[251:] *= 10
    a, b = normalized_residual(errors, 168), normalized_residual(changed, 168)
    pd.testing.assert_frame_equal(a.iloc[:251], b.iloc[:251])


def test_offline_features_isolated_and_two_sided():
    """Offline (iki yönlü) sütunlar hiçbir nedensel sette/grupta yok; ve gerçekten geleceğe bakıyorlar
    (geleceği değiştirmek onları değiştirir — yanlışlıkla nedensel sete girerse bu test fark ettirir)."""
    from src.datasets.lead.features import OFFLINE_COLS, offline_features
    causal = set(FULL) | set(BEHAVIOR) | {c for cols in FE_GROUPS.values() for c in cols}
    assert not causal & set(OFFLINE_COLS)
    clean = _clean(n_hours=500)
    cut = pd.Timestamp("2016-01-01") + pd.Timedelta(hours=300)
    changed = clean.copy()
    changed.loc[changed["timestamp"] > cut, "meter_reading"] *= 3.0
    keep = (full_grid(clean)["timestamp"] <= cut).to_numpy()
    a, b = offline_features(full_grid(clean)), offline_features(full_grid(changed))
    assert not a[keep].equals(b[keep])


def test_peer_deviation_excludes_self_and_falls_back():
    """P4: bina kendi değerini görmez; aynı türde < 3 akran varsa parçanın tümü + bayrak; parça dışı kullanılmaz."""
    from src.datasets.lead.features import peer_deviation
    H = 5
    ratio = np.tile(np.array([0.0, 1.0, 2.0, 3.0, 100.0, 50.0]), (H, 1))   # [H, 6]
    part = np.array(["train"] * 5 + ["val"])
    use = np.array(["ofis", "ofis", "ofis", "ofis", "ofis", "ofis"])
    out = peer_deviation(ratio, part, use)
    assert out["peer_median"][0, 4] == 1.5                                # b=4 hariç train akranları 0,1,2,3
    assert out["peer_fallback"][0, 0] == 0
    assert np.isnan(out["peer_median"][0, 5])                             # val'de başka bina yok
    use2 = np.array(["ofis", "ofis", "okul", "okul", "okul", "ofis"])
    out2 = peer_deviation(ratio, part, use2)
    assert out2["peer_fallback"][0, 0] == 1                               # train'de yalnız 1 ofis akranı
    assert out2["peer_median"][0, 0] == 2.5                               # → train'in tümü (b hariç): 1,2,3,100


def test_peer_features_are_causal():
    """P4 t'den sonraki okumaları değiştirince ≤ t değişmez."""
    from src.datasets.lead.features import peer_features
    clean = _clean(n_hours=500, buildings=(1, 2, 3, 4))
    cut = pd.Timestamp("2016-01-01") + pd.Timedelta(hours=300)
    changed = clean.copy()
    changed.loc[changed["timestamp"] > cut, "meter_reading"] *= 3.0
    part_of = {1: "train", 2: "train", 3: "train", 4: "val"}
    use_of = {1: "a", 2: "a", 3: "b", 4: "a"}
    keep = (full_grid(clean)["timestamp"] <= cut).to_numpy()
    a = peer_features(full_grid(clean), part_of, use_of)
    b = peer_features(full_grid(changed), part_of, use_of)
    pd.testing.assert_frame_equal(a[keep], b[keep])
