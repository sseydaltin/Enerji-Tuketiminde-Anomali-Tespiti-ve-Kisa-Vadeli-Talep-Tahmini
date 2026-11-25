"""LEAD bölmeleri: binaya göre (ana) ve siteye göre gruplu (sağlamlık kontrolü). Protokol §5.

Bütün bölmeler bina düzeyindedir: bir binanın bütün saatleri tek parçadadır.
"""

import numpy as np
import pandas as pd

PARTS = ("train", "val", "test")


def building_split(buildings, seed, n_val=40, n_test=40):
    """Binaları seed'le karıştırıp train / val / test'e ayırır (200 binada 120 / 40 / 40).

    Girdi: bina kimlikleri, seed. Çıktı: {"train": [...], "val": [...], "test": [...]} (sıralı listeler).
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.sort(np.asarray(buildings)))
    test, val, train = order[:n_test], order[n_test:n_test + n_val], order[n_test + n_val:]
    return {"train": sorted(train.tolist()), "val": sorted(val.tolist()), "test": sorted(test.tolist())}


def site_split(sites, seed, fractions=(0.6, 0.2, 0.2)):
    """Siteleri bütün olarak parçalara atar; bina sayıları yaklaşık %60 / %20 / %20.

    Girdi: DataFrame (building_id, site_id), seed. Çıktı: building_split ile aynı biçim.
    Yöntem: siteler seed'le karıştırılır; sırayla önce test, sonra val, bina payı hedefe ulaşana
    kadar doldurulur; kalan siteler train'dir. Siteler dengesiz olduğu için gerçekleşen paylar değişir.
    """
    rng = np.random.default_rng(seed)
    counts = sites.groupby("site_id")["building_id"].nunique()
    order = rng.permutation(np.sort(counts.index.to_numpy()))
    total, targets = counts.sum(), {"test": fractions[2], "val": fractions[1]}
    assigned, part_sites = {"train": [], "val": [], "test": []}, {"train": [], "val": [], "test": []}
    remaining = list(order)
    for part in ("test", "val"):
        while remaining and sum(counts[s] for s in part_sites[part]) < targets[part] * total:
            part_sites[part].append(remaining.pop(0))
    part_sites["train"] = remaining
    for part, site_list in part_sites.items():
        assigned[part] = sorted(sites.loc[sites["site_id"].isin(site_list), "building_id"].tolist())
    return assigned


def split_table(split, kind, seed):
    """Bölmeyi uzun tablo olarak verir: (bölme, seed, parça, building_id)."""
    return pd.DataFrame([{"bolme": kind, "seed": seed, "parca": part, "building_id": b}
                         for part in PARTS for b in split[part]])
