"""LEAD seçim kuralı §18.2: other PR-AUC → genel PR-AUC → sinir ağı tercihi → basitlik."""

import pandas as pd

from src.pipelines.lead_models import select_model_v2


def _results(rows):
    """rows: {(model, set): ([other ×3], [genel ×3])} → val_results biçiminde tablo."""
    out = []
    for (model, fset), (other, genel) in rows.items():
        for seed, o, g in zip((42, 7, 2026), other, genel):
            out += [{"model": model, "set": fset, "bolme": "bina", "seed": seed, "dilim": "other", "pr_auc": o},
                    {"model": model, "set": fset, "bolme": "bina", "seed": seed, "dilim": "genel", "pr_auc": g}]
    return pd.DataFrame(out).assign(extra="{}")


def _chosen(rows):
    table = select_model_v2(_results(rows))
    row = table[table["secilen"]].iloc[0]
    return row["model"], row["set"]


def test_other_decides_when_significant():
    """other'da anlamlı biçimde en iyi model, genel'i kötü olsa da seçilir."""
    rows = {("L1", "full"): ([0.02, 0.021, 0.019], [0.70, 0.71, 0.69]),
            ("L4", "-"): ([0.10, 0.11, 0.09], [0.20, 0.21, 0.19])}
    assert _chosen(rows) == ("L4", "-")


def test_neural_preferred_only_within_noise():
    """İki metrikte de gürültü düzeyinde → sinir ağı; genel'de anlamlı kötüyse sinir ağı seçilmez."""
    tied = {("L1-FE", "full"): ([0.050, 0.060, 0.040], [0.60, 0.70, 0.50]),
            ("L2-FE", "full"): ([0.052, 0.045, 0.050], [0.62, 0.66, 0.52])}
    assert _chosen(tied) == ("L2-FE", "full")
    worse = {("L1-FE", "full"): ([0.050, 0.060, 0.040], [0.60, 0.70, 0.50]),
             ("L2-FE", "full"): ([0.052, 0.045, 0.050], [0.30, 0.40, 0.20])}
    assert _chosen(worse) == ("L1-FE", "full")


def test_simplicity_breaks_remaining_ties():
    """Sinir ağı yoksa basitlik: L0 < L1; aile içinde FE'siz < FE."""
    rows = {("L1-FE", "full"): ([0.050, 0.060, 0.040], [0.60, 0.70, 0.50]),
            ("L1", "full"): ([0.049, 0.058, 0.043], [0.61, 0.68, 0.52])}
    assert _chosen(rows) == ("L1", "full")


def test_ensemble_without_neural_weight_gets_no_preference():
    """§18.2 netleştirme: bir seed'de w_L1 = 1 olan L5 topluluğu sinir ağı sayılmaz → basitlik L1'i seçer."""
    rows = {("L1-FE", "full"): ([0.050, 0.060, 0.040], [0.60, 0.70, 0.50]),
            ("L5-FE", "full"): ([0.051, 0.061, 0.040], [0.61, 0.70, 0.50])}
    table = _results(rows)
    table["extra"] = "{}"
    weights = {42: 1.0, 7: 0.7, 2026: 1.0}
    mask = table["model"] == "L5-FE"
    table.loc[mask, "extra"] = table.loc[mask, "seed"].map(lambda s: '{"w_L1": %s}' % weights[s])
    chosen = select_model_v2(table)
    assert chosen[chosen["secilen"]].iloc[0]["model"] == "L1-FE"
    table.loc[mask, "extra"] = '{"w_L1": 0.8}'                     # her seed'de GRU katkısı var → L5 tercih
    chosen = select_model_v2(table)
    assert chosen[chosen["secilen"]].iloc[0]["model"] == "L5-FE"
