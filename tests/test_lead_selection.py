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
    return pd.DataFrame(out)


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
