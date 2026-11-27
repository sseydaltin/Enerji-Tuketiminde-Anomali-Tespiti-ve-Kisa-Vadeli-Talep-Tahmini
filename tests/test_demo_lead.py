"""LEAD demo sayfası: kayıtlı skorların hizası, eşiklerin val kaydıyla aynılığı ve tablo değerleri.

Veri ve skorlar git'e girmez; yerelde yoksa testler atlanır. Eğitim veya skor hesabı yapılmaz.
"""

import numpy as np
import pandas as pd
import pytest

from app.lead_services import (LEAD_MODELS, building_subtype_table, lead_paths, load_test_frame, load_thresholds,
                               missing_files, slice_results)
from app.pages.lead import make_figure, results_table

SEED = 42


@pytest.fixture(scope="module")
def lead_demo():
    if missing_files(SEED):
        pytest.skip("LEAD verisi veya kayıtlı test skorları yerelde yok")
    return load_test_frame(SEED), load_thresholds(SEED)


def test_scores_align_with_test_rows_and_thresholds_come_from_val(lead_demo):
    """Skor sütunları test satırı sayısında; eşikler val_results.csv'deki eşiklerle aynı (testte seçim yok)."""
    frame, thresholds = lead_demo
    assert frame["building_id"].nunique() == 40
    _, _, tables, _ = lead_paths()
    val = pd.read_csv(tables / "val_results.csv")
    for (model, fset), threshold in thresholds.items():
        assert f"{model}|{fset}" in frame
        saved = val[(val.model == model) & (val.set == fset) & (val.bolme == "bina") & (val.seed == SEED)
                    & (val.dilim == "genel")]["threshold"].iloc[0]
        assert np.isclose(threshold, saved)


def test_slice_table_matches_saved_test_results():
    """Alt tür tablosu test_results.csv'nin 3 seed ortalamasıyla aynı."""
    if missing_files(SEED):
        pytest.skip("LEAD dosyaları yok")
    table = slice_results()
    _, _, tables, _ = lead_paths()
    saved = pd.read_csv(tables / "test_results.csv")
    x = saved[(saved.model == "L1") & (saved.set == "behavior") & (saved.dilim == "other") & (saved.bolme == "bina")]
    row = table[(table.model == LEAD_MODELS[("L1", "behavior")]) & (table.dilim == "Davranışsal (other)")].iloc[0]
    assert np.isclose(row["pr_auc_mean"], x["pr_auc"].mean()) and row["n_seed"] == 3
    assert len(results_table(table)) == len(LEAD_MODELS) * 6


def test_building_table_and_figure(lead_demo):
    """Bina tablosu alt tür sayımlarını doğru topluyor; figür 1 + model sayısı satırlı."""
    frame, thresholds = lead_demo
    building = int(frame.groupby("building_id")["anomaly"].sum().idxmax())
    table = building_subtype_table(frame, building, thresholds)
    part = frame[frame.building_id == building]
    assert table["Anomali saati"].iloc[:3].sum() == int((part["anomaly"] == 1).sum())
    figure = make_figure(frame, building, thresholds)
    assert len(figure.data) == 1 + 3 + len(LEAD_MODELS)
