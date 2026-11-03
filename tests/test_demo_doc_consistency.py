"""Demodaki sayılar docs/hane_vs_sebeke.md ve docs/final_sonuclar.md ile tutarlı mı?

Beklenen değerler koda yazılmaz, belgelerdeki tablolardan okunur. Veri ve ağırlık
gerektiren testler bu dosyalar yoksa atlanır (veri ve ağırlıklar repoya girmez).
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.eld_services import HARD_DAY, eld_paths, eld_test_predictions, predict_eld_day
from app.pages.forecast import overall_result_text
from app.pages.results import _anomaly_table, _cross_dataset_table, _forecast_table
from app.results_services import load_results_bundle
from app.services import (
    best_decile_days,
    comparison_table,
    median_day,
    valid_test_days,
    worst_decile_days,
)
from app.ui import FORECAST_LABELS, METHOD_LABELS
from src.data.load import PROJECT_ROOT
from src.utils.metrics import calculate_metrics

COMPARISON_DOC = PROJECT_ROOT / "docs/hane_vs_sebeke.md"
FINAL_DOC = PROJECT_ROOT / "docs/final_sonuclar.md"
SEEDS = ("", "-s7", "-s2026")
# hane_vs_sebeke.md §1 tablosunun sütunları
FORECAST_COLUMNS = ("hane_mape", "sebeke_mape", "hane_r2", "sebeke_r2", "hane_gain", "sebeke_gain")


def _doc_rows(path, label):
    """Bir Markdown belgesinde ilk sütunu `label` içeren satırların hücrelerini döndürür."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if line.startswith("|") and label in cells[0]:
            rows.append(cells[1:])
    assert rows, f"{path} içinde {label} satırı yok"
    return rows


def _number(cell):
    """'**3,24 %**', '+35,4 %', '−0,244' veya '0,384 ± 0,170' hücresinden (değer, ondalık) okur."""
    text = cell.replace("**", "").replace("−", "-").replace("+", "").replace("%", "").split("±")[0].strip()
    match = re.fullmatch(r"(-?\d+)(?:,(\d+))?", text)
    assert match, f"Sayı okunamadı: {cell!r}"
    decimals = len(match.group(2) or "")
    return float(text.replace(",", ".")), decimals


def _assert_matches(value, cell):
    """Hesaplanan değer, belgedeki değerin yuvarlama payı içinde olmalı."""
    expected, decimals = _number(cell)
    assert abs(value - expected) <= 0.5 * 10 ** (-decimals) + 1e-9, f"{value} ≠ belge {cell}"


def _table_number(text):
    """Arayüz hücresindeki '0,406 ± 0,005' metninin ortalamasını sayıya çevirir."""
    return float(text.split("±")[0].strip().replace(",", "."))


@pytest.fixture(scope="module")
def eld_live():
    """Şebeke tahminlerini demo servisiyle, 3 seed için üretir; sonuç dosyasına yazmaz."""
    try:
        config, data_path = eld_paths()
    except FileNotFoundError as error:
        pytest.skip(f"Şebeke verisi veya ağırlıkları yok: {error}")
    frame = pd.read_parquet(data_path)
    return frame, config, eld_test_predictions(frame, config, SEEDS)


def _seed_mean(results, config, base_id):
    """Bir modelin 3 seed üzerinden ortalama MAE, MAPE* ve R² değerleri."""
    metrics = [
        calculate_metrics(results[f"{base_id}{seed}"][0], results[f"{base_id}{seed}"][1],
                          mape_min_y=config["evaluation"]["mape_min_y"])
        for seed in SEEDS
    ]
    return {key: float(np.mean([metric[key] for metric in metrics])) for key in ("MAE", "MAPE*", "R2")}


@pytest.mark.parametrize("doc_label,base_id", [
    ("F13-ens (final)", "ELD-F13-ens"),
    ("F10-opt Seq2Seq", "ELD-F10-opt"),
    ("F5-opt LSTM", "ELD-F5-opt"),
])
def test_live_eld_forecast_matches_comparison_doc(eld_live, doc_label, base_id):
    """Canlı Şebeke tahmini 3 seed'de belgedeki MAPE, R² ve SN168 kazancını üretmeli."""
    frame, config, results = eld_live
    row = dict(zip(FORECAST_COLUMNS, _doc_rows(COMPARISON_DOC, doc_label)[0]))
    model = _seed_mean(results, config, base_id)
    baseline = calculate_metrics(results["ELD-F2"][0], results["ELD-F2"][1],
                                 mape_min_y=config["evaluation"]["mape_min_y"])
    _assert_matches(model["MAPE*"], row["sebeke_mape"])
    _assert_matches(model["R2"], row["sebeke_r2"])
    _assert_matches(100.0 * (baseline["MAE"] - model["MAE"]) / baseline["MAE"], row["sebeke_gain"])


def test_live_eld_baseline_matches_comparison_doc(eld_live):
    """Canlı SN168 belgedeki MAPE ve R² değerlerini üretmeli."""
    frame, config, results = eld_live
    row = dict(zip(FORECAST_COLUMNS, _doc_rows(COMPARISON_DOC, "F2 Seasonal Naive 168")[0]))
    baseline = calculate_metrics(results["ELD-F2"][0], results["ELD-F2"][1],
                                 mape_min_y=config["evaluation"]["mape_min_y"])
    _assert_matches(baseline["MAPE*"], row["sebeke_mape"])
    _assert_matches(baseline["R2"], row["sebeke_r2"])


def test_eld_hard_day_is_christmas_as_documented(eld_live):
    """25 Aralık geçerli olmalı, en kötü %10'da ve belgedeki yük düzeyinde (~47 MW) kalmalı."""
    frame, config, results = eld_live
    days = valid_test_days(frame, config)
    assert HARD_DAY in days
    table = comparison_table(days, lambda day: predict_eld_day(frame, config, results, day),
                             config["evaluation"]["mape_min_y"])
    assert HARD_DAY in worst_decile_days(table)
    assert table.loc[table["model_mae"].idxmax(), "day"] == HARD_DAY
    day = predict_eld_day(frame, config, results, HARD_DAY)
    assert day["actual_kw"].mean() == pytest.approx(47_000, abs=500)
    target = config["preprocess"]["target"]
    neighbours = [frame.loc[other, target].mean() for other in ("2014-12-24", "2014-12-26")]
    assert 90_000 < min(neighbours) and max(neighbours) < 100_000


def test_day_selection_rules():
    """Varsayılan gün medyana en yakın, kolay/zor dilimleri %10'luk olmalı."""
    table = pd.DataFrame({
        "day": [f"d{i}" for i in range(20)],
        "model_mae": np.arange(1.0, 21.0),
        "baseline_mae": np.full(20, 10.0),
        "gain_pct": np.arange(20.0),
    })
    assert median_day(table) in ("d9", "d10")
    assert best_decile_days(table) == ["d0", "d1"]
    assert worst_decile_days(table) == ["d18", "d19"]


@pytest.mark.parametrize("dataset,prefix,digits", [("household", "hane", 3), ("eld", "sebeke", 1)])
def test_results_tables_match_comparison_doc(dataset, prefix, digits):
    """Sonuçlar sayfası tablosu ve test geneli cümlesi belgedeki değerleri göstermeli."""
    bundle = load_results_bundle(dataset)
    table = _cross_dataset_table(bundle["cross_dataset"])
    table = table[table["Veri seti"].str.startswith("Hane" if dataset == "household" else "Şebeke")]
    table = table.set_index("Model")
    for doc_label, key in (("F13-ens (final)", "F13-ens"), ("F12-L336 XGBoost", "XGBoost"),
                           ("F2 Seasonal Naive 168", "SN168")):
        row = dict(zip(FORECAST_COLUMNS, _doc_rows(COMPARISON_DOC, doc_label)[0]))
        shown = table.loc[FORECAST_LABELS[key]]
        _assert_matches(_table_number(shown["MAPE (%)"]), row[f"{prefix}_mape"])
        _assert_matches(_table_number(shown["R²"]), row[f"{prefix}_r2"])
        if key != "SN168":
            _assert_matches(_table_number(shown["SN168'e göre MAE kazancı (%)"]), row[f"{prefix}_gain"])
    final_row = dict(zip(FORECAST_COLUMNS, _doc_rows(COMPARISON_DOC, "F13-ens (final)")[0]))
    gain_text = overall_result_text(bundle["forecast"], digits).split("%")[1].split(" ")[0]
    _assert_matches(float(gain_text.replace(",", ".")), final_row[f"{prefix}_gain"])


def test_household_forecast_matches_final_doc():
    """Hane MAE/RMSE tablosu ve test geneli cümlesi final belgeyle aynı olmalı."""
    bundle = load_results_bundle("household")
    forecast = bundle["forecast"].set_index("model_key")
    table = _forecast_table(bundle["forecast"]).set_index("Model")
    for doc_label, key in (("F13-ens (final)", "F13-ens"), ("F2 Seasonal Naive 168", "SN168")):
        cells = _doc_rows(FINAL_DOC, doc_label)[0]            # Seed, val MAE, test MAE, val RMSE, test RMSE ...
        _assert_matches(forecast.loc[key, "MAE"], cells[2])
        _assert_matches(forecast.loc[key, "RMSE"], cells[4])
        # Arayüz 3 ondalık gösterir; gösterilen metin aynı değerin yuvarlanmışı olmalı.
        assert _table_number(table.loc[FORECAST_LABELS[key], "MAE"]) == round(forecast.loc[key, "MAE"], 3)
    assert "MAE 0,406 kW" in overall_result_text(bundle["forecast"], 3)


@pytest.mark.parametrize("dataset,column", [("household", 0), ("eld", 1)])
def test_results_anomaly_table_matches_comparison_doc(dataset, column):
    """Sonuçlar sayfasındaki olay F1 ve PR-AUC belgedeki §2 tablosuyla aynı olmalı."""
    table = _anomaly_table(load_results_bundle(dataset)["anomaly"]).set_index("Yöntem")
    for doc_label, method in (("A0 z-skoru", "A0"), ("A3-s2s-norm (final)", "A3-s2s-norm"),
                              ("A5-mean-z", "A5-mean-z")):
        cells = _doc_rows(COMPARISON_DOC, doc_label)[0]
        shown = table.loc[METHOD_LABELS[method]]
        _assert_matches(_table_number(shown["Olay F1"]), cells[column])
        _assert_matches(_table_number(shown["PR-AUC"]), cells[column + 2])
