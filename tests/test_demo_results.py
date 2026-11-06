"""Kayıtlı Results sayfası kaynak, veri kümesi ve seed eşleşme testleri."""

import pandas as pd
import pytest

from app.pages.results import _cross_dataset_table, _forecast_table, _horizon_figure
from app.results_services import (
    ANOMALY_METHODS,
    FORECAST_IDS,
    _read_csv,
    load_results_bundle,
)

EXPECTED_TEST_SEEDS = {2, 12, 22, 32, 42}


@pytest.fixture(scope="module")
def household_results():
    """Gerçek household final test CSV'lerini okur; eğitim veya inference yapmaz."""
    return load_results_bundle("household")


@pytest.fixture(scope="module")
def eld_results():
    """Gerçek ELD final test CSV'lerini okur; ham ELD verisi gerektirmez."""
    return load_results_bundle("eld")


def test_household_results_use_household_files_and_three_f13_seeds(household_results):
    """Household tablo kaynaklarının yolunu, test split'ini ve F13 seed sayısını denetler."""
    assert household_results["dataset"] == "household"
    assert household_results["sources"]["forecast"].endswith(
        "results/tables/forecast_test_results.csv"
    )
    assert household_results["sources"]["horizon"].endswith(
        "results/tables/forecast_test_horizon.csv"
    )
    assert household_results["sources"]["anomaly"].endswith(
        "results/tables/anomaly_protocol_test.csv"
    )
    assert household_results["sources"]["cross_dataset"].endswith(
        "results/eld/tables/hane_vs_sebeke_forecast.csv"
    )
    forecast = household_results["forecast"].set_index("model_key")
    assert forecast.loc["F13-ens", "n_seeds"] == 3
    assert set(forecast.loc["F13-ens", "source_exp_ids"]) == set(
        FORECAST_IDS["household"]["F13-ens"]
    )
    assert forecast.loc["XGBoost", "n_seeds"] == 1
    assert forecast.loc["SN168", "n_seeds"] == 1


def test_eld_results_use_eld_files_and_dataset_prefixed_experiments(eld_results):
    """ELD sonuçlarının ELD tablolarından ve ELD deney kimliklerinden geldiğini doğrular."""
    assert eld_results["dataset"] == "eld"
    assert "/results/eld/tables/" in eld_results["sources"]["forecast"]
    forecast = eld_results["forecast"].set_index("model_key")
    assert set(forecast.loc["F13-ens", "source_exp_ids"]) == set(
        FORECAST_IDS["eld"]["F13-ens"]
    )
    assert all(
        str(exp_id).startswith("ELD-")
        for exp_ids in forecast["source_exp_ids"]
        for exp_id in exp_ids
    )
    assert forecast.loc["F13-ens", "n_seeds"] == 3
    assert eld_results["unit"] == "kW"


def test_horizon_results_have_24_steps_and_only_recorded_target_models(household_results):
    """Horizon grafiğinin üç modeli 24 adım üzerinden doğru dosyadan kullandığını denetler."""
    horizon = household_results["horizon"]
    assert set(horizon["model_key"]) == {"F13-ens", "XGBoost", "SN168"}
    for _, model_rows in horizon.groupby("model_key"):
        assert sorted(model_rows["horizon"]) == list(range(1, 25))
    f13 = horizon.loc[horizon["model_key"] == "F13-ens"]
    assert (f13["n_seeds"] == 3).all()
    figure = _horizon_figure(horizon, "Household (tek hane)", "kW")
    assert len(figure.data) == 3
    assert figure.layout.yaxis.title.text == "MAE (kW)"
    assert "Household" in figure.layout.title.text


@pytest.mark.parametrize("fixture_name", ["household_results", "eld_results"])
def test_anomaly_rows_are_test_main_validation_selected_and_seed_matched(request, fixture_name):
    """Anomali özetinin yalnız test/main/evf1max ve per-seed doğrulamasını kullandığını sınar."""
    bundle = request.getfixturevalue(fixture_name)
    summary = bundle["anomaly"]
    assert set(summary["method"]) == set(ANOMALY_METHODS)
    assert (summary["split"] == "test").all()
    assert (summary["scenario"] == "main").all()
    assert (summary["rule"] == "evf1max").all()
    assert (summary["n_seeds"] == 5).all()
    seed_path = bundle["sources"]["anomaly_per_seed"]
    seed_rows = pd.read_csv(seed_path)
    seed_rows = seed_rows.loc[
        (seed_rows["split"] == "test")
        & (seed_rows["scenario"] == "main")
        & (seed_rows["rule"] == "evf1max")
        & seed_rows["method"].isin(ANOMALY_METHODS)
    ]
    assert set(seed_rows["seed"].astype(int)) == EXPECTED_TEST_SEEDS
    assert summary["event_recall_spike_mean"].between(0, 1).all()
    assert summary["event_recall_drop_mean"].between(0, 1).all()
    assert summary["event_recall_level_shift_mean"].between(0, 1).all()
    assert summary["event_recall_pattern_mean"].between(0, 1).all()


def test_cross_dataset_display_never_exposes_direct_mae_comparison(household_results):
    """Hane/ELD ortak görünümünde MAE bulunmamalı; yalnız ölçekten bağımsız kıyaslar kalmalı."""
    comparison = _cross_dataset_table(household_results["cross_dataset"])
    assert "MAE" not in comparison.columns
    assert "MAE (kW)" not in comparison.columns
    assert set(comparison["Veri kümesi"]) == {
        "Household (tek hane)",
        "ELD (160 müşteri toplamı)",
    }
    assert {"MAPE* (%)", "R²", "SN168'e göre MAE kazancı"}.issubset(comparison.columns)


def test_cross_dataset_values_come_from_saved_comparison_csv(household_results):
    """Ortak MAPE/R²/kazanç satırları kayıtlı 24 saat karşılaştırma CSV'siyle eşleşmelidir."""
    saved = pd.read_csv(household_results["sources"]["cross_dataset"])
    saved = saved.loc[saved["level"] == "b_24h"]
    cross = household_results["cross_dataset"].set_index(["dataset", "model_key"])
    group_names = {
        "F13-ens": "F13-ens (final)",
        "XGBoost": "F12-L336 XGBoost",
        "SN168": "F2 Seasonal Naive 168",
    }
    for model_key, group in group_names.items():
        row = saved.loc[saved["group"] == group].iloc[0]
        assert cross.loc[("household", model_key), "MAPE*"] == pytest.approx(row["hane_MAPE"])
        assert cross.loc[("eld", model_key), "MAPE*"] == pytest.approx(row["sebeke_MAPE"])


def test_missing_result_csv_fails_loudly(tmp_path):
    """Eksik kaynak dosyası boş/yanlış sonuç gibi ele alınmamalıdır."""
    with pytest.raises(FileNotFoundError, match="Kayıtlı sonuç dosyası bulunamadı"):
        _read_csv(tmp_path / "missing.csv", ("split", "MAE"))


def test_forecast_metric_display_preserves_seed_uncertainty(household_results):
    """Çok-seed F13 metriklerinde std, tek-seed modellerde ise uydurma std üretilmez."""
    display = _forecast_table(household_results["forecast"]).set_index("Model")
    f13 = display.loc["F13 ensemble (0,8 F10-opt + 0,2 F5-opt)"]
    xgb = display.loc["F12-L336 XGBoost"]
    assert "±" in f13["MAE"]
    assert "±" not in xgb["MAE"]
    assert f13["Seed sayısı"] == 3
    assert xgb["Seed sayısı"] == 1
