import numpy as np
import pandas as pd
import pytest

from src.data.split import chronological_split
from src.models.baselines import naive_forecasts
from src.models.forecast_baselines import build_train_val_windows
from src.train_forecast import prepare_forecast_data
from src.utils.results_io import load_predictions, record_forecast_results, upsert_csv
from tests.test_forecast_models import make_config


IMPUTED_HOUR = pd.Timestamp("2024-01-13 05:00")


def make_long_frame_and_config():
    """168 → 24 pencereleri için 20 günlük sentetik saatlik veri ve config.

    Train 10 gün, val 5 gün, test 5 gün; val içinde bir saat doldurulmuş.
    """
    index = pd.date_range("2024-01-01", periods=20 * 24, freq="h")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "target": (1.0 + np.sin(np.arange(len(index)) * 2 * np.pi / 24)
                       + rng.random(len(index))).astype(np.float32),
            "weekend": (index.dayofweek >= 5).astype("int8"),
            "is_imputed": np.zeros(len(index), dtype="int8"),
        },
        index=index,
    )
    frame.loc[IMPUTED_HOUR, "is_imputed"] = 1
    config = make_config()
    config["split"] = {
        "train": {"start": "2024-01-01", "end": "2024-01-10"},
        "val": {"start": "2024-01-11", "end": "2024-01-15"},
        "test": {"start": "2024-01-16", "end": "2024-01-20"},
    }
    config["forecast"]["lookback"] = 168
    config["forecast"]["horizon"] = 24
    return frame, config


def test_all_pipelines_use_identical_val_windows_and_kw_targets():
    # Naive, Ridge ve MLP/LSTM/GRU hatları aynı hedef zamanlarını ve aynı
    # ham (kW) hedefleri vermeli; doldurulmuş hedefli pencereler hepsinden düşmeli.
    frame, config = make_long_frame_and_config()
    splits = chronological_split(frame, config)
    forecast = config["forecast"]

    y_naive, _, times_naive = naive_forecasts(
        frame, splits["val"], "target", forecast["lookback"],
        forecast["horizon"], exclude_imputed=True,
    )
    _, (_, y_ridge, times_ridge) = build_train_val_windows(
        splits, "target", forecast["lookback"], forecast["horizon"],
        exclude_imputed=True,
    )
    prepared = prepare_forecast_data(frame, config)

    assert times_naive.equals(times_ridge)
    assert prepared["val_times"].equals(times_ridge)
    np.testing.assert_allclose(y_naive, y_ridge, rtol=1e-6)
    np.testing.assert_allclose(prepared["y_val"], y_ridge, rtol=1e-6)
    # Val'de 5·24 − 24 + 1 = 97 olası pencere var; doldurulmuş saati
    # hedefinde içeren 24 pencere çıkarılmalı → 73 pencere kalır.
    assert len(times_ridge) == 97 - 24
    for start in times_ridge:
        assert not (start <= IMPUTED_HOUR <= start + pd.Timedelta(hours=23))


def test_record_forecast_results_round_trip_and_keeps_other_experiments(tmp_path):
    config = make_config()
    times = pd.date_range("2024-01-03", periods=4, freq="h")
    y_true = np.arange(12, dtype=float).reshape(4, 3) + 1.0
    y_pred = y_true + 0.5

    for exp_id in ("E1", "E2"):
        scores = record_forecast_results(
            tmp_path, config, "val", exp_id, "Model", y_true, y_pred, times,
            train_time_s=1.0, device="cpu", feature_set="B", lookback=4,
        )
    assert scores["MAE"] == pytest.approx(0.5)

    results = pd.read_csv(tmp_path / "results/tables/forecast_val_results.csv")
    assert sorted(results["exp_id"]) == ["E1", "E2"]
    loaded_true, loaded_pred, loaded_times = load_predictions(tmp_path, config, "E2")
    np.testing.assert_allclose(loaded_true, y_true)
    np.testing.assert_allclose(loaded_pred, y_pred)
    assert loaded_times.equals(times)


def test_upsert_csv_replaces_only_matching_key(tmp_path):
    path = tmp_path / "table.csv"
    upsert_csv(path, [{"exp_id": "A", "v": 1}, {"exp_id": "B", "v": 2}], ["exp_id"])
    upsert_csv(path, [{"exp_id": "A", "v": 3}], ["exp_id"])

    table = pd.read_csv(path).set_index("exp_id")
    assert table.loc["A", "v"] == 3
    assert table.loc["B", "v"] == 2
