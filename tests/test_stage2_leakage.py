"""Aşama 2 sızıntı testleri: kullanılan her geçmiş değer tahmin başlangıcından önce mi?"""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.data.split import chronological_split
from src.models.forecast_baselines import build_train_val_windows, ridge_design_matrix
from src.train_forecast import prepare_forecast_data
from tests.test_eval_consistency import make_long_frame_and_config


def frame_with_clock():
    """Sentetik veri + 'clock' sütunu (satır numarası) ile C seti config'i.

    'clock' değeri satırın zamanını kodlar; X'e giren en büyük clock değeri
    ilk hedefin satır numarasından küçük olmalıdır.
    """
    frame, config = make_long_frame_and_config()
    frame["clock"] = np.arange(len(frame), dtype=np.float64)
    config["features"]["extra"] = ["clock"]
    return frame, config


def test_future_features_use_only_values_before_forecast_start():
    frame, config = make_long_frame_and_config()
    config["forecast"]["future_features"] = True
    prepared = prepare_forecast_data(frame, config)

    future = prepared["future_val"]  # [N, 24, 1 + takvim]
    assert future.shape[1:] == (24, 1 + len(config["features"]["calendar"]))
    scaler = prepared["level_scaler"]
    for i, start in enumerate(prepared["val_times"][:30]):
        for h in range(24):
            lag_time = start + pd.Timedelta(hours=h - 168)
            # Haftalık gecikme tahmin başlangıcından (start) önce olmalı.
            assert lag_time < start
            expected = (frame.loc[lag_time, "target"] - scaler.mean_[0]) / scaler.scale_[0]
            assert future[i, h, 0] == pytest.approx(expected, rel=1e-5)
            # Takvim, hedef saatin kendi takvimidir (önceden bilinir).
            target_time = start + pd.Timedelta(hours=h)
            assert future[i, h, 1] == frame.loc[target_time, "weekend"]


def test_future_and_residual_drop_train_windows_without_weekly_history():
    # lookback=48 < 168: ilk pencerelerin haftalık geçmişi yoktur, atılmalıdır.
    frame, config = make_long_frame_and_config()
    config["forecast"]["lookback"] = 48
    plain = prepare_forecast_data(frame, deepcopy(config))
    config["forecast"]["future_features"] = True
    prepared = prepare_forecast_data(frame, config)

    train_start = frame.index[0]
    # Her train penceresinin en erken haftalık gecikmesi train içinde olmalı.
    assert (prepared["train_times"] - pd.Timedelta(hours=168) >= train_start).all()
    assert len(prepared["X_train"]) == len(plain["X_train"]) - (168 - 48)
    # Val pencereleri atılmaz (train sonundan yeterli geçmiş alınır).
    assert prepared["val_times"].equals(plain["val_times"])


def test_residual_target_base_is_seasonal_naive_from_the_past():
    frame, config = make_long_frame_and_config()
    config["forecast"]["residual_target"] = True
    prepared = prepare_forecast_data(frame, config)

    for i, start in enumerate(prepared["val_times"][:30]):
        for h in range(24):
            lag_time = start + pd.Timedelta(hours=h - 168)
            assert lag_time < start
            assert prepared["base_val"][i, h] == pytest.approx(
                frame.loc[lag_time, "target"], rel=1e-6
            )
    # y_val ham kW'da kalır; ölçekli artık + taban geri y'yi verir.
    scaler = prepared["target_scaler"]
    restored = scaler.inverse_transform(
        prepared["y_val_scaled"].reshape(-1, 1)
    ).reshape(prepared["y_val"].shape) + prepared["base_val"]
    np.testing.assert_allclose(restored, prepared["y_val"], rtol=1e-5)


def test_residual_scaler_is_fitted_on_train_residuals_only():
    frame, config = make_long_frame_and_config()
    config["forecast"]["residual_target"] = True
    prepared = prepare_forecast_data(frame, config)

    train_residuals = prepared["y_train"] - prepared["base_train"]
    assert prepared["target_scaler"].mean_[0] == pytest.approx(train_residuals.mean())
    # Val hedefleri train sonrasında; ölçekleyici onları görmemiş olmalı.
    val_residuals = prepared["y_val"] - prepared["base_val"]
    assert prepared["target_scaler"].mean_[0] != pytest.approx(val_residuals.mean())


@pytest.mark.parametrize("feature_set", ["A", "B", "C"])
def test_inputs_end_before_first_target_for_every_feature_set(feature_set):
    frame, config = frame_with_clock()
    config["features"]["default_set"] = feature_set
    if feature_set != "C":
        frame = frame.drop(columns="clock")
        config["features"]["extra"] = []
    prepared = prepare_forecast_data(frame, config)

    if feature_set == "C":
        clock_column = prepared["features"].index("clock")
        scaler = prepared["feature_scaler"]
        clock = (
            prepared["X_val"][:, :, clock_column] * scaler.scale_[clock_column]
            + scaler.mean_[clock_column]
        )
        first_target_rows = frame.index.get_indexer(prepared["val_times"])
        # X'teki en yeni satır = ilk hedefin bir önceki satırı.
        np.testing.assert_allclose(clock[:, -1], first_target_rows - 1, atol=1e-3)
        assert (clock.max(axis=1) < first_target_rows).all()
    assert prepared["X_val"].shape[1:] == (168, len(prepared["features"]))


def test_ridge_calendar_comes_from_last_observed_hour():
    frame, config = make_long_frame_and_config()
    splits = chronological_split(frame, config)
    _, (X_val, _, val_times) = build_train_val_windows(
        splits, "target", 168, 24, True, features=["target", "weekend"],
    )
    design = ridge_design_matrix(X_val, [1, 24, 168], use_calendar=True)

    assert design.shape == (len(X_val), 3 + 1)
    for i, start in enumerate(val_times[:30]):
        last_observed = start - pd.Timedelta(hours=1)
        assert design[i, 0] == pytest.approx(frame.loc[last_observed, "target"])
        assert design[i, 3] == frame.loc[last_observed, "weekend"]


def test_shuffle_is_reproducible_with_same_seed(tmp_path, monkeypatch):
    import src.train_forecast as trainer

    monkeypatch.setattr(trainer, "PROJECT_ROOT", tmp_path)
    frame, config = make_long_frame_and_config()
    config["forecast"]["shuffle_train"] = True
    config["forecast"]["max_epochs"] = 2

    first, _ = trainer.train_model(frame, deepcopy(config), "gru", "cpu", exp_id="S1")
    second, _ = trainer.train_model(frame, deepcopy(config), "gru", "cpu", exp_id="S2")
    assert first["MAE"] == pytest.approx(second["MAE"])


@pytest.mark.parametrize("model_name", ["mlp", "lstm", "gru"])
def test_models_train_with_future_features_and_residual_target(
    tmp_path, monkeypatch, model_name,
):
    import src.train_forecast as trainer

    monkeypatch.setattr(trainer, "PROJECT_ROOT", tmp_path)
    frame, config = make_long_frame_and_config()
    config["forecast"]["future_features"] = True
    config["forecast"]["residual_target"] = True

    scores, _ = trainer.train_model(frame, config, model_name, "cpu", exp_id="X")
    assert np.isfinite(scores["MAE"])
