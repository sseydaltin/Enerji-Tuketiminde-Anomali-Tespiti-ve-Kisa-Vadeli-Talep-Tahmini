"""Aşama 3: seq2seq, CNN-LSTM, XGBoost ve lookback ızgarası testleri."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.cnn_lstm import CNNLSTMForecaster
from src.models.seq2seq import Seq2SeqForecaster
from src.models.xgb_forecast import run_xgboost, xgboost_lags
from src.train_forecast import prepare_forecast_data
from tests.test_eval_consistency import make_long_frame_and_config


def stage3_config():
    """Sentetik veri için CNN ve XGBoost ayarlarını ekler."""
    frame, config = make_long_frame_and_config()
    config["forecast"].update({
        "hidden": 8,
        "layers": 2,
        "dropout": 0.1,
        "cnn": {"channels": 4, "kernel_size": 3, "pool": 2},
        "xgboost": {
            "lags": [1, 2, 24, 48, 168, 336],
            "n_estimators": 20,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "early_stopping_rounds": 5,
        },
    })
    return frame, config


@pytest.mark.parametrize("lookback", [48, 168, 336])
def test_seq2seq_and_cnn_lstm_map_window_to_24_outputs(lookback):
    inputs = torch.zeros((2, lookback, 9))
    calendar = torch.zeros((2, 24, 8))

    seq2seq = Seq2SeqForecaster(9, 8, hidden_size=16, layers=2, dropout=0.2)
    cnn_lstm = CNNLSTMForecaster(9, channels=8, kernel_size=3, pool=2,
                                 hidden_size=16, layers=2, dropout=0.2, horizon=24)

    assert seq2seq(inputs, calendar).shape == (2, 24)
    assert cnn_lstm(inputs).shape == (2, 24)


def test_decoder_calendar_is_target_hour_calendar_without_target_values():
    frame, config = stage3_config()
    config["forecast"]["decoder_calendar"] = True
    prepared = prepare_forecast_data(frame, config)

    future = prepared["future_val"]
    calendar = config["features"]["calendar"]
    assert future.shape[1:] == (24, len(calendar))
    for i, start in enumerate(prepared["val_times"][:30]):
        for h in range(24):
            target_time = start + pd.Timedelta(hours=h)
            assert future[i, h, 0] == frame.loc[target_time, "weekend"]
    # Decoder girdisi hedef sütununu içermez.
    assert "target" not in calendar


def test_val_windows_do_not_depend_on_lookback():
    frame, config = stage3_config()
    times = {}
    for lookback in (48, 168):
        lookback_config = deepcopy(config)
        lookback_config["forecast"]["lookback"] = lookback
        times[lookback] = prepare_forecast_data(frame, lookback_config)["val_times"]
    # Val hedefleri aynı; yalnız girdi geçmişinin uzunluğu değişir.
    assert times[48].equals(times[168])


def test_xgboost_lags_respect_lookback():
    _, config = stage3_config()
    config["forecast"]["lookback"] = 48
    assert xgboost_lags(config) == [1, 2, 24, 48]
    config["forecast"]["lookback"] = 336
    assert xgboost_lags(config) == [1, 2, 24, 48, 168, 336]


def test_xgboost_runs_on_val_only_and_logs(tmp_path, monkeypatch):
    frame, config = stage3_config()
    monkeypatch.setattr("src.models.xgb_forecast.PROJECT_ROOT", tmp_path)

    scores = run_xgboost(frame, config, exp_id="F12-test", device_preference="cpu")

    assert np.isfinite(scores["MAE"])
    results = pd.read_csv(tmp_path / "results/tables/forecast_val_results.csv")
    assert results["exp_id"].tolist() == ["F12-test"]
    experiments = pd.read_csv(tmp_path / "results/experiments.csv")
    assert experiments["test_mae"].isna().all()
    assert experiments.loc[0, "best_epoch"] <= experiments.loc[0, "epochs_run"]


@pytest.mark.parametrize("model_name", ["seq2seq", "cnn_lstm"])
def test_new_architectures_train_with_shared_loop(tmp_path, monkeypatch, model_name):
    import src.train_forecast as trainer

    monkeypatch.setattr(trainer, "PROJECT_ROOT", tmp_path)
    frame, config = stage3_config()

    scores, history = trainer.train_model(frame, config, model_name, "cpu", exp_id="T")
    assert np.isfinite(scores["MAE"])
    assert len(history) == 1
    # Eğitim çağrısı kullanıcının config nesnesini değiştirmemeli.
    assert "decoder_calendar" not in config["forecast"]
