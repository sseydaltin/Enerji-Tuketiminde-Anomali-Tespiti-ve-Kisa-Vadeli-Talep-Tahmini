import numpy as np
import pandas as pd
import pytest
import torch

from src.models.gru import GRUForecaster
from src.models.lstm import LSTMForecaster
from src.train_forecast import _forecast_device, prepare_forecast_data


def make_hourly_data():
    """Train/val/test kümeleri için kontrollü saatlik tablo üretir."""
    index = pd.date_range("2024-01-01", periods=96, freq="h")
    values = np.sin(np.arange(len(index)) / 5).astype(np.float32)
    values[72:] += 1_000_000
    return pd.DataFrame(
        {
            "target": values,
            "weekend": (np.arange(len(index)) % 2).astype("int8"),
            "is_imputed": np.zeros(len(index), dtype="int8"),
        },
        index=index,
    )


def make_config():
    """Küçük testler için train/val/test ve eğitim ayarlarını döndürür."""
    return {
        "seed": 42,
        "paths": {
            "models_dir": "models",
            "results_dir": "results",
            "tables_dir": "results/tables",
            "figures_dir": "results/figures",
            "experiments_file": "results/experiments.csv",
        },
        "preprocess": {"target": "target"},
        "features": {
            "default_set": "B",
            "calendar": ["weekend"],
            "extra": [],
        },
        "split": {
            "train": {"start": "2024-01-01", "end": "2024-01-02"},
            "val": {"start": "2024-01-03", "end": "2024-01-03"},
            "test": {"start": "2024-01-04", "end": "2024-01-04"},
        },
        "forecast": {
            "lookback": 4,
            "horizon": 3,
            "stride": 1,
            "hidden": 4,
            "layers": 1,
            "dropout": 0.2,
            "lr": 0.001,
            "weight_decay": 0.00001,
            "batch_size": 8,
            "max_epochs": 1,
            "early_stopping_patience": 2,
            "scheduler_factor": 0.5,
            "scheduler_patience": 1,
            "grad_clip": 1.0,
        },
        "evaluation": {
            "exclude_imputed_targets": True,
            "mape_min_y": 0.1,
        },
    }


@pytest.mark.parametrize(
    "model",
    [LSTMForecaster, GRUForecaster],
)
def test_recurrent_models_map_168_steps_to_24_outputs(model):
    network = model(
        input_size=9,
        hidden_size=8,
        layers=2,
        dropout=0.2,
        horizon=24,
    )

    predictions = network(torch.zeros((2, 168, 9)))

    assert predictions.shape == (2, 24)


def test_forecast_device_accepts_cpu_and_rejects_unknown_preferences():
    assert _forecast_device("cpu").type == "cpu"

    with pytest.raises(ValueError, match="device tercihi"):
        _forecast_device("unknown")


def test_prepare_forecast_data_fits_scalers_on_train_only_and_excludes_imputed():
    frame = make_hourly_data()
    frame.loc[frame.index[48], "is_imputed"] = 1
    config = make_config()

    prepared = prepare_forecast_data(frame, config)

    assert prepared["feature_scaler"].mean_[0] == pytest.approx(
        frame.loc[frame.index[:48], "target"].mean()
    )
    assert prepared["target_scaler"].mean_[0] == pytest.approx(
        frame.loc[frame.index[:48], "target"].mean()
    )
    assert prepared["X_train"].shape[1:] == (4, 2)
    assert prepared["y_train"].shape[1:] == (3,)
    assert prepared["X_val"].shape[1:] == (4, 2)
    assert prepared["y_val"].shape[1:] == (3,)
    assert prepared["val_times"].min() == frame.index[49]
    assert prepared["val_times"].max() + pd.Timedelta(hours=2) <= frame.index[71]


@pytest.mark.parametrize("model_name", ["lstm", "gru"])
def test_train_model_saves_validation_outputs_only(
    tmp_path,
    monkeypatch,
    model_name,
):
    import src.train_forecast as trainer

    monkeypatch.setattr(trainer, "PROJECT_ROOT", tmp_path)
    summary, horizon_metrics, history = trainer.train_model(
        make_hourly_data(),
        make_config(),
        model_name,
        device_preference="cpu",
    )

    assert summary["split"] == "val"
    assert summary["windows"] == 22
    assert horizon_metrics["horizon"].tolist() == [1, 2, 3]
    assert len(history) == 1
    assert (tmp_path / "models" / f"{'F5' if model_name == 'lstm' else 'F6'}_best.pt").is_file()
    assert (tmp_path / "results/figures" / f"{'F5' if model_name == 'lstm' else 'F6'}_learning_curve.png").is_file()
    experiments = pd.read_csv(tmp_path / "results/experiments.csv")
    assert experiments["test_mae"].isna().all()
    assert experiments["test_rmse"].isna().all()
