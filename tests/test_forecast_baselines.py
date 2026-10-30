import numpy as np
import pandas as pd
import pytest
from sklearn.neural_network import MLPRegressor

from src.models.forecast_baselines import (
    build_train_val_windows,
    fit_mlp,
    fit_ridge,
    run_forecast_baselines,
    select_ridge_lags,
)


def make_hourly_data():
    """Train/val/test bölümleri için küçük, kronolojik saatlik seri üretir."""
    index = pd.date_range("2024-01-01", periods=96, freq="h")
    values = np.arange(len(index), dtype=np.float32) / 10
    return pd.DataFrame(
        {
            "target": values,
            "is_imputed": np.zeros(len(index), dtype="int8"),
        },
        index=index,
    )


def make_config():
    """Test için gerçekçi, küçük baseline yapılandırması döndürür."""
    return {
        "seed": 42,
        "paths": {
            "results_dir": "results",
            "tables_dir": "results/tables",
            "experiments_file": "results/experiments.csv",
            "models_dir": "models",
        },
        "preprocess": {"target": "target"},
        "split": {
            "train": {"start": "2024-01-01", "end": "2024-01-02"},
            "val": {"start": "2024-01-03", "end": "2024-01-03"},
            "test": {"start": "2024-01-04", "end": "2024-01-04"},
        },
        "forecast": {
            "lookback": 4,
            "horizon": 3,
            "stride": 1,
            "ridge": {"lags": [1, 2, 4], "alpha": 1.0},
            "mlp": {"hidden_sizes": [4, 3]},
            "lr": 0.001,
            "weight_decay": 0.00001,
            "batch_size": 8,
            "max_epochs": 2,
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


def test_configured_lags_select_newest_to_oldest_history_values():
    windows = np.arange(12, dtype=np.float32).reshape(1, 4, 3)

    result = select_ridge_lags(windows, [1, 2, 4])

    np.testing.assert_array_equal(result, [[9, 6, 0]])


def test_mlp_regressor_supports_168_hour_input_and_24_hour_output():
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(12, 168, 1)).astype(np.float32)
    y_train = rng.normal(size=(12, 24)).astype(np.float32)
    X_val = rng.normal(size=(3, 168, 1)).astype(np.float32)
    y_val = rng.normal(size=(3, 24)).astype(np.float32)
    forecast = make_config()["forecast"]
    forecast["max_epochs"] = 2
    forecast["mlp"]["hidden_sizes"] = [4, 3]

    (
        predictions,
        x_scaler,
        y_scaler,
        model,
        epochs,
        _,
        _,
        convergence_warning,
    ) = fit_mlp(
        X_train,
        y_train,
        X_val,
        y_val,
        np.arange(20, dtype=float),
        forecast,
        seed=42,
    )

    assert isinstance(model, MLPRegressor)
    assert predictions.shape == (3, 24)
    assert epochs <= 2
    assert isinstance(convergence_warning, bool)
    assert model.early_stopping is False
    assert x_scaler.mean_[0] == pytest.approx(X_train.mean())
    assert y_scaler.mean_[0] == pytest.approx(np.arange(20).mean())


def test_build_windows_keeps_train_and_val_targets_in_their_splits():
    frame = make_hourly_data()
    splits = {
        "train": frame.iloc[:48],
        "val": frame.iloc[48:72].copy(),
        "test": frame.iloc[72:],
    }
    splits["val"].loc[splits["val"].index[0], "is_imputed"] = 1

    train_windows, val_windows = build_train_val_windows(
        splits,
        "target",
        lookback=4,
        horizon=3,
        exclude_imputed=True,
    )

    assert train_windows[0].shape[1:] == (4, 1)
    assert train_windows[1].shape[1:] == (3,)
    assert train_windows[2].max() < splits["val"].index.min()
    assert val_windows[2].min() == splits["val"].index.min() + pd.Timedelta(hours=1)
    assert val_windows[2].max() + pd.Timedelta(hours=2) <= (
        splits["val"].index.max()
    )
    assert len(val_windows[0]) == 21


def test_ridge_scalers_are_fitted_only_on_train_targets():
    train_target = np.arange(10, dtype=float)
    X_train = np.arange(12, dtype=float).reshape(4, 3)
    y_train = np.arange(8, dtype=float).reshape(4, 2)
    X_val = np.full((2, 3), 1000.0)
    y_val = np.zeros((2, 2))

    _, x_scaler, y_scaler, _ = fit_ridge(
        X_train,
        y_train,
        X_val,
        train_target,
        alpha=1.0,
    )

    np.testing.assert_allclose(x_scaler.mean_, X_train.mean(axis=0))
    assert y_scaler.mean_[0] == pytest.approx(train_target.mean())


def test_ridge_rejects_lags_outside_history():
    windows = np.zeros((2, 4, 1))

    with pytest.raises(ValueError):
        select_ridge_lags(windows, [5])


def test_run_uses_val_only_and_logs_schema(tmp_path, monkeypatch):
    frame = make_hourly_data()
    config = make_config()
    config["paths"]["results_dir"] = "results"
    monkeypatch.setattr(
        "src.models.forecast_baselines.PROJECT_ROOT",
        tmp_path,
    )

    results = run_forecast_baselines(frame, config)

    assert results["split"].tolist() == ["val", "val"]
    assert results["model"].tolist() == ["Ridge", "MLP"]
    assert set(results["windows"]) == {22}

    experiments = pd.read_csv(tmp_path / "results/experiments.csv")
    assert experiments["exp_id"].tolist() == ["F3", "F4"]
    assert experiments["test_mae"].isna().all()
    assert experiments["test_rmse"].isna().all()
    assert {"MAPE*", "sMAPE", "R2"}.issubset(results.columns)

    horizon_metrics = pd.read_csv(
        tmp_path / "results/tables/forecast_horizon_metrics.csv"
    )
    assert set(horizon_metrics["horizon"]) == {1, 2, 3}
    assert set(horizon_metrics["split"]) == {"val"}
