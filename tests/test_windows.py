import numpy as np
import pandas as pd
import pytest

from src.data.windows import make_windows, build_datasets


def sample_data():
    idx = pd.date_range("2024-01-01", periods=120, freq="h")
    return pd.DataFrame({
        "Global_active_power": np.arange(120, dtype=float),
        "is_imputed": np.zeros(120, dtype=int),
    }, index=idx)


def test_window_shapes():
    df = sample_data()

    X, y, times = make_windows(
        df, "Global_active_power", ["Global_active_power"],
        lookback=24, horizon=6
    )

    assert X.shape == (91, 24, 1)
    assert y.shape == (91, 6)
    assert len(times) == 91


def test_window_values():
    df = sample_data()

    X, y, times = make_windows(
        df, "Global_active_power", ["Global_active_power"],
        lookback=24, horizon=6
    )

    np.testing.assert_array_equal(X[0, :, 0], np.arange(24))
    np.testing.assert_array_equal(y[0], np.arange(24, 30))
    assert times[0] == df.index[24]


def test_excludes_imputed_targets():
    df = sample_data()
    df.loc[df.index[24], "is_imputed"] = 1

    X, y, times = make_windows(
        df, "Global_active_power", ["Global_active_power"],
        lookback=24, horizon=6,
        exclude_imputed=True,
    )

    assert df.index[24] not in times
    assert len(X) == 90


def test_validation_uses_past_context():
    df = sample_data()
    boundary = df.index[72]

    X, y, times = make_windows(
        df, "Global_active_power", ["Global_active_power"],
        lookback=24, horizon=6,
        target_start=boundary,
        target_end=df.index[-1],
    )

    assert times[0] == boundary
    np.testing.assert_array_equal(
        X[0, :, 0], np.arange(48, 72)
    )
    np.testing.assert_array_equal(
        y[0], np.arange(72, 78)
    )


def test_scaler_fitted_only_on_train():
    df = sample_data()
    df.loc[df.index[72]:, "Global_active_power"] += 1000

    config = {
        "preprocess": {"target": "Global_active_power"},
        "features": {
            "default_set": "A",
            "calendar": [],
            "extra": [],
        },
        "forecast": {
            "lookback": 12,
            "horizon": 6,
            "stride": 1,
        },
        "evaluation": {"exclude_imputed_targets": False},
        "split": {
            "train": {"start": "2024-01-01", "end": "2024-01-02"},
            "val": {"start": "2024-01-03", "end": "2024-01-03"},
            "test": {"start": "2024-01-04", "end": "2024-01-05"},
        },
    }

    datasets, scaler, features = build_datasets(df, config)

    assert features == ["Global_active_power"]
    assert scaler.mean_[0] == pytest.approx(np.mean(np.arange(48)))
    assert datasets["val"][0].shape[0] > 0
    assert datasets["test"][0].shape[0] > 0

    # Hedef değerler orijinal ölçekte kalmalı.
    assert datasets["val"][1][0, 0] == pytest.approx(
        df.loc[datasets["val"][2][0], "Global_active_power"]
    )
