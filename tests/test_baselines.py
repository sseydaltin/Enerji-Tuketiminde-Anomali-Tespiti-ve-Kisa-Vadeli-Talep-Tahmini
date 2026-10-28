import numpy as np
import pandas as pd
import pytest

from src.models import baselines
from src.models.baselines import evaluate, persistence, seasonal_naive


def test_persistence():
    history = np.array([1, 2, 3, 4], dtype=float)

    result = persistence(history, horizon=3)

    np.testing.assert_array_equal(result, [4, 4, 4])


def test_seasonal_naive_24():
    history = np.arange(168, dtype=float)

    result = seasonal_naive(history, horizon=24, period=24)

    np.testing.assert_array_equal(
        result, np.arange(144, 168)
    )


def test_seasonal_naive_168():
    history = np.arange(168, dtype=float)

    result = seasonal_naive(history, horizon=24, period=168)

    np.testing.assert_array_equal(
        result, np.arange(24)
    )


def test_seasonal_naive_rejects_short_history():
    history = np.arange(12, dtype=float)

    with pytest.raises(ValueError):
        seasonal_naive(history, horizon=24, period=168)


def test_evaluate_perfect_prediction():
    actual = np.array([1, 2, 3], dtype=float)
    predicted = np.array([1, 2, 3], dtype=float)

    scores = evaluate(actual, predicted)

    assert scores["MAE"] == pytest.approx(0)
    assert scores["RMSE"] == pytest.approx(0)


def test_evaluate_known_error():
    actual = np.array([1, 2, 3], dtype=float)
    predicted = np.array([2, 3, 4], dtype=float)

    scores = evaluate(actual, predicted)

    assert scores["MAE"] == pytest.approx(1)
    assert scores["RMSE"] == pytest.approx(1)


@pytest.mark.parametrize(
    ("arguments", "expected_split"),
    [([], "val"), (["--split", "test"], "test")],
)
def test_main_evaluates_only_requested_split(
    tmp_path,
    monkeypatch,
    arguments,
    expected_split,
):
    index = pd.date_range("2024-01-01", periods=180, freq="h")
    frame = pd.DataFrame(
        {
            "target": np.arange(len(index), dtype=float),
            "is_imputed": np.zeros(len(index), dtype="int8"),
        },
        index=index,
    )
    splits = {
        "val": frame.iloc[168:174],
        "test": frame.iloc[174:180],
    }
    config = {
        "paths": {"processed_file": "hourly.parquet"},
        "preprocess": {"target": "target"},
        "forecast": {"lookback": 168, "horizon": 2},
        "evaluation": {"exclude_imputed_targets": False},
    }
    monkeypatch.setattr(baselines, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(baselines, "load_config", lambda: config)
    monkeypatch.setattr(pd, "read_parquet", lambda _: frame)
    monkeypatch.setattr(baselines, "chronological_split", lambda *_: splits)
    monkeypatch.setattr("sys.argv", ["baselines.py", *arguments])

    baselines.main()

    results = pd.read_csv(tmp_path / "results/tables/baseline_results.csv")
    assert set(results["split"]) == {expected_split}
    assert len(results) == 3
