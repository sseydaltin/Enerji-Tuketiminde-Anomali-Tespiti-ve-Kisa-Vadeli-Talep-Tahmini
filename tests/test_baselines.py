import numpy as np
import pytest

from src.models.baselines import (
    persistence,
    seasonal_naive,
    evaluate,
)


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
