import numpy as np
import pytest

from src.utils.metrics import calculate_metrics, metrics_by_horizon


def test_calculate_metrics_ignores_near_zero_targets_in_mape():
    actual = np.array([0.0, 0.05, 1.0, 2.0])
    predicted = np.array([0.0, 0.2, 2.0, 1.0])

    scores = calculate_metrics(actual, predicted, mape_min_y=0.1)

    assert scores["MAE"] == pytest.approx(0.5375)
    assert scores["RMSE"] == pytest.approx(np.sqrt(0.505625))
    assert scores["MAPE*"] == pytest.approx(75.0)
    assert scores["sMAPE"] == pytest.approx(
        (0.0 + 1.2 + 2 / 3 + 2 / 3) / 4 * 100
    )
    assert scores["R2"] == pytest.approx(1 - 2.0225 / 2.676875)


def test_mape_is_nan_when_no_target_is_above_threshold():
    scores = calculate_metrics([0.0, 0.1], [0.2, 0.3], mape_min_y=0.1)

    assert np.isnan(scores["MAPE*"])


def test_smape_is_zero_when_both_values_are_zero():
    scores = calculate_metrics([0.0, 1.0], [0.0, 1.0])

    assert scores["sMAPE"] == pytest.approx(0.0)


def test_metrics_by_horizon_returns_one_row_per_forecast_step():
    actual = np.array([[1.0, 2.0], [3.0, 4.0]])
    predicted = np.array([[2.0, 2.0], [3.0, 6.0]])

    result = metrics_by_horizon(actual, predicted)

    assert result["horizon"].tolist() == [1, 2]
    assert result["MAE"].tolist() == pytest.approx([0.5, 1.0])


@pytest.mark.parametrize(
    ("actual", "predicted"),
    [
        ([1.0, 2.0], [1.0]),
        ([[1.0]], [[[1.0]]]),
        ([np.nan], [0.0]),
    ],
)
def test_metrics_reject_invalid_shapes_and_non_finite_values(actual, predicted):
    with pytest.raises(ValueError):
        calculate_metrics(actual, predicted)
