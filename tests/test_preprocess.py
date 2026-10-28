import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    MEAN_COLUMNS,
    SUM_COLUMNS,
    fill_missing_hours,
)


def make_hourly_data():
    index = pd.date_range(
        "2024-01-01", periods=200, freq="h"
    )
    data = {
        col: np.full(200, 1.0)
        for col in MEAN_COLUMNS + SUM_COLUMNS
    }
    return pd.DataFrame(data, index=index)


def test_short_gap_interpolation():
    df = make_hourly_data()
    df["Global_active_power"] = 1.0
    df.loc[df.index[10], "Global_active_power"] = 2.0
    df.loc[df.index[13], "Global_active_power"] = 5.0
    df.loc[df.index[11:13], "Global_active_power"] = np.nan

    result = fill_missing_hours(df, max_gap=3, lag=168)

    assert result.iloc[11]["Global_active_power"] == pytest.approx(3.0)
    assert result.iloc[12]["Global_active_power"] == pytest.approx(4.0)


def test_imputation_flag():
    df = make_hourly_data()
    df.loc[df.index[20], "Global_active_power"] = np.nan

    result = fill_missing_hours(df, max_gap=3, lag=168)

    assert result.iloc[20]["is_imputed"] == 1
    assert result.iloc[19]["is_imputed"] == 0


def test_no_negative_values_from_short_gap():
    df = make_hourly_data()
    df.loc[df.index[30], "Global_active_power"] = 0.2
    df.loc[df.index[31], "Global_active_power"] = np.nan
    df.loc[df.index[32], "Global_active_power"] = 0.3

    result = fill_missing_hours(df, max_gap=3, lag=168)

    assert result["Global_active_power"].min() >= 0


def test_no_remaining_missing_values():
    df = make_hourly_data()
    df.loc[df.index[50:52], "Global_active_power"] = np.nan

    result = fill_missing_hours(df, max_gap=3, lag=168)

    assert result.isna().sum().sum() == 0


def test_boundary_neighbors_are_not_interpolated():
    df = make_hourly_data()
    for column in MEAN_COLUMNS + SUM_COLUMNS:
        df[column] = np.arange(len(df), dtype=float)

    boundary = df.index[96]
    missing_indices = [df.index[94], df.index[96], df.index[98]]
    df.loc[missing_indices, MEAN_COLUMNS + SUM_COLUMNS] = np.nan

    result = fill_missing_hours(
        df,
        max_gap=3,
        lag=24,
        split_boundaries=[boundary],
    )

    for column in MEAN_COLUMNS + SUM_COLUMNS:
        expected = df[column].shift(24).loc[missing_indices]
        pd.testing.assert_series_equal(
            result.loc[missing_indices, column],
            expected,
            check_names=False,
        )
