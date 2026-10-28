import pandas as pd
import pytest

from src.data.split import chronological_split


@pytest.fixture
def config():
    return {
        "split": {
            "train": {"start": "2024-01-01", "end": "2024-01-02"},
            "val": {"start": "2024-01-03", "end": "2024-01-03"},
            "test": {"start": "2024-01-04", "end": "2024-01-04"},
        }
    }


@pytest.fixture
def hourly_data():
    index = pd.date_range(
        "2024-01-01", "2024-01-04 23:00", freq="h"
    )
    return pd.DataFrame({"target": range(len(index))}, index=index)


def test_chronological_split(hourly_data, config):
    result = chronological_split(hourly_data, config)

    assert len(result["train"]) == 48
    assert len(result["val"]) == 24
    assert len(result["test"]) == 24

    assert result["train"].index.max() < result["val"].index.min()
    assert result["val"].index.max() < result["test"].index.min()


def test_split_rejects_duplicate_timestamps(hourly_data, config):
    duplicate = pd.concat([hourly_data, hourly_data.iloc[[0]]])
    duplicate = duplicate.sort_index()

    with pytest.raises(ValueError):
        chronological_split(duplicate, config)


def test_split_rejects_unsorted_timestamps(hourly_data, config):
    shuffled = hourly_data.iloc[::-1]

    with pytest.raises(ValueError):
        chronological_split(shuffled, config)


def test_split_rejects_empty_partition(hourly_data, config):
    config["split"]["test"] = {
        "start": "2025-01-01",
        "end": "2025-01-02",
    }

    with pytest.raises(ValueError):
        chronological_split(hourly_data, config)
