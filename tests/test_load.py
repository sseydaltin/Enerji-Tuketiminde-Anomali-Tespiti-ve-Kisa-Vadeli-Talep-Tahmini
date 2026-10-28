"""Ham veri yükleme modülünün birim testleri."""

import pandas as pd
import pytest

from src.data.load import load_raw_data, summarize_raw_data


@pytest.fixture
def sample_file(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text(
        "Date;Time;Global_active_power;Global_reactive_power;"
        "Voltage;Global_intensity;Sub_metering_1;"
        "Sub_metering_2;Sub_metering_3\n"
        "16/12/2006;17:24:00;4.216;0.418;234.840;"
        "18.400;0.000;1.000;17.000\n"
        "16/12/2006;17:25:00;?;?;?;?;?;?;?\n",
        encoding="utf-8",
    )
    return path


def test_load_raw_data(sample_file):
    df = load_raw_data(file_path=sample_file)

    assert len(df) == 2
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index[0] == pd.Timestamp("2006-12-16 17:24:00")
    assert df.loc[df.index[0], "Global_active_power"] == 4.216


def test_missing_values(sample_file):
    df = load_raw_data(file_path=sample_file)

    assert df.iloc[1].isna().all()


def test_summary(sample_file):
    df = load_raw_data(file_path=sample_file)
    summary = summarize_raw_data(df)

    assert summary["row_count"] == 2
    assert summary["missing_values"]["Global_active_power"] == 1
    assert summary["missing_ratio_percent"]["Global_active_power"] == 50.0


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_data(file_path=tmp_path / "missing.txt")


def test_missing_column(tmp_path):
    path = tmp_path / "invalid.txt"
    path.write_text(
        "Date;Time;Global_active_power\n"
        "16/12/2006;17:24:00;4.216\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Eksik sütunlar"):
        load_raw_data(file_path=path)


def test_duplicate_timestamp(tmp_path, sample_file):
    lines = sample_file.read_text(encoding="utf-8").splitlines()
    path = tmp_path / "duplicate.txt"
    path.write_text(
        lines[0] + "\n" + lines[1] + "\n" + lines[1] + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Tekrarlanan"):
        load_raw_data(file_path=path)
