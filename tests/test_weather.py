"""Aşama 3b testleri: saat dilimi hizalaması, derece-saatler, W1/W2 sızıntı kontrolleri."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.data.load import load_config
from src.data.weather import add_weather_features, align_weather, local_to_utc
from src.train_forecast import prepare_forecast_data
from tests.test_eval_consistency import make_long_frame_and_config

TIMEZONE = "Europe/Paris"


def test_local_time_maps_to_utc_with_daylight_saving():
    local = pd.DatetimeIndex([
        "2008-01-15 12:00",   # kış: UTC+1
        "2008-07-15 12:00",   # yaz: UTC+2
        "2008-03-30 02:00",   # yerelde yok → 03:00 gibi (veri setinde 03:xx kopyası)
        "2008-03-30 03:00",
        "2008-10-26 02:00",   # iki kez yaşanır → yaz saati (ilk geçiş)
        "2008-10-26 03:00",
    ])
    utc = local_to_utc(local, TIMEZONE)
    expected = pd.DatetimeIndex([
        "2008-01-15 11:00", "2008-07-15 10:00", "2008-03-30 01:00",
        "2008-03-30 01:00", "2008-10-26 00:00", "2008-10-26 02:00",
    ]).tz_localize("UTC")
    assert utc.equals(expected)


def test_align_weather_reads_the_matching_utc_hour():
    utc_index = pd.date_range("2008-07-14 00:00", periods=72, freq="h", tz="UTC")
    weather = pd.DataFrame({"temperature_2m": np.arange(72.0)}, index=utc_index)
    local = pd.DatetimeIndex(["2008-07-15 12:00"])
    aligned = align_weather(local, weather, TIMEZONE)
    # Yerel 12:00 (yaz) = UTC 10:00 → 2008-07-15 10:00 UTC, başlangıçtan 34 saat sonra.
    assert aligned["temperature_2m"].iloc[0] == 34.0


def test_degree_hours_and_no_missing_after_alignment():
    config = load_config()
    settings = config["weather"]
    local = pd.date_range("2008-01-01", periods=48, freq="h")
    utc = pd.date_range("2007-12-31 00:00", periods=96, freq="h", tz="UTC")
    temperatures = np.linspace(-5, 30, len(utc))
    weather = pd.DataFrame({
        "temperature_2m": temperatures, "relative_humidity_2m": 50.0,
        "cloud_cover": 10.0, "precipitation": 0.0,
    }, index=utc)
    frame = pd.DataFrame({"target": 1.0}, index=local)
    result = add_weather_features(frame, weather, settings, config["features"]["weather"])

    t = result["temperature_2m"]
    np.testing.assert_allclose(result["heating_degree_hours"], np.maximum(0, 18 - t))
    np.testing.assert_allclose(result["cooling_degree_hours"], np.maximum(0, t - 24))
    assert "temperature_2m" not in frame.columns          # orijinal tablo değişmedi


def test_missing_weather_raises_instead_of_filling():
    config = load_config()
    local = pd.date_range("2008-01-01", periods=5, freq="h")
    utc = pd.date_range("2008-01-01", periods=2, freq="h", tz="UTC")   # eksik kapsam
    weather = pd.DataFrame({column: 1.0 for column in config["weather"]["variables"]}, index=utc)
    with pytest.raises(ValueError, match="eksik"):
        add_weather_features(pd.DataFrame({"target": 1.0}, index=local), weather,
                             config["weather"], config["features"]["weather"])


def weather_frame_and_config():
    """Sentetik veri + 'weather_clock' (satır numarası) hava sütunu.

    weather_clock satırın zamanını kodlar: X'e giren her hava değeri ilk hedef
    saatinin satır numarasından küçük olmalıdır.
    """
    frame, config = make_long_frame_and_config()
    frame["weather_clock"] = np.arange(len(frame), dtype=np.float64)
    config["features"]["weather"] = ["weather_clock"]
    config["forecast"]["lookback"] = 48
    return frame, config


def unscale(prepared, values, column):
    """Train-fit scaler ile ölçeklenmiş bir öznitelik değerini geri çevirir."""
    position = prepared["features"].index(column)
    scaler = prepared["feature_scaler"]
    return values * scaler.scale_[position] + scaler.mean_[position]


def test_w1_weather_inputs_end_before_forecast_start():
    frame, config = weather_frame_and_config()
    config["features"]["default_set"] = "W1"
    prepared = prepare_forecast_data(frame, config)

    column = prepared["features"].index("weather_clock")
    clocks = unscale(prepared, prepared["X_val"][:, :, column], "weather_clock")
    first_target_rows = frame.index.get_indexer(prepared["val_times"])
    # Her pencerede en geç hava değeri tahmin başlangıcından bir saat öncesine ait.
    np.testing.assert_allclose(clocks.max(axis=1), first_target_rows - 1, atol=1e-3)
    assert prepared["future_val"] is None                 # decoder'a hava gitmez

    # Tahmin başlangıcından sonraki havayı değiştirmek W1 girdisini değiştirmez.
    changed = frame.copy()
    start = frame.index.get_loc(prepared["val_times"][0])
    changed.iloc[start:, changed.columns.get_loc("weather_clock")] += 1000.0
    after = prepare_forecast_data(changed, deepcopy(config))
    np.testing.assert_array_equal(after["X_val"][0], prepared["X_val"][0])


def test_weather_scaler_is_fit_on_train_only():
    frame, config = weather_frame_and_config()
    config["features"]["default_set"] = "W1"
    prepared = prepare_forecast_data(frame, config)
    train = frame.loc[config["split"]["train"]["start"]:f"{config['split']['train']['end']} 23:00"]
    position = prepared["features"].index("weather_clock")
    assert prepared["feature_scaler"].mean_[position] == pytest.approx(train["weather_clock"].mean())


def test_w2_oracle_is_a_deliberate_exception_that_sees_target_hour_weather():
    # BİLİNÇLİ İSTİSNA (W2 – oracle / üst sınır): decoder'a hedef saatlerin GERÇEK
    # havası verilir. Bu test sızıntının yalnız decoder'da ve tam olarak hedef
    # saatlerde olduğunu doğrular; encoder girdisi W1 ile aynıdır.
    frame, config = weather_frame_and_config()
    config["features"]["default_set"] = "W2"
    config["forecast"]["decoder_calendar"] = True        # seq2seq'te fit_forecaster açar
    config["forecast"]["decoder_weather"] = True
    oracle = prepare_forecast_data(frame, config)

    w1_config = deepcopy(config)
    w1_config["features"]["default_set"] = "W1"
    w1_config["forecast"]["decoder_weather"] = False
    w1 = prepare_forecast_data(frame, w1_config)
    np.testing.assert_array_equal(oracle["X_val"], w1["X_val"])

    calendar_size = len(config["features"]["calendar"])
    decoder_weather = unscale(oracle, oracle["future_val"][:, :, calendar_size], "weather_clock")
    target_rows = frame.index.get_indexer(oracle["val_times"])[:, None] + np.arange(24)
    np.testing.assert_allclose(decoder_weather, target_rows, atol=1e-3)   # hedef saatin kendisi


def test_w2_requires_seq2seq_decoder():
    frame, config = weather_frame_and_config()
    config["forecast"]["decoder_weather"] = True
    with pytest.raises(ValueError, match="seq2seq"):
        prepare_forecast_data(frame, config)
