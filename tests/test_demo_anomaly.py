"""Anomali demo servislerinin protokol ve zaman hizası regresyon testleri."""

import numpy as np
import pandas as pd
import pytest

from app.anomaly_services import (
    METHODS,
    _analysis_history,
    _event_table,
    analyze_test_range,
    anomaly_demo_paths,
    checkpoint_signature,
    load_anomaly_calibration,
)
from app.pages.anomaly import _make_figure
from app.services import load_f13_models, load_household_data, valid_test_days
from src.data.inject_anomalies import inject_anomalies


@pytest.fixture(scope="module")
def anomaly_demo():
    """Yerel Household test tablosunu ve train checkpoint'lerini yükler; eğitim yapmaz."""
    config, data_path, models_dir, tables_dir, features = anomaly_demo_paths()
    frame = load_household_data(str(data_path))
    calibration = load_anomaly_calibration(str(tables_dir))
    models = load_f13_models(str(models_dir), features)
    return config, frame, calibration, models


def test_anomaly_calibration_uses_saved_validation_thresholds_and_oof_tables(anomaly_demo):
    """Seçili eşik ve artık ölçeklerinin kayıtlı validation/OOF dosyalarından geldiğini denetler."""
    config, _, calibration, _ = anomaly_demo
    selection = pd.read_csv("results/tables/anomaly_protocol_selection.csv").set_index("method")
    assert calibration["thresholds"] == pytest.approx(
        selection.loc[list(METHODS), "threshold"].to_dict()
    )
    assert calibration["windows"] == selection.loc[list(METHODS), "smooth"].astype(int).to_dict()
    assert selection.loc["A3-s2s-norm", "final"]
    assert set(calibration["std_tables"]) == {"A3-s2s", "A3-ens"}
    for table in calibration["std_tables"].values():
        assert len(table) == 168
        assert set(map(tuple, table[["hour", "dow"]].to_numpy())) == {
            (hour, dow) for hour in range(24) for dow in range(7)
        }
    assert config["protocol"]["std_source"] == "oof"


def test_analysis_history_preserves_microsecond_index_and_stops_before_test():
    """Saat çözünürlüğünü ve yalnız geçmiş bağlamını doğrular."""
    index = pd.date_range("2024-01-01", periods=300, freq="h").as_unit("us")
    frame = pd.DataFrame({"value": np.arange(len(index))}, index=index)
    start = index[240]

    history = _analysis_history(frame, start)

    assert history.index.unit == "us"
    assert history.index[-1] == start - pd.Timedelta(hours=1)
    assert len(history) == 192
    assert (history.index < start).all()


def test_selected_type_injection_is_reproducible_and_keeps_source_unchanged(anomaly_demo):
    """Tür filtreli enjeksiyonun deterministik ve kopya üzerinde olduğunu doğrular."""
    config, _, _, _ = anomaly_demo
    index = pd.date_range("2024-01-01", periods=250 * 24, freq="h")
    frame = pd.DataFrame(
        {
            "Global_active_power": 0.8 + (index.hour >= 7).astype(float),
            "is_imputed": np.zeros(len(index), dtype="int8"),
        },
        index=index,
    )
    before = frame.copy(deep=True)

    first, labels = inject_anomalies(
        frame,
        "Global_active_power",
        config["injection"],
        seed=17,
        anomaly_types=("spike",),
    )
    second, repeated_labels = inject_anomalies(
        frame,
        "Global_active_power",
        config["injection"],
        seed=17,
        anomaly_types=("spike",),
    )

    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(labels, repeated_labels)
    assert labels["is_anomaly"].any()
    assert set(labels.loc[labels["is_anomaly"] == 1, "anomaly_type"]) == {"spike"}
    with pytest.raises(ValueError, match="anomaly_types"):
        inject_anomalies(
            frame,
            "Global_active_power",
            config["injection"],
            seed=17,
            anomaly_types=("unknown",),
        )


def test_detected_events_keep_inclusive_hourly_bounds_and_split_on_invalid_hours():
    """Olay bloklarının saat sınırlarını ve geçersiz saatte bölünmesini doğrular."""
    index = pd.date_range("2024-01-01", periods=7, freq="h").as_unit("us")
    scores = pd.DataFrame(
        {
            "A0": [0.0, 4.0, 5.0, 0.0, 6.0, 7.0, 8.0],
            "A3-s2s-norm": [0.0] * 7,
            "A5-mean-z": [0.0] * 7,
        },
        index=index,
    )
    labels = pd.DataFrame(
        {"is_anomaly": [0, 1, 1, 0, 1, 1, 1], "anomaly_type": ["none", "spike", "spike", "none", "spike", "spike", "spike"]},
        index=index,
    )
    valid = np.array([True, True, True, True, True, False, True])

    events = _event_table(scores, {"A0": 3.0, "A3-s2s-norm": 1.0, "A5-mean-z": 1.0}, valid, labels)
    a0_events = events.loc[events["Yöntem"] == "A0"].reset_index(drop=True)

    assert len(a0_events) == 3
    assert a0_events.loc[0, "Başlangıç"] == index[1]
    assert a0_events.loc[0, "Bitiş"] == index[2]
    assert a0_events.loc[1, "Başlangıç"] == index[4]
    assert a0_events.loc[1, "Bitiş"] == index[4]
    assert a0_events.loc[2, "Başlangıç"] == index[6]
    assert a0_events.loc[2, "Bitiş"] == index[6]


def test_real_test_scoring_returns_aligned_scores_with_validation_thresholds(anomaly_demo):
    """Gerçek test verisindeki skor ve eşik hizasını modellerle doğrular."""
    config, frame, calibration, models = anomaly_demo
    day = pd.Timestamp(valid_test_days(frame, config)[0])
    start = day
    end = day + pd.Timedelta(hours=23)

    result = analyze_test_range(
        frame,
        config,
        start,
        end,
        False,
        "spike",
        "main",
        2,
        calibration,
        checkpoint_signature(anomaly_demo_paths()[2]),
        models,
    )

    expected_index = pd.date_range(start, end, freq="h").as_unit(frame.index.unit)
    assert result["scores"].index.equals(expected_index)
    assert result["real_kw"].index.equals(expected_index)
    assert result["injected_kw"].equals(result["real_kw"])
    assert result["valid"].index.equals(expected_index)
    assert result["thresholds"] == calibration["thresholds"]
    for method in METHODS:
        assert result["scores"][method].notna().all()
        assert np.isfinite(result["scores"][method]).all()


def test_real_test_scoring_injects_only_selected_type_on_a_copy(anomaly_demo):
    """Gerçek test dönemi enjeksiyonunun yalnız kopyayı etkilediğini doğrular."""
    config, frame, calibration, models = anomaly_demo
    days = pd.date_range("2010-04-01", periods=30, freq="D")
    start = days[0]
    end = days[-1] + pd.Timedelta(hours=23)
    original = frame.loc[start:end, "Global_active_power"].copy()

    result = analyze_test_range(
        frame,
        config,
        start,
        end,
        True,
        "spike",
        "main",
        2,
        calibration,
        checkpoint_signature(anomaly_demo_paths()[2]),
        models,
    )

    assert result["seed"] == 2
    assert result["scenario"] == "main"
    assert result["labels"]["is_anomaly"].any()
    assert set(result["labels"].loc[result["labels"]["is_anomaly"] == 1, "anomaly_type"]) == {
        "spike"
    }
    pd.testing.assert_series_equal(frame.loc[start:end, "Global_active_power"], original)
    changed = ~np.isclose(result["injected_kw"], result["real_kw"])
    assert changed.any()
    assert result["labels"]["is_anomaly"].astype(bool).to_numpy()[changed].all()


def test_plot_contains_consumption_scores_and_validation_thresholds():
    """Grafiğin tüketim, üç skor serisi ve eşik çizgilerini içerdiğini doğrular."""
    index = pd.date_range("2024-01-01", periods=3, freq="h")
    values = pd.Series([1.0, 2.0, 3.0], index=index)
    result = {
        "real_kw": values,
        "imputed": pd.Series([False, False, False], index=index),
        "injected_kw": pd.Series([1.0, 4.0, 3.0], index=index),
        "scores": pd.DataFrame(
            {"A0": [0.1, 0.2, 0.3], "A3-s2s-norm": [0.2, 0.3, 0.4],
             "A5-mean-z": [0.3, 0.4, 0.5]},
            index=index,
        ),
        "thresholds": {"A0": 1.0, "A3-s2s-norm": 2.0, "A5-mean-z": 3.0},
        "valid": pd.Series([True, True, True], index=index),
        "labels": pd.DataFrame(
            {"is_anomaly": [0, 1, 0], "anomaly_type": ["none", "spike", "none"]},
            index=index,
        ),
        "seed": 2,
    }

    figure = _make_figure(result, "spike")

    assert any(trace.name == "Gerçek ölçüm" for trace in figure.data)
    assert any(trace.name == "Sentetik Ani yükselme (spike)" for trace in figure.data)
    assert len(figure.layout.shapes) == 3
