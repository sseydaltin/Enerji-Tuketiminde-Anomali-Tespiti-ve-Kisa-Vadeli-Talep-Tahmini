"""Anomali demo servislerinin protokol ve zaman hizası regresyon testleri."""

import numpy as np
import pandas as pd
import pytest

from app import anomaly_services
from app.anomaly_services import (
    CAUGHT,
    METHODS,
    MISSED,
    _analysis_history,
    _event_table,
    analyze_test_range,
    anomaly_demo_paths,
    checkpoint_signature,
    complete_test_days,
    event_outcomes,
    load_anomaly_calibration,
    scale_injection,
)
from app.pages.anomaly import _make_figure
from src.utils.anomaly_metrics import event_metrics
from app.services import load_f13_models, load_household_data, valid_test_days
from app.ui import METHOD_LABELS
from src.anomaly_protocol import scenario_settings
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


def _small_result():
    """Grafik ve olay sınıflaması testleri için küçük, elle kurulmuş analiz çıktısı."""
    index = pd.date_range("2024-01-01", periods=6, freq="h")
    return {
        "real_kw": pd.Series([1.0, 2.0, 3.0, 1.0, 2.0, 3.0], index=index),
        "imputed": pd.Series([False] * 6, index=index),
        "injected_kw": pd.Series([1.0, 4.0, 3.0, 1.0, 2.0, 3.0], index=index),
        # Eşik 2: saat 1 enjekte olayı yakalar, saat 5 yanlış alarmdır, saat 3 olayı kaçar.
        "scores": pd.DataFrame(
            {"A0": [0.0] * 6, "A3-s2s-norm": [0.1, 2.5, 0.1, 0.1, 0.1, 3.0], "A5-mean-z": [0.0] * 6},
            index=index,
        ),
        # Ham skor saat 3'te eşiği geçer ama eşik yumuşatılmış skora uygulanır.
        "raw_scores": pd.DataFrame(
            {"A0": [0.0] * 6, "A3-s2s-norm": [0.1, 4.0, 0.1, 5.9, 0.1, 3.0], "A5-mean-z": [0.0] * 6},
            index=index,
        ),
        "windows": {"A0": 3, "A3-s2s-norm": 3, "A5-mean-z": 3},
        "thresholds": {"A0": 1.0, "A3-s2s-norm": 2.0, "A5-mean-z": 3.0},
        "valid": pd.Series([True] * 6, index=index),
        "labels": pd.DataFrame(
            {
                "is_anomaly": [0, 1, 0, 1, 0, 0],
                "anomaly_type": ["none", "spike", "none", "spike", "none", "none"],
            },
            index=index,
        ),
        "seed": 2,
    }


def test_plot_contains_consumption_score_threshold_bands_and_flagged_hours():
    """Grafik tüketimi, seçili skoru, eşiği, kırmızı bantları ve alarm noktalarını içerir."""
    figure = _make_figure(_small_result(), "spike", "A3-s2s-norm")

    names = {trace.name for trace in figure.data}
    assert {"Gerçek ölçüm", "Sentetik Ani yükselme", "Enjekte edilen anomali", "İşaretlenen saat"} <= names
    # 2 olay × 2 satır bant + 1 eşik çizgisi.
    assert len(figure.layout.shapes) == 5
    flagged = next(trace for trace in figure.data if trace.name == "İşaretlenen saat")
    assert len(flagged.x) == 2


def test_event_outcomes_match_project_event_metrics():
    """Sayfadaki yakalandı/kaçırıldı ve yanlış alarm sayıları proje metrikleriyle aynı olmalı."""
    result = _small_result()
    table, false_alarms = event_outcomes(result, "A3-s2s-norm")

    assert table["Durum"].tolist() == [CAUGHT, MISSED]
    # Ham skoru 5,9 olan olay kaçırılmış olabilir: eşik yumuşatılmış skora uygulanır.
    assert table["Ham skor (en yüksek)"].tolist() == [4.0, 5.9]
    assert table["Yumuşatılmış skor (en yüksek)"].tolist() == [2.5, 0.1]
    assert false_alarms == 1

    flagged = (result["valid"] & result["scores"]["A3-s2s-norm"].ge(2.0)).to_numpy()
    metrics = event_metrics(result["labels"]["anomaly_type"].to_numpy(), flagged)
    assert metrics["event_recall"] == pytest.approx((table["Durum"] == CAUGHT).mean())
    assert metrics["event_precision"] == pytest.approx(1 - false_alarms / metrics["n_pred_blocks"])


def test_severity_one_keeps_protocol_and_other_values_scale_only_magnitude(anomaly_demo):
    """Şiddet ×1 protokolü aynen bırakmalı; başka değerler yalnız büyüklüğü değiştirmeli."""
    config, _, _, _ = anomaly_demo
    settings = scenario_settings(config, "main")
    assert scale_injection(settings, 1.0) == settings

    doubled = scale_injection(settings, 2.0)
    assert doubled["spike"]["sigma_low"] == settings["spike"]["sigma_low"] * 2
    assert doubled["level_shift"]["factor_low"] == 1 + (settings["level_shift"]["factor_low"] - 1) * 2
    assert doubled["drop"] == settings["drop"]
    assert doubled["pattern"] == settings["pattern"]
    assert doubled["spike"]["max_len"] == settings["spike"]["max_len"]


@pytest.mark.parametrize("scenario", ["main", "hard"])
def test_every_type_and_seed_injects_without_error(anomaly_demo, scenario):
    """Tüm test kopyasına enjeksiyon, her tip ve seed için hatasız çalışmalı.

    Kısa aralığa enjeksiyon seviye kaymasında 'hedef orana ulaşamadı' hatası veriyordu.
    """
    config, frame, calibration, models = anomaly_demo
    days = complete_test_days(frame, config)
    start = pd.Timestamp(days[0])
    end = start + pd.Timedelta(days=30) - pd.Timedelta(hours=1)
    signature = checkpoint_signature(anomaly_demo_paths()[2])
    for seed in config["protocol"]["scenarios"][scenario]["seeds"]["test"]:
        for kind in ("spike", "drop", "level_shift", "pattern"):
            result = analyze_test_range(
                frame, config, start, end, True, kind, scenario, seed, calibration, signature, models,
            )
            assert result["scores"].index[0] == start and result["scores"].index[-1] == end


def test_clean_test_alarms_match_final_doc(anomaly_demo):
    """Enjeksiyonsuz tüm testte final yöntem belgedeki 8 alarmı ve 6,21 skorunu vermeli."""
    config, frame, calibration, models = anomaly_demo
    days = complete_test_days(frame, config)
    result = analyze_test_range(
        frame, config, pd.Timestamp(days[0]), pd.Timestamp(days[-1]) + pd.Timedelta(hours=23),
        False, "spike", "main", 2, calibration, checkpoint_signature(anomaly_demo_paths()[2]), models,
    )
    alarms = result["events"].loc[result["events"]["Yöntem"] == "A3-s2s-norm"]
    saved = pd.read_csv(anomaly_demo_paths()[3] / "anomaly_test_top_events_clean.csv")
    assert len(alarms) == 8 == len(saved)
    assert alarms["En yüksek skor"].max() == pytest.approx(saved["max_score"].max(), abs=1e-2)
    assert alarms["En yüksek skor"].max() == pytest.approx(6.21, abs=5e-3)
    assert set(alarms["Başlangıç"].astype(str)) == set(saved["start"].astype(str))


def test_demo_uses_oof_std_not_stale_in_sample_table(anomaly_demo):
    """Demo ölçeği eski in-sample tablodan farklı olmalı; eşikler OOF ile seçildi."""
    _, _, calibration, _ = anomaly_demo
    stale = pd.read_csv("results/tables/A3-s2s_train_residual_std.csv")["std"].to_numpy()
    assert not np.allclose(calibration["std_tables"]["A3-s2s"]["std"].to_numpy(), stale)


def test_demo_scoring_reproduces_saved_final_seed_results(anomaly_demo, monkeypatch):
    """Demo puanlaması, protokolün karışık enjeksiyonuyla kayıtlı final olay F1'ini vermeli.

    Demo tek tip enjekte eder; burada tip filtresi kaldırılarak protokolle aynı dört
    tipli kopya kurulur. Böylece ölçek, referans ve yumuşatmanın final ile aynı olduğu gösterilir.
    """
    config, frame, calibration, models = anomaly_demo
    saved = pd.read_csv("results/tables/anomaly_protocol_test_per_seed.csv")
    saved = saved[(saved["scenario"] == "main") & saved["method"].isin(METHODS)]
    signature = checkpoint_signature(anomaly_demo_paths()[2])
    original = anomaly_services.inject_anomalies

    def inject_all_types(*args, anomaly_types=None, **kwargs):
        """Tip filtresini yok sayar: dört tip birlikte, protokoldeki gibi."""
        return original(*args, **kwargs)

    monkeypatch.setattr(anomaly_services, "inject_anomalies", inject_all_types)
    for seed in config["protocol"]["scenarios"]["main"]["seeds"]["test"]:
        # __wrapped__: önbelleği atla, tek tipli kayıtlarla karışmasın.
        scored = anomaly_services._score_whole_test.__wrapped__(
            frame, config, True, "spike", "main", seed, 1.0, calibration, signature, models,
        )
        types = scored["labels"]["anomaly_type"].to_numpy()
        clean = scored["injected"]["is_imputed"].to_numpy() == 0
        for method in METHODS:
            smoothed = scored["smoothed"][method]
            flagged = (smoothed >= calibration["thresholds"][method]).to_numpy() & clean
            expected = saved[(saved["seed"] == seed) & (saved["method"] == method)]["event_f1"].iloc[0]
            assert event_metrics(types, flagged)["event_f1"] == pytest.approx(expected, abs=1e-6)


def test_experimental_method_runs_only_on_val_hold(anomaly_demo):
    """Deneysel yöntem test sonuçlarında bulunmamalı; val-hold'da kayıtlı kuralı uygulamalı."""
    config, frame, calibration, models = anomaly_demo
    signature = checkpoint_signature(anomaly_demo_paths()[2])
    days = complete_test_days(frame, config)
    on_test = analyze_test_range(
        frame, config, pd.Timestamp(days[0]), pd.Timestamp(days[0]) + pd.Timedelta(hours=23),
        False, "spike", "main", 2, calibration, signature, models,
    )
    assert not set(anomaly_services.EXPERIMENTAL_METHODS) & set(on_test["scores"].columns)
    assert on_test["experimental"] is None

    hold_days = complete_test_days(frame, config, anomaly_services.EXPERIMENT_SPLIT)
    assert pd.Timestamp(hold_days[-1]) < pd.Timestamp(config["split"]["test"]["start"])
    start = pd.Timestamp(hold_days[0])
    end = pd.Timestamp(hold_days[-1]) + pd.Timedelta(hours=23)
    seed = config["protocol"]["scenarios"]["main"]["seeds"]["val_hold"][0]
    result = analyze_test_range(
        frame, config, start, end, True, "drop", "main", seed, calibration, signature, models,
        1.0, anomaly_services.EXPERIMENT_SPLIT,
    )
    assert result["experimental"] is not None
    table, _ = event_outcomes(result, "A6-dual-floor")
    # Enjekte düşüş 0,05 kW, taban ~0,155 kW: kural bu olayları yakalamalı (2 saat ve üstü sürer).
    assert result["experimental"]["A6-dual-floor"]["k"] is not None
    assert (table["Durum"] == CAUGHT).all()


def test_both_experimental_options_have_requested_labels_and_saved_settings(anomaly_demo):
    """İki seçenek de etiketlenmeli; ayarlar kayıtlı seçim tablosundan aynen gelmeli."""
    _, _, calibration, _ = anomaly_demo
    assert METHOD_LABELS["A6-dual"] == "Geliştirilmiş: kısa sıçramalar (deneysel)"
    assert METHOD_LABELS["A6-dual-floor"] == "Geliştirilmiş: sıçrama + taban yük (deneysel)"
    saved = pd.read_csv("results/tables/a6_selection.csv").set_index("method")
    settings = calibration["experimental"]
    assert set(settings) == set(anomaly_services.EXPERIMENTAL_METHODS)
    assert settings["A6-dual"]["t2"] == pytest.approx(saved.loc["A6-dual", "t2"])
    assert settings["A6-dual"]["k"] is None
    assert settings["A6-dual-floor"]["t2"] == pytest.approx(saved.loc["A6-dual-floor", "t2"])
    assert settings["A6-dual-floor"]["k"] == saved.loc["A6-dual-floor", "k"]


def test_demo_experimental_rules_reproduce_saved_val_hold_results(anomaly_demo, monkeypatch):
    """Demo kuralları, karışık enjeksiyonla kayıtlı val-hold olay F1'ini birebir vermeli.

    Yeni ölçüm değildir: hiçbir dosya yazılmaz, hiçbir ayar seçilmez. Yalnız demodaki iki
    seçeneğin deney kodundaki kuralla aynı olduğu gösterilir.
    """
    config, frame, calibration, models = anomaly_demo
    saved = pd.read_csv("results/tables/a6_val_hold_per_seed.csv")
    signature = checkpoint_signature(anomaly_demo_paths()[2])
    target = config["preprocess"]["target"]
    original = anomaly_services.inject_anomalies

    def inject_all_types(*args, anomaly_types=None, **kwargs):
        """Tip filtresini yok sayar: dört tip birlikte, protokoldeki gibi."""
        return original(*args, **kwargs)

    monkeypatch.setattr(anomaly_services, "inject_anomalies", inject_all_types)
    for seed in config["protocol"]["scenarios"]["main"]["seeds"][anomaly_services.EXPERIMENT_SPLIT]:
        scored = anomaly_services._score_whole_test.__wrapped__(
            frame, config, True, "spike", "main", seed, 1.0, calibration, signature, models,
            anomaly_services.EXPERIMENT_SPLIT,
        )
        types = scored["labels"]["anomaly_type"].to_numpy()
        clean = scored["injected"]["is_imputed"].to_numpy() == 0
        for method, settings in calibration["experimental"].items():
            flags = anomaly_services.experimental_flags(scored, settings, target).to_numpy() & clean
            expected = saved[(saved["seed"] == seed) & (saved["method"] == method)]["event_f1"].iloc[0]
            assert event_metrics(types, flags)["event_f1"] == pytest.approx(expected, abs=1e-6)
