"""Anomali hattı testleri: enjeksiyon, geçmişe bakan skorlar, AE yardımcıları, metrikler."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.inject_anomalies import ANOMALY_TYPES, inject_anomalies
from src.data.load import load_config
from src.evaluate_anomaly import evaluate_split
from src.models.anomaly_baselines import rolling_zscore, window_features
from src.models.autoencoder import LSTMAutoencoder, make_windows, point_scores
from src.train_autoencoder import clean_windows
from src.utils.anomaly_metrics import evaluate_threshold, event_metrics, f1_max_threshold, runs


def make_frame(days=120, seed=0):
    """Günlük desenli sentetik saatlik tablo (birkaç doldurulmuş saat ile)."""
    index = pd.date_range("2024-01-01", periods=days * 24, freq="h")
    rng = np.random.default_rng(seed)
    hours = index.hour.to_numpy()
    values = 0.5 + 1.0 * ((hours >= 7) & (hours <= 22)) + 0.1 * rng.random(len(index))
    frame = pd.DataFrame({
        "Global_active_power": values,
        "is_imputed": np.zeros(len(index), dtype="int8"),
        "hour_sin": np.sin(2 * np.pi * hours / 24),
        "hour_cos": np.cos(2 * np.pi * hours / 24),
    }, index=index)
    frame.iloc[100:110, frame.columns.get_loc("is_imputed")] = 1
    return frame


@pytest.fixture
def injection_settings():
    return load_config()["injection"]


def test_injection_is_reproducible_and_does_not_modify_original(injection_settings):
    frame = make_frame()
    before = frame.copy()

    first, labels_1 = inject_anomalies(frame, "Global_active_power", injection_settings, seed=1)
    second, labels_2 = inject_anomalies(frame, "Global_active_power", injection_settings, seed=1)
    _, labels_other = inject_anomalies(frame, "Global_active_power", injection_settings, seed=2)

    pd.testing.assert_frame_equal(frame, before)          # orijinal değişmedi
    pd.testing.assert_frame_equal(labels_1, labels_2)     # aynı seed → aynı etiket
    pd.testing.assert_frame_equal(first, second)
    assert not labels_1.equals(labels_other)              # farklı seed → farklı etiket


def test_injection_ratio_types_and_imputed_hours(injection_settings):
    frame = make_frame()
    injected, labels = inject_anomalies(frame, "Global_active_power", injection_settings, seed=1)

    ratio = labels["is_anomaly"].mean()
    assert injection_settings["target_ratio"] <= ratio <= injection_settings["max_ratio"]
    assert set(labels["anomaly_type"]) <= set(ANOMALY_TYPES) | {"none"}
    assert (labels["is_anomaly"] == 1).eq(labels["anomaly_type"] != "none").all()
    # Doldurulmuş saatlere enjeksiyon yapılmaz.
    assert labels.loc[frame["is_imputed"] == 1, "is_anomaly"].sum() == 0
    # Yalnız etiketli saatlerin değeri değişir; diğer sütunlar hiç değişmez.
    changed = ~np.isclose(injected["Global_active_power"], frame["Global_active_power"])
    assert not (changed & (labels["is_anomaly"] == 0).to_numpy()).any()
    pd.testing.assert_frame_equal(injected.drop(columns="Global_active_power"),
                                  frame.drop(columns="Global_active_power"))


def test_injected_values_follow_type_rules(injection_settings):
    frame = make_frame(days=200)
    injected, labels = inject_anomalies(frame, "Global_active_power", injection_settings, seed=3)
    original = frame["Global_active_power"]
    new = injected["Global_active_power"]
    types = labels["anomaly_type"]

    drop = types == "drop"
    if drop.any():
        assert np.allclose(new[drop], injection_settings["drop"]["value_kw"])
    spike = types == "spike"
    if spike.any():
        factor = new[spike] / original[spike]
        assert factor.between(injection_settings["spike"]["factor_low"],
                              injection_settings["spike"]["factor_high"]).all()
    shift = types == "level_shift"
    if shift.any():
        assert np.allclose(new[shift] / original[shift], injection_settings["level_shift"]["factor"])
    pattern = types == "pattern"
    if pattern.any():
        assert set(frame.index[pattern].hour) <= set(injection_settings["pattern"]["night_hours"])


def test_rolling_zscore_looks_only_at_the_past():
    frame = make_frame()
    series = frame["Global_active_power"]
    base = rolling_zscore(series, 168, 0.05)
    t = 400

    future_changed = series.copy()
    future_changed.iloc[t + 1:] += 10.0                   # gelecek değişirse
    assert rolling_zscore(future_changed, 168, 0.05).iloc[t] == pytest.approx(base.iloc[t])
    now_changed = series.copy()
    now_changed.iloc[t] += 10.0                           # saatin kendisi değişirse
    assert rolling_zscore(now_changed, 168, 0.05).iloc[t] > base.iloc[t]
    assert base.iloc[:168].isna().all()                   # 7 günlük geçmiş yoksa skor yok


def test_window_features_do_not_use_future_values():
    frame = make_frame()
    base = window_features(frame, "Global_active_power", ["hour_sin", "hour_cos"])
    changed = frame.copy()
    changed.iloc[401:, 0] += 10.0
    after = window_features(changed, "Global_active_power", ["hour_sin", "hour_cos"])
    pd.testing.assert_series_equal(base.iloc[400], after.iloc[400])


def test_autoencoder_shapes_and_point_score_averaging():
    model = LSTMAutoencoder(1, hidden=64, latent=32, dropout=0.1)
    assert model(torch.zeros((3, 24, 1))).shape == (3, 24, 1)

    windows, starts = make_windows(np.arange(5.0), window=3)
    assert windows.shape == (3, 3)
    errors = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    scores = point_scores(errors, starts, length=5, window=3)
    # Saat 2 üç pencerede: konum 2 (3), konum 1 (5), konum 0 (7) → ortalama 5.
    np.testing.assert_allclose(scores, [1.0, 3.0, 5.0, 7.0, 9.0])


def test_clean_windows_drop_imputed_and_excluded_period():
    frame = make_frame(days=20)
    all_windows = clean_windows(frame, "Global_active_power", 24, 0.0, 1.0)
    excluded = clean_windows(frame, "Global_active_power", 24, 0.0, 1.0,
                             exclude=("2024-01-10", "2024-01-11"))
    total = len(frame) - 24 + 1
    # Saat 100–109 doldurulmuş: bunları içeren 24 + 10 − 1 = 33 pencere atılır.
    assert len(all_windows) == total - 33
    assert len(excluded) < len(all_windows)


def test_f1_max_threshold_and_type_recall():
    labels = np.array([0, 0, 0, 1, 1, 0])
    scores = np.array([0.1, 0.2, 0.1, 0.9, 0.8, 0.3])
    types = np.array(["none", "none", "none", "spike", "drop", "none"])

    threshold = f1_max_threshold(labels, scores)
    result = evaluate_threshold(labels, scores, types, threshold)
    assert result["f1"] == pytest.approx(1.0)
    assert result["recall_spike"] == 1.0 and result["recall_drop"] == 1.0
    assert np.isnan(result["recall_pattern"])
    assert result["pr_auc"] == pytest.approx(1.0)


def make_run(seed, scores, labels):
    """evaluate_split'in beklediği tek kopya sözlüğü (doldurulmuş saat yok)."""
    return {"seed": seed, "scores": scores, "labels": labels,
            "imputed": np.zeros(len(labels), dtype=bool)}


def test_test_split_reuses_val_thresholds_without_selecting_new_ones():
    config = load_config()
    labels = pd.DataFrame({"is_anomaly": [0, 1, 0, 1], "anomaly_type": ["none", "spike", "none", "drop"]})
    scores = pd.DataFrame({"A0": [0.1, 0.9, 0.2, 0.4]})
    val_thresholds = pd.DataFrame({"method": ["A0"], "rule": ["f1max"], "threshold": [0.5]})

    result = evaluate_split([make_run(2, scores, labels)], "test", config, val_thresholds)
    assert result["threshold"].tolist() == [0.5]          # val eşiği aynen
    assert result.loc[0, "recall"] == pytest.approx(0.5)  # 0.4 < 0.5 kaçırıldı


def test_val_threshold_is_pooled_over_injection_seeds():
    config = load_config()
    labels = pd.DataFrame({"is_anomaly": [0, 1, 0, 0], "anomaly_type": ["none", "spike", "none", "none"]})
    run_a = make_run(1, pd.DataFrame({"A0": [0.1, 0.9, 0.2, 0.3]}), labels)
    run_b = make_run(11, pd.DataFrame({"A0": [0.1, 0.5, 0.6, 0.2]}), labels)

    result = evaluate_split([run_a, run_b], "val", config)
    f1max = result[result["rule"] == "f1max"]
    assert f1max["seed"].tolist() == [1, 11]
    assert f1max["threshold"].nunique() == 1              # her yöntemin TEK eşiği
    assert result["threshold"].nunique() == 1 + len(config["anomaly"]["threshold_percentiles"])


def test_event_metrics_count_an_event_if_any_hour_is_flagged():
    assert runs(np.array([0, 1, 1, 0, 1])) == [(1, 3), (4, 5)]
    types = np.array(["none", "level_shift", "level_shift", "level_shift", "none",
                      "drop", "drop", "none", "none", "none"])
    predicted = np.array([0, 0, 1, 0, 0, 0, 0, 0, 1, 1], dtype=bool)

    result = event_metrics(types, predicted)
    assert result["n_events"] == 2
    assert result["event_recall"] == pytest.approx(0.5)            # seviye kayması yakalandı, drop kaçtı
    assert result["event_recall_level_shift"] == 1.0 and result["event_recall_drop"] == 0.0
    assert result["event_precision"] == pytest.approx(0.5)         # 2 tahmin bloğundan 1'i doğru
    # Nokta recall'ü olay recall'ünden düşük: 5 anomali saatinden 1'i işaretli.
    point = evaluate_threshold((types != "none").astype(int), predicted.astype(float), types, 0.5)
    assert point["recall"] == pytest.approx(0.2)


# ---------------------------------------------------------------- A-2 / A-3 / A-4

from copy import deepcopy

from src.evaluate_anomaly import choose_smoothing, smooth_scores
from src.models.residual_detector import normalized_scores, one_step_predictions, residual_std_table
from src.models.seq2seq import Seq2SeqForecaster


def tiny_seq2seq_checkpoint(lookback=6):
    """Rastgele ağırlıklı küçük seq2seq ve ona uygun sahte checkpoint (ölçek = kimlik)."""
    torch.manual_seed(0)
    features = ["Global_active_power", "hour_sin", "hour_cos"]
    model = Seq2SeqForecaster(len(features), len(features) - 1, 8, 1, 0.0).eval()
    checkpoint = {
        "model_name": "seq2seq", "features": features,
        "feature_mean": [0.0] * 3, "feature_scale": [1.0] * 3,
        "target_mean": [0.0], "target_scale": [1.0],
        "forecast": {"lookback": lookback, "horizon": 24},
    }
    return model, checkpoint


def test_one_step_prediction_uses_only_hours_before_t():
    frame = make_frame(days=5)
    model, checkpoint = tiny_seq2seq_checkpoint()
    base = one_step_predictions(model, checkpoint, frame, "cpu")
    assert base.iloc[:6].isna().all() and base.iloc[6:].notna().all()

    t = 50
    changed = frame.copy()
    changed.iloc[t:, 0] += 5.0                            # t ve sonrası değişirse
    after = one_step_predictions(model, checkpoint, changed, "cpu")
    assert after.iloc[t] == pytest.approx(base.iloc[t])   # ŷ[t] değişmez (pencere t−6 … t−1)
    assert after.iloc[t + 1] != pytest.approx(base.iloc[t + 1])


def test_seq2seq_first_output_needs_only_first_decoder_step():
    # Decoder nedensel LSTM: 24 adımlı girdinin ilk çıktısı = yalnız ilk adımla çıktı.
    model, _ = tiny_seq2seq_checkpoint()
    inputs, future = torch.randn(4, 6, 3), torch.randn(4, 24, 2)
    with torch.no_grad():
        full = model(inputs, future)[:, 0]
        single = model(inputs, future[:, :1])[:, 0]
    torch.testing.assert_close(full, single)


def test_residual_std_table_and_normalized_scores():
    index = pd.date_range("2024-01-01", periods=14 * 24, freq="h")
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"y": rng.normal(size=len(index)), "is_imputed": 0}, index=index)
    # Saat 18'de artıklar 10 kat büyük.
    train.loc[train.index.hour == 18, "y"] *= 10
    table = residual_std_table(train, pd.Series(0.0, index=index), "y")
    assert len(table) == 168
    errors = pd.Series(1.0, index=index[:48])
    scores = normalized_scores(errors, table)
    assert scores[errors.index.hour == 18].mean() < scores[errors.index.hour == 3].mean() / 3


def test_smoothing_is_past_only_and_choice_prefers_smaller_window():
    scores = pd.DataFrame({"m": [0.0, 0.0, 3.0, 0.0, 0.0, 9.0]})
    smoothed = smooth_scores(scores, 3)["m"]
    assert np.isnan(smoothed.iloc[1]) and smoothed.iloc[2] == pytest.approx(1.0)
    assert smoothed.iloc[4] == pytest.approx(1.0)         # 9.0 (t=5) henüz görülmedi
    assert smooth_scores(scores, 1) is scores

    summary = pd.DataFrame({"method": ["m"] * 3, "rule": ["f1max"] * 3,
                            "smooth": [1, 3, 6], "f1_mean": [0.2, 0.3, 0.3]})
    assert choose_smoothing(summary) == {"m": 3}         # eşitlikte küçük pencere


def test_severity_change_keeps_event_positions(injection_settings):
    frame = make_frame(days=200)
    stronger = deepcopy(injection_settings)
    stronger["level_shift"]["factor"] = 3.0
    stronger["spike"]["factor_low"] = stronger["spike"]["factor_high"] = 6.0
    _, base_labels = inject_anomalies(frame, "Global_active_power", injection_settings, seed=1)
    injected, labels = inject_anomalies(frame, "Global_active_power", stronger, seed=1)
    pd.testing.assert_frame_equal(base_labels, labels)   # aynı yer ve tip, yalnız büyüklük farklı
    shift = labels["anomaly_type"] == "level_shift"
    if shift.any():
        ratio = injected["Global_active_power"][shift] / frame["Global_active_power"][shift]
        assert np.allclose(ratio, 3.0)


# ---------------------------------------------------------------- A-5

from src.evaluate_anomaly import add_combinations, paired_differences, to_common_scale
from src.models.residual_detector import multi_step_predictions


def test_multi_step_h1_matches_one_step_and_horizon_h_uses_history_up_to_t_minus_h():
    frame = make_frame(days=5)
    model, checkpoint = tiny_seq2seq_checkpoint()
    multi = multi_step_predictions(model, checkpoint, frame, "cpu")
    one = one_step_predictions(model, checkpoint, frame, "cpu")
    np.testing.assert_allclose(multi[1].to_numpy(), one.to_numpy(), rtol=1e-5, atol=1e-6)
    assert multi[24].iloc[:6 + 23].isna().all() and multi[24].iloc[6 + 23:].notna().all()

    t, h = 80, 12
    changed = frame.copy()
    changed.iloc[t - h + 1:, 0] += 5.0                     # t−h+1 ve sonrası değişirse
    after = multi_step_predictions(model, checkpoint, changed, "cpu")
    assert after.loc[frame.index[t], h] == pytest.approx(multi.loc[frame.index[t], h])
    assert after.loc[frame.index[t], h - 1] != pytest.approx(multi.loc[frame.index[t], h - 1])


def test_combination_scales_are_comparable_and_rank_is_ecdf():
    reference = np.sort(np.arange(1.0, 101.0))             # 1 … 100
    values = pd.Series([0.0, 50.0, 100.0, np.nan])
    ranks = to_common_scale(values, reference, "rank")
    np.testing.assert_allclose(ranks.iloc[:3], [0.0, 0.5, 1.0])
    assert np.isnan(ranks.iloc[3])

    scores = pd.DataFrame({"a": [1.0, 100.0], "b": [10.0, 1000.0]})
    combination = {"components": ["a", "b"],
                   "references": {"a": reference, "b": reference * 10}}
    add_combinations(scores, combination)
    # Aynı göreli konumdaki skorlar ortak ölçekte eşit olur → ortalama = maksimum.
    np.testing.assert_allclose(scores["A5-mean-z"], scores["A5-max-z"])
    np.testing.assert_allclose(scores["A5-mean-rank"], [0.01, 1.0])


def test_paired_differences_per_seed():
    per_seed = pd.DataFrame({
        "method": ["A0"] * 3 + ["X"] * 3, "seed": [1, 2, 3] * 2, "rule": "f1max",
        "f1": [0.1, 0.2, 0.3, 0.2, 0.2, 0.5], "pr_auc": 0.1, "event_recall": 0.5,
    })
    row = paired_differences(per_seed, "A0").iloc[0]
    assert row["f1_diff_mean"] == pytest.approx(0.1) and row["f1_n_positive"] == 2
