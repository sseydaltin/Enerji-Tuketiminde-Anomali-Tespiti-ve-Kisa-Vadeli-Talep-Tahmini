"""A0–A3 anomali dedektörlerini enjekte edilmiş val (varsayılan) veya test kopyalarında değerlendirir.

Protokol (A-1): val'e 5 farklı seed ile enjeksiyon yapılır; eşik her yöntem için
5 kopyanın BİRLEŞİMİNDE F1-max ile seçilir; metrikler kopya başına hesaplanıp
ortalama ± std raporlanır. Test tek kopyadır (seed=2) ve val eşiklerini kullanır.
A-2: A3 tahmin artığı skorları. A-3: yalnız geçmişe bakan skor yumuşatma
(pencere val'de seçilir). A-4: şiddet taraması (ayrı tablo, yalnız val).
A-5a: çok ufuklu (h=1…24) normalize artık skoru. A-5b: A0 ile en iyi A3
varyantının ortak ölçekte (z veya rank) ortalaması / maksimumu.
"""

import argparse
from copy import deepcopy
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, roc_curve

from src.data.inject_anomalies import ANOMALY_TYPES, inject_anomalies
from src.data.load import PROJECT_ROOT, load_config
from src.data.split import chronological_split
from src.models.anomaly_baselines import (
    fit_isolation_forest,
    isolation_forest_scores,
    rolling_zscore,
    window_features,
)
from src.models.autoencoder import LSTMAutoencoder, make_windows, point_scores, reconstruct
from src.models.residual_detector import (
    combine_predictions,
    load_forecaster,
    multi_step_predictions,
    normalized_scores,
    one_step_predictions,
    out_of_fold_predictions,
    residual_std_table,
)
from src.utils.anomaly_metrics import evaluate_threshold, f1_max_threshold
from src.utils.console import setup_console
from src.utils.results_io import resolve_path, upsert_csv
from src.utils.seed import set_seed

METHOD_LABELS = {
    "A0": "Hareketli z-skoru",
    "A1": "Isolation Forest",
    "A2": "LSTM-AE (ölçekli kare hata)",
    "A2-rel": "LSTM-AE (göreli hata)",
    "A2-noAug": "LSTM-AE, Ağu-2008 çıkarılmış (ölçekli)",
    "A2-noAug-rel": "LSTM-AE, Ağu-2008 çıkarılmış (göreli)",
    "A2-lat8": "LSTM-AE, gizil 8 – tanılama (ölçekli)",
    "A2-lat8-rel": "LSTM-AE, gizil 8 – tanılama (göreli)",
    "A3-s2s": "Tahmin artığı, seq2seq F10-opt (|hata| kW)",
    "A3-s2s-norm": "Tahmin artığı, seq2seq F10-opt (|hata| / std saat×gün)",
    "A3-ens": "Tahmin artığı, topluluk F13-ens (|hata| kW)",
    "A3-ens-norm": "Tahmin artığı, topluluk F13-ens (|hata| / std saat×gün)",
    "A3-s2s-h4": "Çok ufuklu artık F10-opt, h∈{1,6,12,24} normalize ortalama",
    "A3-s2s-hmean": "Çok ufuklu artık F10-opt, h=1…24 normalize ortalama",
    "A3-s2s-hmax": "Çok ufuklu artık F10-opt, h=1…24 normalize maksimum",
    "A5-mean-z": "Birleşim A0 + en iyi A3: z ölçeğinde ortalama",
    "A5-max-z": "Birleşim A0 + en iyi A3: z ölçeğinde maksimum",
    "A5-mean-rank": "Birleşim A0 + en iyi A3: rank ölçeğinde ortalama",
    "A5-max-rank": "Birleşim A0 + en iyi A3: rank ölçeğinde maksimum",
}
MULTI_HORIZON_METHODS = ("A3-s2s-h4", "A3-s2s-hmean", "A3-s2s-hmax")
COMBINATION_METHODS = ("A5-mean-z", "A5-max-z", "A5-mean-rank", "A5-max-rank")
# Değerlendirilecek AE checkpoint'leri (models/{exp_id}_best.pt; yoksa atlanır).
AUTOENCODER_EXPERIMENTS = ("A2", "A2-noAug", "A2-lat8")
# Figürlerde gösterilen ana yöntemler (tanılama koşuları tablolarda kalır).
FIGURE_METHODS = ("A0", "A1", "A2", "A2-rel", "A3-s2s", "A3-s2s-norm", "A3-ens", "A3-ens-norm")
# A-5 figürleri: referanslar + çok ufuklu varyantlar + birleşimler.
FIGURE_METHODS_A5 = ("A0", "A3-ens-norm") + MULTI_HORIZON_METHODS + COMBINATION_METHODS
# Ortalama ± std'si raporlanan metrikler.
SUMMARY_METRICS = (
    ["precision", "recall", "f1", "roc_auc", "pr_auc",
     "event_recall", "event_precision", "event_f1"]
    + [f"recall_{kind}" for kind in ANOMALY_TYPES]
    + [f"event_recall_{kind}" for kind in ANOMALY_TYPES]
)


def injection_seeds(config, split):
    """Val için 5 enjeksiyon seed'i, test için tek seed (2) döndürür."""
    if split == "val":
        return list(config["injection"]["val_seeds"])
    return [config["injection"]["test_seed"]]


def autoencoder_scores(checkpoint_path, series, window, relative_eps, device):
    """Kayıtlı LSTM-AE ile her saat için iki skor üretir.

    Girdi: checkpoint yolu, saatlik kW serisi (geçmiş bağlam + değerlendirilen
    küme), pencere (24), göreli skorun payda alt sınırı (kW).
    Çıktı: (ölçekli kare hata skoru, göreli skor) — ikisi de seriyle aynı indeksli.
    Neden: Mutlak hata tüketim düzeyiyle büyür (EDA: val kış, test yaz ağırlıklı).
    Göreli varyant hatayı penceredeki ortalama düzeye böler:
    |hata (kW)| / (pencere ortalaması (kW) + eps).
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    settings = checkpoint["settings"]
    model = LSTMAutoencoder(1, settings["hidden"], settings["latent"], settings["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    mean, scale = checkpoint["target_mean"], checkpoint["target_scale"]

    values_kw = series.to_numpy(dtype=np.float64)
    windows_kw, starts = make_windows(values_kw, window)
    windows_scaled = ((windows_kw - mean) / scale).astype(np.float32)
    rebuilt_scaled = reconstruct(model, windows_scaled, 512, device)
    squared = (rebuilt_scaled - windows_scaled) ** 2                  # [N, 24] ölçekli
    error_kw = np.abs(rebuilt_scaled - windows_scaled) * scale        # [N, 24] kW
    level_kw = windows_kw.mean(axis=1, keepdims=True)                 # [N, 1]
    relative = error_kw / (level_kw + relative_eps)
    length = len(series)
    return (
        pd.Series(point_scores(squared, starts, length, window), index=series.index),
        pd.Series(point_scores(relative, starts, length, window), index=series.index),
    )


def fit_detectors(splits, config, device, std_source="in_sample"):
    """Train'e bağlı her şeyi bir kez hazırlar: Isolation Forest, tahmin modelleri, artık std'leri.

    Girdi: std_source = "in_sample" (A-1–A-5: tüm train ile eğitilmiş modelin
    train artıkları) veya "oof" (final protokol: CV katlarının kat dışı artıkları).
    Çıktı: {"forest", "forecasters": {exp_id: (model, checkpoint)}, "std_tables",
    "multi_std", "in_sample_std_tables"} — skorlar seçilen kaynağın tablolarını kullanır.
    Neden: A1 yalnız train'e fit edilir; A3'ün tahmin modelleri train'de eğitilmiş
    kayıtlı ağırlıklardır (OOF'ta da skor bu ağırlıklarla hesaplanır; OOF modelleri
    yalnız std için). Hiçbiri val/test enjeksiyonunu görmez.
    """
    target = config["preprocess"]["target"]
    anomaly = config["anomaly"]
    forest = fit_isolation_forest(
        window_features(splits["train"], target, config["features"]["calendar"]),
        anomaly["isolation_forest"], config["seed"],
    )
    models_dir = resolve_path(PROJECT_ROOT, config["paths"]["models_dir"])
    residual = anomaly["residual"]
    names = {residual["seq2seq"], *residual["ensemble"]}
    forecasters = {name: load_forecaster(models_dir / f"{name}_best.pt", device) for name in names}

    train = splits["train"]
    train_predictions = {name: one_step_predictions(model, checkpoint, train, device)
                         for name, (model, checkpoint) in forecasters.items()}
    std_tables = {
        "A3-s2s": residual_std_table(train, train_predictions[residual["seq2seq"]], target),
        "A3-ens": residual_std_table(
            train, combine_predictions(train_predictions, residual["ensemble"]), target
        ),
    }
    in_sample_std_tables = std_tables
    oof = None
    if std_source == "oof":
        # Kat checkpoint'leri: models/{ana model}-oof{k}_best.pt (yalnız std kalibrasyonu).
        paths = {name: [models_dir / f"{name}-oof{k}_best.pt"
                        for k in range(1, residual["oof_folds"] + 1)] for name in names}
        oof = {name: out_of_fold_predictions(train, paths[name], device) for name in names}
        std_tables = {
            "A3-s2s": residual_std_table(train, oof[residual["seq2seq"]], target),
            "A3-ens": residual_std_table(train, combine_predictions(oof, residual["ensemble"]), target),
        }
        train_multi = out_of_fold_predictions(train, paths[residual["seq2seq"]], device, multi=True)
    else:
        # A-5a: her ufuk h için ayrı std tablosu (yalnız train artıkları).
        model, checkpoint = forecasters[residual["seq2seq"]]
        train_multi = multi_step_predictions(model, checkpoint, train, device)
    multi_std = {h: residual_std_table(train, train_multi[h], target) for h in train_multi.columns}
    return {"forest": forest, "forecasters": forecasters, "std_tables": std_tables,
            "multi_std": multi_std, "in_sample_std_tables": in_sample_std_tables,
            "train_predictions": train_predictions, "oof_predictions": oof}


def residual_scores(full, config, fitted, device):
    """A3 skorları: h=1 tahminin mutlak hatası (kW) ve saat×gün std'siyle normalize hâli."""
    target = config["preprocess"]["target"]
    residual = config["anomaly"]["residual"]
    predictions = {name: one_step_predictions(model, checkpoint, full, device)
                   for name, (model, checkpoint) in fitted["forecasters"].items()}
    variants = {
        "A3-s2s": predictions[residual["seq2seq"]],
        "A3-ens": combine_predictions(predictions, residual["ensemble"]),
    }
    scores = {}
    for name, prediction in variants.items():
        error = (full[target] - prediction).abs()
        scores[name] = error
        scores[f"{name}-norm"] = normalized_scores(error, fitted["std_tables"][name])
    scores.update(multi_horizon_scores(full, config, fitted, device))
    return scores


def multi_horizon_scores(full, config, fitted, device):
    """A-5a: her ufuk için normalize hata, sonra ufuklar üzerinden özet skorlar.

    Çıktı: {"A3-s2s-h4": h∈{1,6,12,24} ortalaması, "A3-s2s-hmean": 24 ufkun
    ortalaması, "A3-s2s-hmax": 24 ufkun maksimumu}.
    Neden: h=1 tahmini anomali girdi penceresine girince yeni düzeyi izler;
    uzun ufuklu tahminler anomali öncesi geçmişe dayandığı için seviye kaymasında
    daha uzun süre büyük hata verir. Her ufuk kendi train std'siyle normalize
    edilir (uzun ufukta olağan hata daha büyüktür). Bir ufuk eksikse skor NaN.
    """
    target = config["preprocess"]["target"]
    residual = config["anomaly"]["residual"]
    model, checkpoint = fitted["forecasters"][residual["seq2seq"]]
    predictions = multi_step_predictions(model, checkpoint, full, device)   # [T, 24]
    normalized = pd.DataFrame({
        h: normalized_scores((full[target] - predictions[h]).abs(), fitted["multi_std"][h])
        for h in predictions.columns
    })
    subset = residual["multi_horizon_subset"]
    return {
        "A3-s2s-h4": normalized[subset].mean(axis=1, skipna=False),
        "A3-s2s-hmean": normalized.mean(axis=1, skipna=False),
        "A3-s2s-hmax": normalized.max(axis=1, skipna=False),
    }


# Her değerlendirme kümesinin geçmiş bağlamını veren (ENJEKSİYONSUZ) önceki küme.
PREVIOUS_SPLIT = {"val": "train", "val_dev": "train", "val_hold": "val_dev", "test": "val"}


def with_context(splits, split, frame, config):
    """Değerlendirilen kopyanın önüne bir önceki kümenin son 8 gününü (ENJEKSİYONSUZ) ekler."""
    anomaly = config["anomaly"]
    previous = splits[PREVIOUS_SPLIT[split]]
    context_hours = anomaly["zscore_window"] + anomaly["ae"]["window"]
    return pd.concat((previous.tail(context_hours), frame))


def compute_scores(splits, config, split, seed, fitted, device, injection_settings=None,
                   combination=None):
    """Seçilen kümenin bir enjekte kopyası için tüm yöntemlerin HAM skorlarını hesaplar.

    Girdi: kronolojik bölmeler, config, küme adı, enjeksiyon seed'i, fit_detectors
    çıktısı, cihaz, isteğe bağlı değiştirilmiş enjeksiyon ayarları (A-4),
    isteğe bağlı birleşim tanımı (A-5b; verilirse birleşik skorlar eklenir).
    Çıktı: (skor tablosu [bağlam + küme saatleri, yöntem], etiketler, enjekte kopya).
    Neden: Geçmiş bağlam (son 8 gün) bir önceki kümeden ENJEKSİYONSUZ alınır;
    yalnız değerlendirilen kümeye anomali eklenir. Bağlam saatleri de döndürülür
    ki geçmişe bakan yumuşatma küme başında da tam pencereyle hesaplansın.
    """
    target = config["preprocess"]["target"]
    settings = injection_settings or config["injection"]
    injected, labels = inject_anomalies(splits[split], target, settings, seed)
    scores = score_frame(with_context(splits, split, injected, config), config, fitted, device)
    if combination is not None:
        add_combinations(scores, combination)
    return scores, labels, injected


def score_frame(full, config, fitted, device):
    """Bağlamlı bir saatlik tablo için tüm yöntemlerin HAM skorları ([saat, yöntem])."""
    anomaly = config["anomaly"]
    target = config["preprocess"]["target"]
    scores = pd.DataFrame(index=full.index)
    scores["A0"] = rolling_zscore(full[target], anomaly["zscore_window"], anomaly["zscore_min_std"])
    scores["A1"] = isolation_forest_scores(
        fitted["forest"], window_features(full, target, config["features"]["calendar"])
    )
    models_dir = resolve_path(PROJECT_ROOT, config["paths"]["models_dir"])
    for exp_id in AUTOENCODER_EXPERIMENTS:
        path = models_dir / f"{exp_id}_best.pt"
        if not path.exists():
            print(f"UYARI: {path} yok; {exp_id} atlandı.")
            continue
        squared, relative = autoencoder_scores(
            path, full[target], anomaly["ae"]["window"], anomaly["relative_eps_kw"], device
        )
        scores[exp_id] = squared
        scores[f"{exp_id}-rel"] = relative
    for name, values in residual_scores(full, config, fitted, device).items():
        scores[name] = values
    return scores


def fit_combination(splits, config, fitted, device, components, reference_split="val"):
    """A-5b: birleşimin ölçek referansı = ENJEKSİYONSUZ val'deki ham skor dağılımı.

    Girdi: bileşen yöntem adları (ör. ["A0", "A3-ens-norm"]).
    Çıktı: {"components": [...], "references": {yöntem: sıralı skor dizisi}}.
    Neden: A0 (|z|) ve A3 (std katı) farklı ölçekte; doğrudan ortalama büyük
    ölçekli olanı baskın yapar. Referans val'den alınır (etiket ve enjeksiyon
    yok); test'te aynı referans aynen uygulanır. Doldurulmuş saatler hariç.
    """
    val = splits[reference_split]
    clean = score_frame(with_context(splits, reference_split, val, config), config, fitted, device)
    clean = clean.loc[val.index][val["is_imputed"].to_numpy() == 0]
    references = {m: np.sort(clean[m].dropna().to_numpy()) for m in components}
    return {"components": list(components), "references": references}


def to_common_scale(values, reference, kind):
    """Skoru ortak ölçeğe getirir.

    z: (s − ort) / std (referansın); rank: referansta s'den küçük-eşit skorların
    oranı (0–1, ampirik dağılım fonksiyonu). NaN skorlar NaN kalır.
    """
    if kind == "z":
        return (values - reference.mean()) / reference.std(ddof=1)
    ranks = np.searchsorted(reference, values.to_numpy(), side="right") / len(reference)
    return pd.Series(ranks, index=values.index).where(values.notna())


def add_combinations(scores, combination):
    """A-5b: bileşenleri z ve rank ölçeğine getirip ortalamasını ve maksimumunu ekler (yerinde)."""
    for kind in ("z", "rank"):
        scaled = pd.DataFrame({m: to_common_scale(scores[m], combination["references"][m], kind)
                               for m in combination["components"]})
        scores[f"A5-mean-{kind}"] = scaled.mean(axis=1, skipna=False)
        scores[f"A5-max-{kind}"] = scaled.max(axis=1, skipna=False)


def smooth_scores(scores, window):
    """Yalnız geçmişe bakan hareketli ortalama: s̃[t] = ort(s[t−w+1], …, s[t]).

    Neden: Uzun anomaliler (seviye kayması) birkaç saat boyunca orta düzey skor
    verir; ortalama tek saatlik gürültüyü bastırıp süreklilik gösteren sapmayı
    öne çıkarır. Merkezli pencere kullanılmaz (gelecek bilgisi yok). w=1 ham skordur.
    """
    if window == 1:
        return scores
    return scores.rolling(window, min_periods=window).mean()


def make_runs(outputs, windows):
    """Kopya çıktılarından değerlendirme girdisi kurar; her yöntem kendi penceresiyle yumuşatılır.

    Girdi: outputs = [(seed, ham skorlar (bağlamlı), etiketler, enjekte kopya)],
    windows = {yöntem: pencere}. Çıktı: evaluate_split'in beklediği runs listesi.
    """
    runs = []
    for seed, scores, labels, injected in outputs:
        smoothed = pd.DataFrame({
            method: smooth_scores(scores[[method]], window)[method]
            for method, window in windows.items()
        })
        runs.append({"seed": seed, "scores": smoothed.loc[injected.index], "labels": labels,
                     "imputed": injected["is_imputed"].to_numpy(dtype=bool)})
    return runs


def valid_hours(run, method):
    """Değerlendirilecek saatler: skoru olan ve doldurulmamış (is_imputed=0)."""
    return run["scores"][method].notna().to_numpy() & ~run["imputed"]


def evaluate_split(runs, split, config, val_thresholds=None):
    """Her yöntem ve eşik kuralı için KOPYA BAŞINA metrikleri hesaplar.

    Girdi: runs = [{"seed", "scores", "labels", "imputed"}, ...] (val: 5, test: 1).
    val: F1-max eşiği 5 kopyanın birleşiminde etiketlerle, yüzdelik eşikler
    birleşik skor dağılımından seçilir — her yöntemin TEK eşiği vardır.
    test: eşikler val tablosundan okunur ve aynen uygulanır (test'te seçim yok).
    Çıktı: [yöntem × kural × seed] satırlı tablo.
    """
    rows = []
    percentiles = config["anomaly"]["threshold_percentiles"]
    for method in runs[0]["scores"].columns:
        if split == "val":
            pooled_y = np.concatenate([
                run["labels"]["is_anomaly"].to_numpy()[valid_hours(run, method)] for run in runs
            ])
            pooled_s = np.concatenate([
                run["scores"][method].to_numpy()[valid_hours(run, method)] for run in runs
            ])
            rules = {"f1max": f1_max_threshold(pooled_y, pooled_s)}
            for p in percentiles:
                rules[f"p{p:g}"] = float(np.percentile(pooled_s, p))
        else:
            chosen = val_thresholds[val_thresholds["method"] == method]
            rules = dict(zip(chosen["rule"], chosen["threshold"]))
        for rule, threshold in rules.items():
            for run in runs:
                metrics = evaluate_threshold(
                    run["labels"]["is_anomaly"].to_numpy(),
                    run["scores"][method].to_numpy(),
                    run["labels"]["anomaly_type"].to_numpy(),
                    threshold,
                    valid_hours(run, method),
                )
                rows.append({"split": split, "seed": run["seed"], "method": method,
                             "label": METHOD_LABELS.get(method, method), "rule": rule, **metrics})
    return pd.DataFrame(rows)


def summarize(per_seed, group_columns=("method", "rule")):
    """Kopya başına sonuçlardan ortalama ve std (ddof=1) tablosu.

    Neden: Tek enjeksiyon kopyasında ~100 olay var; olay yerleşimi şansı
    sonuçları oynatır. 5 kopyanın std'si bu belirsizliği gösterir.
    """
    rows = []
    for keys, group in per_seed.groupby(list(group_columns), sort=False):
        row = dict(zip(group_columns, keys))
        row.update({"label": group["label"].iloc[0], "threshold": group["threshold"].iloc[0],
                    "n_seeds": len(group), "n_events_mean": group["n_events"].mean()})
        for metric in SUMMARY_METRICS:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_smoothing(outputs, methods, config):
    """A-3: verilen yöntemleri her yumuşatma penceresiyle val'de değerlendirir (kopya başına)."""
    rows = []
    for window in config["anomaly"]["smoothing_windows"]:
        runs = make_runs(outputs, {method: window for method in methods})
        rows.append(evaluate_split(runs, "val", config).assign(smooth=window))
    return pd.concat(rows, ignore_index=True)


def paired_differences(per_seed, reference, metrics=("f1", "pr_auc", "event_recall"), rule="f1max"):
    """Her yöntem için (yöntem − referans) farkının seed'ler üzerinden ort ± std'si.

    Neden: Aynı seed'de iki yöntem aynı enjekte kopyayı görür; fark, kopyadan
    gelen ortak oynamayı (olayların kolay/zor yerleşimi) büyük ölçüde siler.
    n_positive = farkın pozitif olduğu seed sayısı.
    """
    best = per_seed[per_seed["rule"] == rule].set_index(["method", "seed"])
    rows = []
    for method in best.index.get_level_values("method").unique():
        if method == reference:
            continue
        row = {"reference": reference, "method": method}
        for metric in metrics:
            diff = best.loc[method, metric] - best.loc[reference, metric]
            row[f"{metric}_diff_mean"] = diff.mean()
            row[f"{metric}_diff_std"] = diff.std(ddof=1)
            row[f"{metric}_n_positive"] = int((diff > 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def choose_smoothing(smoothing_summary):
    """Her yöntem için F1-max eşiğindeki ortalama F1'i en yüksek pencereyi seçer.

    Neden: Eşik seçimiyle aynı kriter (F1) kullanılır; eşitlikte küçük pencere
    (daha az gecikme) tercih edilir. Seçim yalnız val'de yapılır.
    """
    best = smoothing_summary[smoothing_summary["rule"] == "f1max"]
    best = best.sort_values(["method", "f1_mean", "smooth"], ascending=[True, False, True])
    return best.groupby("method", sort=False)["smooth"].first().astype(int).to_dict()


def severity_sweep(splits, config, fitted, device, methods, windows, thresholds, combination=None):
    """A-4: seviye kayması ve spike şiddetini değiştirip olay recall'ünü ölçer.

    Girdi: en iyi yöntemler, her birinin seçilmiş penceresi ve val'de seçilmiş
    F1-max eşiği (yeniden seçilmez). Çıktı: [tip, çarpan, yöntem, seed] tablosu.
    Neden: Ana enjeksiyonda seviye kayması ×1,5 ve spike U(2, 4); tespit
    edilebilirliğin şiddete nasıl bağlı olduğunu görmek için yalnız ilgili
    çarpan değiştirilir. Rastgele sayı tüketimi değişmediğinden olayların
    yerleri ve tipleri ana enjeksiyonla aynıdır; yalnız büyüklük değişir.
    """
    severity = config["anomaly"]["severity"]
    plans = [("level_shift", factor) for factor in severity["level_shift_factors"]]
    plans += [("spike", factor) for factor in severity["spike_factors"]]
    rows = []
    for kind, factor in plans:
        settings = deepcopy(config["injection"])
        if kind == "level_shift":
            settings["level_shift"]["factor"] = factor
        else:
            settings["spike"]["factor_low"] = settings["spike"]["factor_high"] = factor
        outputs = []
        for seed in injection_seeds(config, "val"):
            scores, labels, injected = compute_scores(splits, config, "val", seed, fitted,
                                                      device, settings, combination)
            outputs.append((seed, scores, labels, injected))
        for run in make_runs(outputs, {method: windows[method] for method in methods}):
            for method in methods:
                metrics = evaluate_threshold(
                    run["labels"]["is_anomaly"].to_numpy(), run["scores"][method].to_numpy(),
                    run["labels"]["anomaly_type"].to_numpy(), thresholds[method],
                    valid_hours(run, method),
                )
                rows.append({"type": kind, "factor": factor, "method": method,
                             "smooth": windows[method], "seed": run["seed"],
                             "threshold": thresholds[method],
                             "event_recall": metrics[f"event_recall_{kind}"],
                             "point_recall": metrics[f"recall_{kind}"]})
    return pd.DataFrame(rows)


def save_figures(runs, per_seed, summary, figures_dir, split, prefix="A0-A3", notes=None):
    """Figürler: skor serisi (ilk kopya), birleşik karmaşıklık matrisleri, ROC/PR, tip recall."""
    notes = notes or {}
    tag = split.upper()
    first = runs[0]
    methods = list(first["scores"].columns)
    best = summary[summary["rule"] == "f1max"].set_index("method")
    best_seeds = per_seed[per_seed["rule"] == "f1max"]
    seeds_text = ",".join(str(run["seed"]) for run in runs)
    label = lambda method: METHOD_LABELS.get(method, method)  # noqa: E731

    figure, axes = plt.subplots(len(methods), 1, figsize=(14, 2.6 * len(methods)), sharex=True)
    for axis, method in zip(np.atleast_1d(axes), methods):
        scores, labels = first["scores"], first["labels"]
        axis.plot(scores.index, scores[method], linewidth=0.5, color="gray", label="skor")
        axis.axhline(best.loc[method, "threshold"], color="black", linestyle="--",
                     linewidth=0.8, label="F1-max eşiği")
        for kind, color in zip(ANOMALY_TYPES, ("red", "blue", "orange", "purple")):
            mask = (labels["anomaly_type"] == kind).to_numpy()
            axis.scatter(scores.index[mask], scores[method][mask], s=6, color=color, label=kind)
        axis.set_title(f"{label(method)} ({method}{notes.get(method, '')}) – {tag} enjekte kopya, "
                       f"seed={first['seed']}", fontsize=9)
        axis.set_ylabel("Skor")
    np.atleast_1d(axes)[0].legend(fontsize=7, ncol=6, loc="upper right")
    np.atleast_1d(axes)[-1].set_xlabel("Zaman (saat)")
    figure.tight_layout()
    figure.savefig(figures_dir / f"{prefix}_{split}_score_series.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(1, len(methods), figsize=(3.2 * len(methods), 3.4))
    for axis, method in zip(np.atleast_1d(axes), methods):
        totals = best_seeds[best_seeds["method"] == method][["tn", "fp", "fn", "tp"]].sum()
        matrix = np.array([[totals["tn"], totals["fp"]], [totals["fn"], totals["tp"]]])
        axis.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                axis.text(j, i, int(matrix[i, j]), ha="center", va="center")
        axis.set_xticks([0, 1], ["Normal", "Anomali"])
        axis.set_yticks([0, 1], ["Normal", "Anomali"])
        axis.set_xlabel("Tahmin")
        axis.set_ylabel("Gerçek (enjekte)")
        axis.set_title(f"{method}{notes.get(method, '')} – {tag}, F1-max\n(seed {seeds_text} toplamı)",
                       fontsize=7)
    figure.tight_layout()
    figure.savefig(figures_dir / f"{prefix}_{split}_confusion_matrices.png", dpi=200)
    plt.close(figure)

    figure, (roc_axis, pr_axis) = plt.subplots(1, 2, figsize=(13, 5.5))
    for method in methods:
        y = np.concatenate([run["labels"]["is_anomaly"].to_numpy()[valid_hours(run, method)]
                            for run in runs])
        s = np.concatenate([run["scores"][method].to_numpy()[valid_hours(run, method)]
                            for run in runs])
        fpr, tpr, _ = roc_curve(y, s)
        precision, recall, _ = precision_recall_curve(y, s)
        name = f"{method}{notes.get(method, '')}"
        roc_axis.plot(fpr, tpr, label=f"{name} (AUC={best.loc[method, 'roc_auc_mean']:.3f})")
        pr_axis.plot(recall, precision, label=f"{name} (AP={best.loc[method, 'pr_auc_mean']:.3f})")
    roc_axis.plot([0, 1], [0, 1], color="gray", linestyle=":")
    roc_axis.set(xlabel="Yanlış pozitif oranı", ylabel="Doğru pozitif oranı (recall)",
                 title=f"ROC – {tag}, seed {seeds_text} birleşik\n(lejant: kopya ortalaması)")
    pr_axis.axhline(y.mean(), color="gray", linestyle=":", label="Rastgele")
    pr_axis.set(xlabel="Recall", ylabel="Precision", title=f"Precision-Recall – {tag}, birleşik")
    roc_axis.legend(fontsize=7)
    pr_axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(figures_dir / f"{prefix}_{split}_roc_pr.png", dpi=200)
    plt.close(figure)

    figure, (point_axis, event_axis) = plt.subplots(1, 2, figsize=(15, 4.5), sharey=True)
    width = 0.8 / len(methods)
    positions = np.arange(len(ANOMALY_TYPES))
    for axis, metric, name in ((point_axis, "recall", "Nokta (saat)"),
                               (event_axis, "event_recall", "Olay")):
        for k, method in enumerate(methods):
            means = [best.loc[method, f"{metric}_{kind}_mean"] for kind in ANOMALY_TYPES]
            stds = [best.loc[method, f"{metric}_{kind}_std"] for kind in ANOMALY_TYPES]
            axis.bar(positions + k * width, means, width, yerr=stds, capsize=2,
                     label=f"{method}{notes.get(method, '')}")
        axis.set_xticks(positions + width * (len(methods) - 1) / 2, ANOMALY_TYPES)
        axis.set_ylim(0, 1)
        axis.set_title(f"{name} bazlı tip recall'ü – {tag}, F1-max, ort ± std ({len(runs)} seed)",
                       fontsize=9)
    point_axis.set_ylabel("Recall")
    event_axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(figures_dir / f"{prefix}_{split}_type_recall.png", dpi=200)
    plt.close(figure)


def save_smoothing_figure(smoothing_summary, figures_dir, methods, file_name):
    """A-3/A-5: yöntem başına F1 (F1-max eşiğinde) ve PR-AUC'nin pencereye göre değişimi (VAL)."""
    table = smoothing_summary[(smoothing_summary["rule"] == "f1max")
                              & smoothing_summary["method"].isin(methods)]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for axis, metric, name in ((axes[0], "f1", "F1 (F1-max eşiğinde)"), (axes[1], "pr_auc", "PR-AUC")):
        for method, group in table.groupby("method", sort=False):
            axis.errorbar(group["smooth"], group[f"{metric}_mean"], yerr=group[f"{metric}_std"],
                          marker="o", capsize=3, label=method)
        axis.set_xticks(sorted(table["smooth"].unique()))
        axis.set_xlabel("Geçmişe bakan hareketli ortalama penceresi (saat)")
        axis.set_ylabel(f"{name}, ort ± std (5 seed)")
        axis.set_title(f"Skor yumuşatma – VAL: {name}")
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(figures_dir / file_name, dpi=200)
    plt.close(figure)


def save_severity_figure(severity_summary, figures_dir):
    """A-4: şiddete göre olay recall'ü (VAL, val'de seçilmiş eşikler sabit)."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, kind, name in ((axes[0], "level_shift", "Seviye kayması çarpanı"),
                             (axes[1], "spike", "Spike çarpanı")):
        part = severity_summary[severity_summary["type"] == kind]
        for method, group in part.groupby("method", sort=False):
            axis.errorbar(group["factor"], group["event_recall_mean"], yerr=group["event_recall_std"],
                          marker="o", capsize=3, label=f"{method} (w={int(group['smooth'].iloc[0])})")
        axis.set_xlabel(f"{name} (× normal değer)")
        axis.set_xticks(sorted(part["factor"].unique()))
        axis.set_ylim(0, 1.05)
        axis.set_title(f"Şiddet taraması – VAL, {kind} olay recall'ü")
    axes[0].set_ylabel("Olay recall'ü, ort ± std (5 seed)")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figures_dir / "A-4_val_severity.png", dpi=200)
    plt.close(figure)


def experiment_rows(summary, config, suffix_for):
    """Özet tablosundan experiments.csv satırları (F1-max kuralı)."""
    rows = []
    for _, best in summary[summary["rule"] == "f1max"].iterrows():
        def fmt(metric):
            return f"{best[metric + '_mean']:.3f}±{best[metric + '_std']:.3f}"
        window = int(best["smooth"])
        rows.append({
            "exp_id": f"{best['method']}{suffix_for(window)}-eval5s",
            "date": date.today().isoformat(), "model": best["label"], "seed": config["seed"],
            "lookback": window, "val_mae": "", "val_rmse": "", "test_mae": "", "test_rmse": "",
            "note": (
                f"Anomali val, 5 enjeksiyon seed {config['injection']['val_seeds']}; "
                f"yumuşatma w={window} (geçmişe bakan ort.); birleşik F1-max eşik={best['threshold']:.4g}: "
                f"P={fmt('precision')} R={fmt('recall')} F1={fmt('f1')}; ROC-AUC={fmt('roc_auc')}; "
                f"PR-AUC={fmt('pr_auc')}; olay R={fmt('event_recall')} P={fmt('event_precision')}; "
                f"ayrıntı anomaly_val_results.csv / anomaly_val_smoothing.csv"
            ),
        })
    return rows


def main():
    """Varsayılan: yalnız val. --split test yalnız açık final değerlendirmede kullanılır."""
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_severity", action="store_true", help="A-4 taramasını atla")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    df = pd.read_parquet(resolve_path(PROJECT_ROOT, config["paths"]["processed_file"]))
    tables_dir = resolve_path(PROJECT_ROOT, config["paths"]["tables_dir"])
    figures_dir = resolve_path(PROJECT_ROOT, config["paths"]["figures_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    experiments_file = resolve_path(PROJECT_ROOT, config["paths"]["experiments_file"])

    splits = chronological_split(df, config)
    fitted = fit_detectors(splits, config, device)
    for name, table in fitted["std_tables"].items():
        table.to_csv(tables_dir / f"{name}_train_residual_std.csv", index=False)

    outputs, injection_rows = [], []
    for seed in injection_seeds(config, args.split):
        scores, labels, injected = compute_scores(splits, config, args.split, seed, fitted, device)
        outputs.append((seed, scores, labels, injected))
        counts = labels[labels["is_anomaly"] == 1]["anomaly_type"].value_counts()
        injection_rows.append({"seed": seed, **counts.to_dict(),
                               "ratio": labels["is_anomaly"].mean()})
    methods = list(outputs[0][1].columns)

    if args.split == "val":
        # A-3: her pencere için tüm yöntemler değerlendirilir; pencere val'de seçilir.
        smoothing_per_seed = evaluate_smoothing(outputs, methods, config)
        first_summary = summarize(smoothing_per_seed, ("method", "rule", "smooth"))
        first_windows = choose_smoothing(first_summary)
        # A-5b: A0 + en iyi A3 varyantı (A-5a dahil; seçilmiş penceredeki F1 ortalaması).
        chosen = first_summary[(first_summary["rule"] == "f1max")
                               & (first_summary["smooth"] == first_summary["method"].map(first_windows))]
        a3 = chosen[chosen["method"].str.startswith("A3")]
        best_a3 = a3.loc[a3["f1_mean"].idxmax(), "method"]
        combination = fit_combination(splits, config, fitted, device, ["A0", best_a3])
        for _, scores, _, _ in outputs:
            add_combinations(scores, combination)
        smoothing_per_seed = pd.concat(
            [smoothing_per_seed, evaluate_smoothing(outputs, COMBINATION_METHODS, config)],
            ignore_index=True,
        )
        smoothing_summary = summarize(smoothing_per_seed, ("method", "rule", "smooth"))
        smoothing_summary.to_csv(tables_dir / "anomaly_val_smoothing.csv", index=False)
        windows = choose_smoothing(smoothing_summary)
        per_seed = smoothing_per_seed[
            smoothing_per_seed["smooth"] == smoothing_per_seed["method"].map(windows)
        ].reset_index(drop=True)
        pd.DataFrame([
            {"component": m, "reference": "val enjeksiyonsuz, ham skor (w=1)",
             "n_hours": len(ref), "mean": ref.mean(), "std": ref.std(ddof=1),
             "median": np.median(ref), "p99": np.percentile(ref, 99)}
            for m, ref in combination["references"].items()
        ]).to_csv(tables_dir / "anomaly_val_combination.csv", index=False)
        save_smoothing_figure(smoothing_summary, figures_dir, FIGURE_METHODS, "A-3_val_smoothing.png")
        save_smoothing_figure(smoothing_summary, figures_dir, FIGURE_METHODS_A5, "A-5_val_smoothing.png")
        # Ham (w=1) satırlar A-1 ile aynı adlarla, seçilmiş w>1 satırları -sm{w} ekiyle.
        raw_summary = smoothing_summary[smoothing_summary["smooth"] == 1]
        upsert_csv(experiments_file, experiment_rows(raw_summary, config, lambda w: ""), ["exp_id"])
    else:
        # Test: pencere, eşik ve birleşim bileşenleri val tablolarından; test'te seçim yok.
        val_table = pd.read_csv(tables_dir / "anomaly_val_results.csv")
        components = pd.read_csv(tables_dir / "anomaly_val_combination.csv")["component"].tolist()
        combination = fit_combination(splits, config, fitted, device, components)
        for _, scores, _, _ in outputs:
            add_combinations(scores, combination)
        windows = val_table.groupby("method")["smooth"].first().astype(int).to_dict()
        runs = make_runs(outputs, windows)
        per_seed = evaluate_split(runs, "test", config, val_table)
        per_seed["smooth"] = per_seed["method"].map(windows)

    summary = summarize(per_seed, ("method", "rule", "smooth"))
    per_seed.to_csv(tables_dir / f"anomaly_{args.split}_results_per_seed.csv", index=False)
    summary.to_csv(tables_dir / f"anomaly_{args.split}_results.csv", index=False)
    score_tables = []
    for seed, scores, labels, injected in outputs:
        raw = scores.loc[injected.index]
        score_tables.append(labels.assign(seed=seed, **{f"score_{m}": raw[m] for m in raw}))
    pd.concat(score_tables).to_csv(tables_dir / f"anomaly_{args.split}_scores.csv", index_label="time")
    injection = pd.DataFrame(injection_rows)
    injection.to_csv(tables_dir / f"anomaly_{args.split}_injection_summary.csv", index=False)

    for prefix, selection in (("A0-A3", FIGURE_METHODS), ("A-5", FIGURE_METHODS_A5)):
        figure_methods = [m for m in selection if m in windows]
        figure_runs = make_runs(outputs, {m: windows[m] for m in figure_methods})
        notes = {m: f", w={windows[m]}" for m in figure_methods}
        save_figures(figure_runs, per_seed[per_seed["method"].isin(figure_methods)],
                     summary[summary["method"].isin(figure_methods)], figures_dir, args.split,
                     prefix=prefix, notes=notes)

    if args.split == "val":
        chosen = summary[(summary["rule"] == "f1max") & (summary["smooth"] > 1)]
        upsert_csv(experiments_file, experiment_rows(chosen, config, lambda w: f"-sm{w}"), ["exp_id"])
        # Eşleştirilmiş seed farkları: A0'a ve en iyi yönteme karşı.
        best = summary[summary["rule"] == "f1max"].sort_values("f1_mean", ascending=False)
        paired = pd.concat([paired_differences(per_seed, "A0"),
                            paired_differences(per_seed, best["method"].iloc[0])], ignore_index=True)
        paired.to_csv(tables_dir / "anomaly_val_paired.csv", index=False)

        if not args.skip_severity:
            # A-4/A-5: en iyi k yöntem + config'teki sabit referanslar; eşikler sabit.
            severity_settings = config["anomaly"]["severity"]
            top = best["method"].head(severity_settings["top_k"]).tolist()
            top = list(dict.fromkeys(top + severity_settings["always_include"]))
            thresholds = best.set_index("method")["threshold"].to_dict()
            severity = severity_sweep(splits, config, fitted, device, top, windows, thresholds,
                                      combination)
            severity.to_csv(tables_dir / "anomaly_val_severity_per_seed.csv", index=False)
            severity_summary = (
                severity.groupby(["type", "factor", "method", "smooth", "threshold"], sort=False)
                .agg(event_recall_mean=("event_recall", "mean"),
                     event_recall_std=("event_recall", "std"),
                     point_recall_mean=("point_recall", "mean"),
                     point_recall_std=("point_recall", "std"),
                     n_seeds_with_type=("event_recall", "count"))
                .reset_index()
            )
            severity_summary.to_csv(tables_dir / "anomaly_val_severity.csv", index=False)
            save_severity_figure(severity_summary, figures_dir)
            rows = []
            for method, group in severity_summary.groupby("method", sort=False):
                parts = [f"{r.type}×{r.factor:g}: {r.event_recall_mean:.2f}±{r.event_recall_std:.2f}"
                         for r in group.itertuples()]
                rows.append({
                    "exp_id": f"A4sev-{method}", "date": date.today().isoformat(),
                    "model": METHOD_LABELS.get(method, method), "seed": config["seed"],
                    "lookback": windows[method], "val_mae": "", "val_rmse": "", "test_mae": "",
                    "test_rmse": "",
                    "note": ("Şiddet taraması (val, 5 seed; eşik ve w ana val seçiminden sabit); "
                             "olay recall'ü " + "; ".join(parts) + "; anomaly_val_severity.csv"),
                })
            upsert_csv(experiments_file, rows, ["exp_id"])
            print("\nŞiddet taraması (olay recall'ü):")
            print(severity_summary.round(3).to_string(index=False))

    table = summary[summary["rule"] == "f1max"]
    columns = ["method", "smooth", "threshold"] + [f"{m}_{s}" for m in
               ("f1", "pr_auc", "roc_auc", "event_recall", "event_precision") for s in ("mean", "std")]
    print(table[columns].round(3).to_string(index=False))
    print("\nEnjeksiyon özetleri (saat):\n" + injection.to_string(index=False))


if __name__ == "__main__":
    main()
