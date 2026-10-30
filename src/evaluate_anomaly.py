"""A0–A2 anomali dedektörlerini enjekte edilmiş val (varsayılan) veya test kopyasında değerlendirir."""

import argparse
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
from src.utils.anomaly_metrics import evaluate_threshold, f1_max_threshold
from src.utils.console import setup_console
from src.utils.exp_logger import resolve_path, upsert_csv
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
}
# Değerlendirilecek AE checkpoint'leri (models/{exp_id}_best.pt; yoksa atlanır).
AUTOENCODER_EXPERIMENTS = ("A2", "A2-noAug", "A2-lat8")


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


def compute_scores(df, config, split, device):
    """Seçilen kümenin enjekte kopyası için tüm yöntemlerin skorlarını hesaplar.

    Çıktı: (skor tablosu [saat, yöntem], etiket tablosu, enjekte kopya).
    Neden: Geçmiş bağlam (son 8 gün) bir önceki kümeden ENJEKSİYONSUZ alınır;
    yalnız değerlendirilen kümeye anomali eklenir. A1 ve A2 yalnız train'de
    fit/eğitilmiştir; burada yalnız skor üretilir.
    """
    anomaly = config["anomaly"]
    target = config["preprocess"]["target"]
    splits = chronological_split(df, config)
    seed = config["injection"]["val_seed"] if split == "val" else config["injection"]["test_seed"]
    injected, labels = inject_anomalies(splits[split], target, config["injection"], seed)
    previous = splits["train"] if split == "val" else splits["val"]
    context_hours = anomaly["zscore_window"] + anomaly["ae"]["window"]
    full = pd.concat((previous.tail(context_hours), injected))

    scores = pd.DataFrame(index=full.index)
    scores["A0"] = rolling_zscore(full[target], anomaly["zscore_window"], anomaly["zscore_min_std"])

    calendar = config["features"]["calendar"]
    # A1: yalnız train (enjeksiyonsuz) özniteliklerine fit.
    forest = fit_isolation_forest(
        window_features(splits["train"], target, calendar),
        anomaly["isolation_forest"], config["seed"],
    )
    scores["A1"] = isolation_forest_scores(forest, window_features(full, target, calendar))

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

    # Yalnız değerlendirilen kümenin saatleri.
    return scores.loc[injected.index], labels, injected


def evaluate_split(scores, labels, imputed, split, config, val_thresholds=None):
    """Her yöntem için eşik kurallarında metrikleri hesaplar.

    val: F1-max eşiği etiketlerle seçilir; yüzdelik eşikler skor dağılımından.
    test: eşikler val tablosundan okunur ve aynen uygulanır (test'te seçim yok).
    Değerlendirmeye doldurulmuş (is_imputed=1) saatler ve skoru olmayan saatler girmez.
    """
    rows = []
    percentiles = config["anomaly"]["threshold_percentiles"]
    for method in scores.columns:
        mask = scores[method].notna().to_numpy() & ~imputed
        y = labels["is_anomaly"].to_numpy()[mask]
        s = scores[method].to_numpy()[mask]
        types = labels["anomaly_type"].to_numpy()[mask]
        if split == "val":
            rules = {"f1max": f1_max_threshold(y, s)}
            for p in percentiles:
                rules[f"p{p:g}"] = float(np.percentile(s, p))
        else:
            chosen = val_thresholds[val_thresholds["method"] == method]
            rules = dict(zip(chosen["rule"], chosen["threshold"]))
        for rule, threshold in rules.items():
            rows.append({
                "split": split, "method": method, "label": METHOD_LABELS[method],
                "rule": rule, **evaluate_threshold(y, s, types, threshold),
            })
    return pd.DataFrame(rows)


def save_figures(scores, labels, imputed, results, figures_dir, split):
    """VAL figürleri: skor serileri, karmaşıklık matrisleri, ROC/PR, tip recall."""
    tag = split.upper()
    methods = list(scores.columns)
    best = results[results["rule"] == "f1max"].set_index("method")
    anomalous = labels["is_anomaly"].to_numpy().astype(bool)

    figure, axes = plt.subplots(len(methods), 1, figsize=(14, 2.6 * len(methods)), sharex=True)
    for axis, method in zip(np.atleast_1d(axes), methods):
        axis.plot(scores.index, scores[method], linewidth=0.5, color="gray", label="skor")
        axis.axhline(best.loc[method, "threshold"], color="black", linestyle="--",
                     linewidth=0.8, label="F1-max eşiği")
        for kind, color in zip(ANOMALY_TYPES, ("red", "blue", "orange", "purple")):
            mask = (labels["anomaly_type"] == kind).to_numpy()
            axis.scatter(scores.index[mask], scores[method][mask], s=6, color=color, label=kind)
        axis.set_title(f"{METHOD_LABELS[method]} ({method}) – {tag} (enjekte kopya)", fontsize=9)
        axis.set_ylabel("Skor")
    np.atleast_1d(axes)[0].legend(fontsize=7, ncol=6, loc="upper right")
    np.atleast_1d(axes)[-1].set_xlabel("Zaman (saat)")
    figure.tight_layout()
    figure.savefig(figures_dir / f"A0-A2_{split}_score_series.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(1, len(methods), figsize=(3.2 * len(methods), 3.4))
    for axis, method in zip(np.atleast_1d(axes), methods):
        row = best.loc[method]
        matrix = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
        axis.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                axis.text(j, i, int(matrix[i, j]), ha="center", va="center")
        axis.set_xticks([0, 1], ["Normal", "Anomali"])
        axis.set_yticks([0, 1], ["Normal", "Anomali"])
        axis.set_xlabel("Tahmin")
        axis.set_ylabel("Gerçek (enjekte)")
        axis.set_title(f"{method} – {tag}, F1-max", fontsize=8)
    figure.tight_layout()
    figure.savefig(figures_dir / f"A0-A2_{split}_confusion_matrices.png", dpi=200)
    plt.close(figure)

    figure, (roc_axis, pr_axis) = plt.subplots(1, 2, figsize=(12, 5))
    for method in methods:
        mask = scores[method].notna().to_numpy() & ~imputed
        y, s = anomalous[mask], scores[method].to_numpy()[mask]
        fpr, tpr, _ = roc_curve(y, s)
        precision, recall, _ = precision_recall_curve(y, s)
        roc_axis.plot(fpr, tpr, label=f"{method} (AUC={best.loc[method, 'roc_auc']:.3f})")
        pr_axis.plot(recall, precision, label=f"{method} (AP={best.loc[method, 'pr_auc']:.3f})")
    roc_axis.plot([0, 1], [0, 1], color="gray", linestyle=":")
    roc_axis.set(xlabel="Yanlış pozitif oranı", ylabel="Doğru pozitif oranı (recall)",
                 title=f"ROC eğrileri – {tag}")
    pr_axis.axhline(anomalous[~imputed].mean(), color="gray", linestyle=":", label="Rastgele")
    pr_axis.set(xlabel="Recall", ylabel="Precision", title=f"Precision-Recall eğrileri – {tag}")
    roc_axis.legend(fontsize=8)
    pr_axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figures_dir / f"A0-A2_{split}_roc_pr.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 4.5))
    width = 0.8 / len(methods)
    for k, method in enumerate(methods):
        values = [best.loc[method, f"recall_{kind}"] for kind in ANOMALY_TYPES]
        axis.bar(np.arange(len(ANOMALY_TYPES)) + k * width, values, width, label=method)
    axis.set_xticks(np.arange(len(ANOMALY_TYPES)) + width * (len(methods) - 1) / 2, ANOMALY_TYPES)
    axis.set_ylabel("Recall (F1-max eşiğinde)")
    axis.set_ylim(0, 1)
    axis.set_title(f"Anomali tipine göre recall – {tag}")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figures_dir / f"A0-A2_{split}_type_recall.png", dpi=200)
    plt.close(figure)


def main():
    """Varsayılan: yalnız val. --split test yalnız açık final değerlendirmede kullanılır."""
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="cuda")
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

    scores, labels, injected = compute_scores(df, config, args.split, device)
    imputed = injected["is_imputed"].to_numpy(dtype=bool)
    val_thresholds = None
    if args.split == "test":
        # Eşikler yalnız val'de seçilmiş olmalı; test'te yeniden seçilmez.
        val_thresholds = pd.read_csv(tables_dir / "anomaly_val_results.csv")
    results = evaluate_split(scores, labels, imputed, args.split, config, val_thresholds)
    results.to_csv(tables_dir / f"anomaly_{args.split}_results.csv", index=False)
    labels.assign(**{f"score_{m}": scores[m] for m in scores.columns}).to_csv(
        tables_dir / f"anomaly_{args.split}_scores.csv", index_label="time"
    )
    summary = labels[labels["is_anomaly"] == 1]["anomaly_type"].value_counts()
    summary.rename("hours").to_csv(tables_dir / f"anomaly_{args.split}_injection_summary.csv")
    save_figures(scores, labels, imputed, results, figures_dir, args.split)

    if args.split == "val":
        rows = []
        for method, group in results.groupby("method", sort=False):
            best = group[group["rule"] == "f1max"].iloc[0]
            rows.append({
                "exp_id": f"{method}-eval", "date": date.today().isoformat(),
                "model": METHOD_LABELS[method], "seed": config["seed"],
                "val_mae": "", "val_rmse": "", "test_mae": "", "test_rmse": "",
                "note": (
                    f"Anomali val (enjekte, seed={config['injection']['val_seed']}); "
                    f"F1-max eşik={best['threshold']:.4g}: P={best['precision']:.3f} "
                    f"R={best['recall']:.3f} F1={best['f1']:.3f}; ROC-AUC={best['roc_auc']:.3f}; "
                    f"PR-AUC={best['pr_auc']:.3f}; ayrıntı anomaly_val_results.csv"
                ),
            })
        upsert_csv(resolve_path(PROJECT_ROOT, config["paths"]["experiments_file"]), rows, ["exp_id"])

    columns = ["method", "rule", "threshold", "precision", "recall", "f1", "roc_auc", "pr_auc",
               "recall_spike", "recall_drop", "recall_level_shift", "recall_pattern"]
    print(results[columns].round(3).to_string(index=False))
    print("\nEnjekte saat sayıları:\n" + summary.to_string())


if __name__ == "__main__":
    main()
