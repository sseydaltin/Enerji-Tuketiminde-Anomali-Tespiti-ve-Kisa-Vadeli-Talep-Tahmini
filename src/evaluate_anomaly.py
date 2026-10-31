"""A0–A2 anomali dedektörlerini enjekte edilmiş val (varsayılan) veya test kopyalarında değerlendirir.

Protokol (A-1): val'e 5 farklı seed ile enjeksiyon yapılır; eşik her yöntem için
5 kopyanın BİRLEŞİMİNDE F1-max ile seçilir; metrikler kopya başına hesaplanıp
ortalama ± std raporlanır. Test tek kopyadır (seed=2) ve val eşiklerini kullanır.
"""

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


def compute_scores(splits, config, split, seed, forest, device):
    """Seçilen kümenin bir enjekte kopyası için tüm yöntemlerin skorlarını hesaplar.

    Girdi: kronolojik bölmeler, config, küme adı, enjeksiyon seed'i, train'e
    fit edilmiş Isolation Forest, cihaz.
    Çıktı: (skor tablosu [saat, yöntem], etiket tablosu, enjekte kopya).
    Neden: Geçmiş bağlam (son 8 gün) bir önceki kümeden ENJEKSİYONSUZ alınır;
    yalnız değerlendirilen kümeye anomali eklenir. A1 ve A2 yalnız train'de
    fit/eğitilmiştir; burada yalnız skor üretilir.
    """
    anomaly = config["anomaly"]
    target = config["preprocess"]["target"]
    injected, labels = inject_anomalies(splits[split], target, config["injection"], seed)
    previous = splits["train"] if split == "val" else splits["val"]
    context_hours = anomaly["zscore_window"] + anomaly["ae"]["window"]
    full = pd.concat((previous.tail(context_hours), injected))

    scores = pd.DataFrame(index=full.index)
    scores["A0"] = rolling_zscore(full[target], anomaly["zscore_window"], anomaly["zscore_min_std"])
    scores["A1"] = isolation_forest_scores(
        forest, window_features(full, target, config["features"]["calendar"])
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

    # Yalnız değerlendirilen kümenin saatleri.
    return scores.loc[injected.index], labels, injected


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


def summarize(per_seed):
    """Kopya başına sonuçlardan yöntem × kural için ortalama ve std (ddof=1) tablosu.

    Neden: Tek enjeksiyon kopyasında ~100 olay var; olay yerleşimi şansı
    sonuçları oynatır. 5 kopyanın std'si bu belirsizliği gösterir.
    """
    rows = []
    for (method, rule), group in per_seed.groupby(["method", "rule"], sort=False):
        row = {"method": method, "label": group["label"].iloc[0], "rule": rule,
               "threshold": group["threshold"].iloc[0], "n_seeds": len(group),
               "n_events_mean": group["n_events"].mean()}
        for metric in SUMMARY_METRICS:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def save_figures(runs, per_seed, summary, figures_dir, split, prefix="A0-A2"):
    """Figürler: skor serisi (ilk kopya), birleşik karmaşıklık matrisleri, ROC/PR, tip recall."""
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
        axis.set_title(f"{label(method)} ({method}) – {tag} enjekte kopya, seed={first['seed']}",
                       fontsize=9)
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
        axis.set_title(f"{method} – {tag}, F1-max\n(seed {seeds_text} toplamı)", fontsize=7)
    figure.tight_layout()
    figure.savefig(figures_dir / f"{prefix}_{split}_confusion_matrices.png", dpi=200)
    plt.close(figure)

    figure, (roc_axis, pr_axis) = plt.subplots(1, 2, figsize=(12, 5))
    for method in methods:
        y = np.concatenate([run["labels"]["is_anomaly"].to_numpy()[valid_hours(run, method)]
                            for run in runs])
        s = np.concatenate([run["scores"][method].to_numpy()[valid_hours(run, method)]
                            for run in runs])
        fpr, tpr, _ = roc_curve(y, s)
        precision, recall, _ = precision_recall_curve(y, s)
        roc_axis.plot(fpr, tpr, label=f"{method} (AUC={best.loc[method, 'roc_auc_mean']:.3f})")
        pr_axis.plot(recall, precision, label=f"{method} (AP={best.loc[method, 'pr_auc_mean']:.3f})")
    roc_axis.plot([0, 1], [0, 1], color="gray", linestyle=":")
    roc_axis.set(xlabel="Yanlış pozitif oranı", ylabel="Doğru pozitif oranı (recall)",
                 title=f"ROC – {tag}, seed {seeds_text} birleşik\n(lejant: kopya ortalaması)")
    pr_axis.axhline(y.mean(), color="gray", linestyle=":", label="Rastgele")
    pr_axis.set(xlabel="Recall", ylabel="Precision", title=f"Precision-Recall – {tag}, birleşik")
    roc_axis.legend(fontsize=8)
    pr_axis.legend(fontsize=8)
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
            axis.bar(positions + k * width, means, width, yerr=stds, capsize=2, label=method)
        axis.set_xticks(positions + width * (len(methods) - 1) / 2, ANOMALY_TYPES)
        axis.set_ylim(0, 1)
        axis.set_title(f"{name} bazlı tip recall'ü – {tag}, F1-max, ort ± std ({len(runs)} seed)",
                       fontsize=9)
    point_axis.set_ylabel("Recall")
    event_axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(figures_dir / f"{prefix}_{split}_type_recall.png", dpi=200)
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

    splits = chronological_split(df, config)
    target = config["preprocess"]["target"]
    # A1: yalnız train (enjeksiyonsuz) özniteliklerine bir kez fit.
    forest = fit_isolation_forest(
        window_features(splits["train"], target, config["features"]["calendar"]),
        config["anomaly"]["isolation_forest"], config["seed"],
    )
    runs, score_tables, injection_rows = [], [], []
    for seed in injection_seeds(config, args.split):
        scores, labels, injected = compute_scores(splits, config, args.split, seed, forest, device)
        runs.append({"seed": seed, "scores": scores, "labels": labels,
                     "imputed": injected["is_imputed"].to_numpy(dtype=bool)})
        score_tables.append(labels.assign(seed=seed, **{f"score_{m}": scores[m] for m in scores}))
        counts = labels[labels["is_anomaly"] == 1]["anomaly_type"].value_counts()
        injection_rows.append({"seed": seed, **counts.to_dict(),
                               "ratio": labels["is_anomaly"].mean()})

    val_thresholds = None
    if args.split == "test":
        # Eşikler yalnız val'de seçilmiş olmalı; test'te yeniden seçilmez.
        val_thresholds = pd.read_csv(tables_dir / "anomaly_val_results.csv")
    per_seed = evaluate_split(runs, args.split, config, val_thresholds)
    summary = summarize(per_seed)
    per_seed.to_csv(tables_dir / f"anomaly_{args.split}_results_per_seed.csv", index=False)
    summary.to_csv(tables_dir / f"anomaly_{args.split}_results.csv", index=False)
    pd.concat(score_tables).to_csv(tables_dir / f"anomaly_{args.split}_scores.csv", index_label="time")
    injection = pd.DataFrame(injection_rows)
    injection.to_csv(tables_dir / f"anomaly_{args.split}_injection_summary.csv", index=False)
    save_figures(runs, per_seed, summary, figures_dir, args.split)

    if args.split == "val":
        rows = []
        for _, best in summary[summary["rule"] == "f1max"].iterrows():
            def fmt(metric):
                return f"{best[metric + '_mean']:.3f}±{best[metric + '_std']:.3f}"
            rows.append({
                "exp_id": f"{best['method']}-eval5s", "date": date.today().isoformat(),
                "model": best["label"], "seed": config["seed"],
                "val_mae": "", "val_rmse": "", "test_mae": "", "test_rmse": "",
                "note": (
                    f"Anomali val, 5 enjeksiyon seed {config['injection']['val_seeds']}; "
                    f"birleşik F1-max eşik={best['threshold']:.4g}: P={fmt('precision')} "
                    f"R={fmt('recall')} F1={fmt('f1')}; ROC-AUC={fmt('roc_auc')}; "
                    f"PR-AUC={fmt('pr_auc')}; olay R={fmt('event_recall')} "
                    f"P={fmt('event_precision')}; ayrıntı anomaly_val_results.csv"
                ),
            })
        upsert_csv(resolve_path(PROJECT_ROOT, config["paths"]["experiments_file"]), rows, ["exp_id"])

    table = summary[summary["rule"] == "f1max"]
    columns = ["method", "threshold"] + [f"{m}_{s}" for m in
               ("precision", "recall", "f1", "roc_auc", "pr_auc", "event_recall", "event_precision")
               for s in ("mean", "std")]
    print(table[columns].round(3).to_string(index=False))
    print("\nEnjeksiyon özetleri (saat):\n" + injection.to_string(index=False))


if __name__ == "__main__":
    main()
