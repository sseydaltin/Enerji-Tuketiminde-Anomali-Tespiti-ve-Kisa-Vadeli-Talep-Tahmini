"""Aşama 1 – mevcut anomali yöntemlerinin teşhisi (docs/gelistirme_protokolu.md §5).

Yalnız kış-dev (= val-dev, 2009-10 – 2009-12), ana senaryo, 10 enjeksiyon seed'i.
Pencere ve eşikler final protokolde val-dev'de seçilmiş KAYITLI değerlerdir
(results/tables/anomaly_protocol_selection.csv); burada HİÇBİR seçim yapılmaz.
val-hold ve test kullanılmaz.

Çıktılar (results/tables/diag_*.csv, results/figures/D1_*.png):
- yöntem başına: olay F1 (kayıtlı sonuçla eşleşme kontrolü), günlük yanlış alarm, tespit gecikmesi
- yanlış alarmların günün saatine göre dağılımı ve seed'ler arası tekrar eden yanlış alarm olayları
- tip ve hafta içi / hafta sonu kırılımında olay recall'ü
- en uzun kaçırılan olaylar
- enjeksiyonsuz kış-dev'de A3 normalize skorunun saate göre dağılımı (eşiğin neye uygulandığı)
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.anomaly_protocol import protocol_splits, score_copies
from src.data.inject_anomalies import ANOMALY_TYPES, train_sigma_table
from src.data.load import PROJECT_ROOT, load_config
from src.evaluate_anomaly import fit_combination, fit_detectors, make_runs, score_frame, valid_hours, with_context
from src.utils.anomaly_metrics import alarm_diagnostics, event_metrics, runs
from src.utils.console import setup_console
from src.utils.results_io import resolve_path
from src.utils.seed import set_seed

SPLIT = "val_dev"        # kış-dev; bu betik başka küme kabul etmez


def flags_for(run, method, threshold):
    """Bir kopyada yöntemin işaretlediği saatler [N] (tam eksen; geçersiz saat işaretlenmez)."""
    scores = run["scores"][method].to_numpy(dtype=float)
    valid = valid_hours(run, method)
    with np.errstate(invalid="ignore"):
        return (scores >= threshold) & valid, valid


def event_rows(run, method, predicted):
    """Kopyadaki her enjekte olay için bir satır: tip, başlangıç, süre, yakalandı mı, gecikme."""
    types = run["labels"]["anomaly_type"].to_numpy()
    index = run["labels"].index
    rows = []
    for kind in ANOMALY_TYPES:
        for start, end in runs(types == kind):
            hits = np.flatnonzero(predicted[start:end])
            rows.append({
                "seed": run["seed"], "method": method, "type": kind,
                "start": index[start], "length_h": int(end - start),
                "weekend": index[start].dayofweek >= 5,
                "detected": bool(hits.size), "delay_h": int(hits[0]) if hits.size else np.nan,
            })
    return rows


def false_alarm_starts(run, predicted):
    """Yanlış alarm bloklarının başlangıç zaman damgaları (anomaliyle kesişmeyen bloklar)."""
    anomalous = run["labels"]["anomaly_type"].to_numpy() != "none"
    index = run["labels"].index
    return [index[start] for start, end in runs(predicted) if not anomalous[start:end].any()]


def diagnose(runs_, windows, thresholds):
    """Her yöntem × seed için metrikleri, olay satırlarını ve yanlış alarm saatlerini toplar."""
    metric_rows, events, fa_hours = [], [], []
    for method, threshold in thresholds.items():
        for run in runs_:
            predicted, valid = flags_for(run, method, threshold)
            types = run["labels"]["anomaly_type"].to_numpy()
            ev = event_metrics(types, predicted)
            al = alarm_diagnostics(types, predicted, int(valid.sum()))
            metric_rows.append({
                "seed": run["seed"], "method": method, "smooth": windows[method], "threshold": threshold,
                "event_f1": ev["event_f1"], "event_recall": ev["event_recall"],
                "event_precision": ev["event_precision"], "n_events": ev["n_events"],
                "n_alarm_blocks": ev["n_pred_blocks"], "false_alarms": al["false_alarms"],
                "false_alarms_per_day": al["false_alarms_per_day"],
                "delay_mean_h": al["detection_delay_mean"], "delay_median_h": al["detection_delay_median"],
            })
            events += event_rows(run, method, predicted)
            fa_hours += [{"method": method, "seed": run["seed"], "start": t, "hour": t.hour}
                         for t in false_alarm_starts(run, predicted)]
    return pd.DataFrame(metric_rows), pd.DataFrame(events), pd.DataFrame(fa_hours)


def check_against_saved(per_seed, tables_dir):
    """Olay F1'i kayıtlı protokol sonucuyla (val_dev, ana) karşılaştırır; fark > 1e-9 ise hata."""
    saved = pd.read_csv(tables_dir / "anomaly_protocol_val_per_seed.csv")
    saved = saved[(saved["split"] == SPLIT) & (saved["scenario"] == "main")]
    merged = per_seed.merge(saved[["seed", "method", "event_f1"]], on=["seed", "method"],
                            suffixes=("", "_saved"))
    if len(merged) != len(per_seed):
        raise ValueError("Kayıtlı protokol tablosunda eksik yöntem/seed var.")
    gap = (merged["event_f1"] - merged["event_f1_saved"]).abs().max()
    if gap > 1e-9:
        raise ValueError(f"Olay F1 kayıtlı sonuçtan farklı (en büyük fark {gap:.3g}).")
    return gap


def summary_table(per_seed, events):
    """Yöntem başına özet: ortalama ± std ve tip / hafta içi–sonu olay recall'ü."""
    agg = per_seed.groupby("method").agg(
        event_f1_mean=("event_f1", "mean"), event_f1_std=("event_f1", "std"),
        false_alarms_per_day_mean=("false_alarms_per_day", "mean"),
        false_alarms_per_day_std=("false_alarms_per_day", "std"),
        delay_mean_h=("delay_mean_h", "mean"), delay_median_h=("delay_median_h", "median"),
        alarm_blocks_mean=("n_alarm_blocks", "mean"), n_events_mean=("n_events", "mean"),
    )
    by_type = events.pivot_table(index="method", columns="type", values="detected", aggfunc="mean")
    by_type.columns = [f"recall_{c}" for c in by_type.columns]
    by_day = events.pivot_table(index="method", columns="weekend", values="detected", aggfunc="mean")
    by_day.columns = ["recall_weekday" if not c else "recall_weekend" for c in by_day.columns]
    return agg.join(by_type).join(by_day).reset_index()


def clean_score_by_hour(splits, config, fitted, device, combination, method):
    """Enjeksiyonsuz kış-dev'de bir yöntemin (yumuşatılmamış) skorunun saate göre yüzdelikleri."""
    from src.evaluate_anomaly import add_combinations
    clean = splits[SPLIT]
    scores = score_frame(with_context(splits, SPLIT, clean, config), config, fitted, device)
    add_combinations(scores, combination)
    series = scores.loc[clean.index, method]
    series = series[clean["is_imputed"].to_numpy() == 0]
    table = series.groupby(series.index.hour).quantile([0.5, 0.9, 0.99]).unstack()
    table.columns = ["p50", "p90", "p99"]
    return table.rename_axis("hour").reset_index()


def save_figures(fa_hours, events, clean_a3, threshold_a3, figures_dir):
    """D1 figürleri: yanlış alarm saatleri, gecikme dağılımı, temiz A3 skoru (kış-dev)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    counts = fa_hours.groupby(["hour", "method"]).size().unstack(fill_value=0) / fa_hours["seed"].nunique()
    counts.reindex(range(24), fill_value=0).plot(ax=ax, marker="o")
    ax.set_xlabel("Yanlış alarm bloğunun başladığı saat (0–23)")
    ax.set_ylabel("Yanlış alarm / kopya")
    ax.set_title("Yanlış alarmların günün saatine göre dağılımı – kış-dev (val-dev), ana senaryo, 10 seed")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "D1_false_alarms_by_hour_valdev.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    detected = events[events["detected"]]
    detected.boxplot(column="delay_h", by=["method"], ax=ax, rot=30)
    ax.set_xlabel("Yöntem")
    ax.set_ylabel("Tespit gecikmesi (saat)")
    ax.set_title("Yakalanan olaylarda tespit gecikmesi – kış-dev (val-dev), ana senaryo, 10 seed")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(figures_dir / "D1_detection_delay_valdev.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    for col, style in (("p50", "-"), ("p90", "--"), ("p99", ":")):
        ax.plot(clean_a3["hour"], clean_a3[col], style, label=f"{col} (enjeksiyonsuz)")
    ax.axhline(threshold_a3, color="red", label=f"Eşik {threshold_a3:.2f} (w=3 skora uygulanır)")
    ax.set_xlabel("Günün saati")
    ax.set_ylabel("A3-s2s-norm ham skoru (|hata| / OOF std, birimsiz)")
    ax.set_title("A3-s2s-norm skorunun saate göre dağılımı – enjeksiyonsuz kış-dev (val-dev)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "D1_a3_clean_score_by_hour_valdev.png", dpi=200)
    plt.close(fig)


def main():
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_config()
    set_seed(config["seed"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    df = pd.read_parquet(resolve_path(PROJECT_ROOT, config["paths"]["processed_file"]))
    tables_dir = resolve_path(PROJECT_ROOT, config["paths"]["tables_dir"])
    figures_dir = resolve_path(PROJECT_ROOT, config["paths"]["figures_dir"])
    protocol = config["protocol"]

    # Final protokolle aynı hazırlık (train/OOF fit; kombinasyon referansı temiz val_dev).
    splits = protocol_splits(df, config)
    fitted = fit_detectors(splits, config, device, std_source=protocol["std_source"])
    sigma_table = train_sigma_table(splits["train"], config["preprocess"]["target"])
    combination = fit_combination(splits, config, fitted, device,
                                  protocol["combination_components"], reference_split=SPLIT)

    # Kayıtlı (val_dev'de seçilmiş) pencere ve eşikler; yeni seçim yok.
    selection = pd.read_csv(tables_dir / "anomaly_protocol_selection.csv").set_index("method")
    windows = selection["smooth"].astype(int).to_dict()
    thresholds = selection["threshold"].to_dict()

    outputs = score_copies(splits, config, SPLIT, "main", fitted, device, combination, sigma_table)
    runs_ = make_runs(outputs, windows)
    per_seed, events, fa_hours = diagnose(runs_, windows, thresholds)
    gap = check_against_saved(per_seed, tables_dir)

    summary = summary_table(per_seed, events)
    missed = (events[~events["detected"]].sort_values("length_h", ascending=False)
              .groupby("method").head(5))
    clean_a3 = clean_score_by_hour(splits, config, fitted, device, combination, "A3-s2s-norm")

    per_seed.to_csv(tables_dir / "diag_valdev_per_seed.csv", index=False)
    events.to_csv(tables_dir / "diag_valdev_events.csv", index=False)
    summary.to_csv(tables_dir / "diag_valdev_summary.csv", index=False)
    missed.to_csv(tables_dir / "diag_valdev_longest_missed.csv", index=False)
    clean_a3.to_csv(tables_dir / "diag_valdev_a3_clean_score_by_hour.csv", index=False)
    fa_by_hour = fa_hours.groupby(["method", "hour"]).size().rename("count").reset_index()
    fa_by_hour.to_csv(tables_dir / "diag_valdev_false_alarms_by_hour.csv", index=False)
    # Aynı zaman damgasında kaç seed'de yanlış alarm var? 10/10 = enjeksiyondan bağımsız,
    # temiz verideki gerçek bir olağandışı olay (etiketi olmadığı için "yanlış" sayılıyor).
    recurring = (fa_hours.groupby(["method", "start"])["seed"].nunique().rename("n_seeds")
                 .reset_index().sort_values(["method", "n_seeds"], ascending=[True, False]))
    recurring.to_csv(tables_dir / "diag_valdev_false_alarm_events.csv", index=False)
    save_figures(fa_hours, events, clean_a3, thresholds["A3-s2s-norm"], figures_dir)

    print(f"Kayıtlı olay F1 ile en büyük fark: {gap:.2e}")
    print(summary.round(3).to_string(index=False))
    print("\nEn uzun kaçırılan olaylar (yöntem başına 5):")
    print(missed[["method", "seed", "type", "start", "length_h"]].to_string(index=False))


if __name__ == "__main__":
    main()
