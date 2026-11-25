"""Anomali geliştirme değerlendirmesi: kış-dev + yaz-dev, kolay/orta/zor (docs/gelistirme_protokolu.md).

Kurallar:
- Kümeler: kış-dev (= val_dev, 2009-10 – 2009-12) ve yaz-dev (= OOF kat 3, 2009-06 – 2009-09).
  val_hold ve test bu modülde HİÇ kullanılmaz.
- Kış-dev: final protokolün modelleri ve OOF std'si (tüm train ile).
- Yaz-dev: kat 3'ü görmemiş zincir (src/train_summer_dev.py): Optuna öncesi tahmin modelleri,
  kat 1–2 OOF std'si, yeniden fit edilmiş A1, yeniden eğitilmiş A2; σ tablosu ve spike sınırı
  yalnız 2009-05-31'e kadarki veriden.
- Seçim (pencere, eşik) yalnız ORTA senaryoda, iki kümenin kopyaları birlikte; kolay ve zor
  aynı pencere/eşikle ayrıca raporlanır.

Bu betik (Aşama 2) mevcut yöntemlerin yeni protokoldeki başlangıç noktasını üretir.
Komut: python -m src.anomaly_dev
"""

import argparse
from copy import deepcopy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.anomaly_protocol import protocol_splits
from src.data.inject_anomalies import inject_anomalies, sigma_for, train_sigma_table
from src.data.load import PROJECT_ROOT, load_config
from src.evaluate_anomaly import (
    add_combinations,
    fit_combination,
    fit_detectors,
    make_runs,
    score_frame,
    valid_hours,
    with_context,
)
from src.utils.anomaly_metrics import alarm_diagnostics, evaluate_threshold, event_f1_max_threshold
from src.utils.console import setup_console
from src.utils.results_io import resolve_path
from src.utils.seed import set_seed

# Geliştirme kümesi adı → bölme sözlüğündeki anahtar (with_context bağlamı için).
SET_KEYS = {"winter_dev": "val_dev", "summer_dev": "summer_dev"}
SCENARIOS = ("easy", "medium", "hard")


def set_config(config, set_name):
    """Kümeye özgü config: kış-dev için aynen; yaz-dev için kat 3'ü görmemiş modeller.

    Yaz-dev'de: train 2009-05-31'de biter (σ, A1, std tabloları bu kesimden), A3 tahmin
    modelleri S-F10-L48 / S-F5-L48 (F13 ağırlıkları 0,8 / 0,2 aynen), std için 2 OOF katı,
    A2 skoru S-A2 ağırlıklarıyla ve yalnız A2 (diğer AE'ler yazı görmüş ağırlıklardır).
    """
    if set_name == "winter_dev":
        return config
    config = deepcopy(config)
    dev = config["development"]
    config["split"]["train"] = {"start": config["split"]["train"]["start"], "end": dev["summer_train_end"]}
    residual = config["anomaly"]["residual"]
    residual["seq2seq"] = "S-F10-L48"
    residual["ensemble"] = {"S-F10-L48": 0.8, "S-F5-L48": 0.2}
    residual["oof_folds"] = 2
    config["anomaly"]["ae_experiments"] = ["A2"]
    config["anomaly"]["ae_checkpoints"] = {"A2": "S-A2"}
    return config


def dev_splits(df, config, set_name):
    """{"train": ..., <küme anahtarı>: ...}; yaz-dev'de train 2009-05-31'de biter."""
    if set_name == "winter_dev":
        splits = protocol_splits(df, config)
        return {"train": splits["train"], "val_dev": splits["val_dev"]}
    sets = config["development"]["sets"]["summer_dev"]
    train_end = pd.Timestamp(config["split"]["train"]["end"]) + pd.Timedelta(hours=23)
    dev_end = pd.Timestamp(sets["end"]) + pd.Timedelta(hours=23)
    train = df.loc[config["split"]["train"]["start"]:train_end].copy()
    summer = df.loc[sets["start"]:dev_end].copy()
    if not train.index.max() < summer.index.min():
        raise ValueError("Yaz-dev train kesimiyle çakışıyor.")
    return {"train": train, "summer_dev": summer}


def dev_scenario_settings(config, scenario, train):
    """Senaryonun enjeksiyon ayarları: config `injection` + tip bazlı değişiklikler.

    cap_at_train_max: spike değeri train'deki (doldurulmamış) en yüksek saatlik tüketimle
    sınırlanır; sınır o kümenin train kesiminden hesaplanır (yaz-dev'de 2009-05-31'e kadar).
    """
    target = config["preprocess"]["target"]
    settings = deepcopy(config["injection"])
    for kind, spec in config["development"]["scenarios"][scenario]["injection"].items():
        spec = dict(spec)
        if spec.pop("cap_at_train_max", False):
            spec["max_kw"] = float(train.loc[train["is_imputed"] == 0, target].max())
        settings[kind] = spec
    return settings


def prepare_set(df, config, set_name, device):
    """Bir geliştirme kümesi için config, bölmeler, fit edilmiş dedektörler, σ ve A5 referansı."""
    cfg = set_config(config, set_name)
    splits = dev_splits(df, cfg, set_name)
    key = SET_KEYS[set_name]
    fitted = fit_detectors(splits, cfg, device, std_source="oof")
    sigma_table = train_sigma_table(splits["train"], cfg["preprocess"]["target"])
    # A5 ölçek referansı: enjeksiyonsuz aynı küme (kış-dev'de final protokolle aynı kural).
    combination = fit_combination(splits, cfg, fitted, device, cfg["protocol"]["combination_components"],
                                  reference_split=key)
    return {"config": cfg, "splits": splits, "key": key, "fitted": fitted,
            "sigma_table": sigma_table, "combination": combination}


def score_set(prepared, set_name, scenario, device):
    """Bir küme × senaryonun bütün seed kopyalarını puanlar: [(seed, ham skorlar, etiketler, kopya)]."""
    cfg, splits, key = prepared["config"], prepared["splits"], prepared["key"]
    target = cfg["preprocess"]["target"]
    settings = dev_scenario_settings(cfg, scenario, splits["train"])
    frame = splits[key]
    sigma = sigma_for(frame, prepared["sigma_table"])
    outputs = []
    for seed in cfg["development"]["scenarios"][scenario]["seeds"][set_name]:
        injected, labels = inject_anomalies(frame, target, settings, seed, sigma)
        scores = score_frame(with_context(splits, key, injected, cfg), cfg, prepared["fitted"], device)
        add_combinations(scores, prepared["combination"])
        outputs.append((seed, scores[cfg["protocol"]["methods"]], labels, injected))
    return outputs


def copy_metrics(run, method, threshold):
    """Bir kopyada nokta + olay metrikleri, PR-AUC, günlük yanlış alarm ve tespit gecikmesi."""
    labels = run["labels"]["is_anomaly"].to_numpy()
    types = run["labels"]["anomaly_type"].to_numpy()
    scores = run["scores"][method].to_numpy(dtype=float)
    valid = valid_hours(run, method)
    metrics = evaluate_threshold(labels, scores, types, threshold, valid)
    with np.errstate(invalid="ignore"):
        predicted = (scores >= threshold) & valid
    alarms = alarm_diagnostics(types, predicted, int(valid.sum()))
    metrics.update({k: v for k, v in alarms.items() if k != "delays"})
    return metrics


def select_windows_thresholds(outputs_by_set, methods, config):
    """ORTA senaryo, iki kümenin kopyaları birlikte: her yöntem × pencere için olay-F1 eşiği.

    Kural (final protokolle aynı, yalnız kümeler farklı): eşik birleşik kopyalarda olay F1'i
    en büyük yapan yüzdelik; pencere ∈ {1, 3, 6} ortalama olay F1 ile (eşitlikte küçük).
    Çıktı: ({yöntem: pencere}, {yöntem: eşik}, pencere tablosu).
    """
    settings = config["protocol"]["threshold_quantiles"]
    quantiles = np.linspace(settings["low"], settings["high"], settings["n"])
    rows = []
    for window in config["anomaly"]["smoothing_windows"]:
        runs_by_set = {s: make_runs(o, {m: window for m in methods}) for s, o in outputs_by_set.items()}
        for method in methods:
            copies = [(run["labels"]["anomaly_type"].to_numpy(), run["scores"][method].to_numpy(),
                       valid_hours(run, method)) for runs_ in runs_by_set.values() for run in runs_]
            threshold, _ = event_f1_max_threshold(copies, quantiles)
            f1s = [copy_metrics(run, method, threshold)["event_f1"]
                   for runs_ in runs_by_set.values() for run in runs_]
            rows.append({"method": method, "smooth": window, "threshold": threshold,
                         "event_f1_mean": float(np.mean(f1s))})
    table = pd.DataFrame(rows).sort_values(["method", "event_f1_mean", "smooth"],
                                           ascending=[True, False, True])
    best = table.groupby("method").first()
    return best["smooth"].astype(int).to_dict(), best["threshold"].to_dict(), table


def evaluate_fixed(outputs, windows, thresholds, set_name, scenario):
    """Sabit pencere/eşiklerle kopya başına metrik tablosu."""
    runs_ = make_runs(outputs, windows)
    rows = []
    for method in windows:
        for run in runs_:
            rows.append({"set": set_name, "scenario": scenario, "seed": run["seed"], "method": method,
                         "smooth": windows[method], **copy_metrics(run, method, thresholds[method])})
    return pd.DataFrame(rows)


def summarize(per_seed):
    """Küme × senaryo × yöntem: ortalama ± std (seed'ler) ve tip bazlı olay recall'ü."""
    metrics = ["event_f1", "event_recall", "event_precision", "pr_auc", "false_alarms_per_day",
               "detection_delay_mean", "event_recall_spike", "event_recall_drop",
               "event_recall_level_shift", "event_recall_pattern"]
    grouped = per_seed.groupby(["set", "scenario", "method"])[metrics]
    summary = grouped.mean().add_suffix("_mean").join(grouped.std()[["event_f1"]].add_suffix("_std"))
    return summary.reset_index()


def combined_table(summary, scenario):
    """Seçim ölçütü: bir senaryoda kış-dev ve yaz-dev olay F1 ortalamalarının ortalaması.

    İki kümenin seed'leri farklı olduğu için eşleştirme yapılmaz; küme değerleri ayrıca gösterilir.
    """
    part = summary[summary["scenario"] == scenario]
    table = part.pivot_table(index="method", columns="set", values="event_f1_mean")
    table["both_mean"] = table[list(SET_KEYS)].mean(axis=1)
    return table.sort_values("both_mean", ascending=False).reset_index()


def check_winter_medium_equals_main(outputs, tables_dir):
    """Kış-dev orta = final protokolün val_dev ana senaryosu: kayıtlı pencere/eşiklerle birebir aynı olay F1."""
    selection = pd.read_csv(tables_dir / "anomaly_protocol_selection.csv").set_index("method")
    windows = selection["smooth"].astype(int).to_dict()
    table = evaluate_fixed(outputs, windows, selection["threshold"].to_dict(), "winter_dev", "medium")
    saved = pd.read_csv(tables_dir / "anomaly_protocol_val_per_seed.csv")
    saved = saved[(saved["split"] == "val_dev") & (saved["scenario"] == "main")]
    merged = table.merge(saved[["seed", "method", "event_f1"]], on=["seed", "method"], suffixes=("", "_saved"))
    gap = (merged["event_f1"] - merged["event_f1_saved"]).abs().max()
    if len(merged) != len(table) or gap > 1e-9:
        raise ValueError(f"Kış-dev orta senaryo final ana senaryoyla eşleşmiyor (fark {gap}).")
    return gap


def save_figure(summary, figures_dir, file_name, title):
    """Olay F1: yöntem × (küme, senaryo) çubuk grafiği."""
    pivot = summary.pivot_table(index="method", columns=["set", "scenario"], values="event_f1_mean")
    pivot = pivot.reindex(columns=[(s, c) for s in SET_KEYS for c in SCENARIOS if (s, c) in pivot.columns])
    fig, ax = plt.subplots(figsize=(12, 4.5))
    pivot.plot.bar(ax=ax, rot=20)
    ax.set_xlabel("Yöntem")
    ax.set_ylabel("Olay F1 (10 seed ortalaması)")
    ax.set_title(title)
    ax.legend([f"{s} / {c}" for s, c in pivot.columns], fontsize=8, ncol=2)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(figures_dir / file_name, dpi=200)
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
    methods = config["protocol"]["methods"]

    outputs = {}
    for set_name in SET_KEYS:
        prepared = prepare_set(df, config, set_name, device)
        for scenario in SCENARIOS:
            outputs[(set_name, scenario)] = score_set(prepared, set_name, scenario, device)
            print(f"puanlandı: {set_name} / {scenario}")

    gap = check_winter_medium_equals_main(outputs[("winter_dev", "medium")], tables_dir)
    print(f"Kış-dev orta = final val_dev ana (en büyük olay F1 farkı {gap:.1e})")

    selection_scenario = config["development"]["selection_scenario"]
    windows, thresholds, window_table = select_windows_thresholds(
        {s: outputs[(s, selection_scenario)] for s in SET_KEYS}, methods, config)
    selection = pd.DataFrame({"method": methods, "smooth": [windows[m] for m in methods],
                              "threshold": [thresholds[m] for m in methods]})
    per_seed = pd.concat([evaluate_fixed(o, windows, thresholds, s, c) for (s, c), o in outputs.items()],
                         ignore_index=True)
    summary = summarize(per_seed)

    selection.to_csv(tables_dir / "dev_baseline_selection.csv", index=False)
    window_table.to_csv(tables_dir / "dev_baseline_windows.csv", index=False)
    per_seed.to_csv(tables_dir / "dev_baseline_per_seed.csv", index=False)
    summary.to_csv(tables_dir / "dev_baseline_summary.csv", index=False)
    combined = combined_table(summary, selection_scenario)
    combined.to_csv(tables_dir / "dev_baseline_combined_medium.csv", index=False)
    save_figure(summary, figures_dir, "D2_dev_baseline_event_f1.png",
                "Mevcut yöntemler, yeni geliştirme protokolü – kış-dev ve yaz-dev (val_hold/test yok), "
                "eşik orta senaryoda")

    columns = ["set", "scenario", "method", "event_f1_mean", "event_f1_std", "pr_auc_mean",
               "false_alarms_per_day_mean", "detection_delay_mean_mean"]
    print(selection.to_string(index=False))
    print(summary[columns].round(3).to_string(index=False))
    print(combined.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
