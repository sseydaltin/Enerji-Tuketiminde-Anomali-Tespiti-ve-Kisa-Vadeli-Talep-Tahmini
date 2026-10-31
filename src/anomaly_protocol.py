"""Final anomali protokolü: train/OOF → val-dev (seçim) → val-hold (bağımsız ölçüm) → test.

1. Train: dedektörler ve tahmin modelleri yalnız train ile; A3'ün normalize skor
   std'si OOF (kat dışı) artıklarından (fit_detectors, std_source="oof").
2. val-dev (2009-10 – 2009-12): ANA senaryonun 10 enjekte kopyasında her yöntem
   için yumuşatma penceresi ve olay-F1 eşiği seçilir; final yöntem buradaki olay
   F1'iyle seçilir (fark gürültü düzeyindeyse en basit olan).
3. val-hold (2010-01 – 2010-03): seçilen pencere ve eşikler aynen uygulanır;
   bu sonuca göre HİÇBİR ŞEY değiştirilmez. Zor senaryo aynı eşiklerle ayrıca raporlanır.
4. test: yalnız kullanıcı "final değerlendirme" dediğinde (--split test).
Ayrıntı ve gerekçe: docs/final_test_protokolu.md.
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

from src.data.inject_anomalies import ANOMALY_TYPES, inject_anomalies, sigma_for, train_sigma_table
from src.data.load import PROJECT_ROOT, load_config
from src.data.split import chronological_split
from src.evaluate_anomaly import (
    METHOD_LABELS,
    add_combinations,
    fit_combination,
    fit_detectors,
    make_runs,
    paired_differences,
    smooth_scores,
    score_frame,
    summarize,
    valid_hours,
    with_context,
)
from src.models.residual_detector import combine_predictions, one_step_predictions, residual_std_table
from src.utils.anomaly_metrics import evaluate_threshold, event_f1_max_threshold, runs
from src.utils.console import setup_console
from src.utils.results_io import resolve_path, upsert_csv
from src.utils.seed import set_seed

RULE = "evf1max"   # eşik kuralı: birleşik olay F1'inin maksimumu
PROTOCOL_LABELS = {
    "A3-s2s-norm": "Tahmin artığı F10-opt, |h=1 hatası| / OOF std saat×gün",
    "A3-ens-norm": "Tahmin artığı F13-ens, |h=1 hatası| / OOF std saat×gün",
    "A3-s2s-hmean": "Çok ufuklu artık F10-opt, h=1…24 ort. (ufuk başına OOF std)",
    "A5-mean-z": "A0 + A3-ens-norm, z ölçeğinde ortalama (referans: temiz val-dev)",
}


def protocol_splits(df, config):
    """Kronolojik bölmelere val_dev ve val_hold'u ekler (val'in iki ardışık parçası).

    Çıktı: {"train", "val", "test", "val_dev", "val_hold"}.
    Neden: Seçim (pencere, eşik, yöntem) val_dev'de; val_hold yalnız ölçüm için.
    İki parça kesişmez, aralarında boşluk yoktur ve birlikte val'i tam kaplar.
    """
    splits = chronological_split(df, config)
    val = splits["val"]
    for name in ("val_dev", "val_hold"):
        settings = config["split"][name]
        end = pd.Timestamp(settings["end"]) + pd.Timedelta(hours=23)
        splits[name] = val.loc[settings["start"]:end].copy()
    dev, hold = splits["val_dev"], splits["val_hold"]
    if not dev.index.max() < hold.index.min():
        raise ValueError("val_dev ve val_hold çakışıyor.")
    if len(dev) + len(hold) != len(val):
        raise ValueError("val_dev + val_hold val'i tam kaplamıyor.")
    return splits


def scenario_settings(config, scenario):
    """Senaryonun enjeksiyon ayarları: config `injection` + senaryonun tip bazlı değişiklikleri."""
    settings = deepcopy(config["injection"])
    for kind, spec in config["protocol"]["scenarios"][scenario]["injection"].items():
        settings[kind] = dict(spec)        # tipin ayarı bütünüyle değiştirilir
    return settings


def score_copies(splits, config, split, scenario, fitted, device, combination, sigma_table):
    """Bir küme × senaryo için tüm seed'lerin enjekte kopyalarını puanlar.

    Çıktı: [(seed, skorlar [bağlam + küme, yöntem], etiketler, enjekte kopya)].
    Neden: Her seed ayrı bir kopya üretir; orijinal küme değişmez. Bağlam bir
    önceki kümeden ENJEKSİYONSUZ alınır (val_hold için temiz val_dev).
    """
    target = config["preprocess"]["target"]
    methods = config["protocol"]["methods"]
    settings = scenario_settings(config, scenario)
    frame = splits[split]
    sigma = sigma_for(frame, sigma_table)
    outputs = []
    for seed in config["protocol"]["scenarios"][scenario]["seeds"][split]:
        injected, labels = inject_anomalies(frame, target, settings, seed, sigma)
        scores = score_frame(with_context(splits, split, injected, config), config, fitted, device)
        add_combinations(scores, combination)
        outputs.append((seed, scores[methods], labels, injected))
    return outputs


def per_seed_rows(runs, method, threshold, split, scenario, window):
    """Bir yöntem için kopya başına nokta + olay metrikleri (sabit eşik ve pencere)."""
    rows = []
    for run in runs:
        metrics = evaluate_threshold(
            run["labels"]["is_anomaly"].to_numpy(), run["scores"][method].to_numpy(),
            run["labels"]["anomaly_type"].to_numpy(), threshold, valid_hours(run, method),
        )
        rows.append({"split": split, "scenario": scenario, "seed": run["seed"], "method": method,
                     "label": PROTOCOL_LABELS.get(method, METHOD_LABELS.get(method, method)),
                     "rule": RULE, "smooth": window, **metrics})
    return rows


def select_on_dev(outputs, config):
    """val_dev (ana senaryo): her yöntem × pencere için olay-F1 eşiği; pencere olay F1 ile seçilir.

    Çıktı: (tüm pencerelerin kopya başına tablosu, {yöntem: pencere}, {yöntem: eşik}).
    Neden: Eşik 10 kopyanın birleşiminde olay F1'i en büyük yapan değer; pencere,
    o eşikteki ortalama olay F1'i en büyük olan (eşitlikte küçük pencere).
    """
    settings = config["protocol"]["threshold_quantiles"]
    quantiles = np.linspace(settings["low"], settings["high"], settings["n"])
    rows = []
    for window in config["anomaly"]["smoothing_windows"]:
        runs = make_runs(outputs, {method: window for method in config["protocol"]["methods"]})
        for method in config["protocol"]["methods"]:
            copies = [(run["labels"]["anomaly_type"].to_numpy(), run["scores"][method].to_numpy(),
                       valid_hours(run, method)) for run in runs]
            threshold, _ = event_f1_max_threshold(copies, quantiles)
            rows += per_seed_rows(runs, method, threshold, "val_dev", "main", window)
    per_seed = pd.DataFrame(rows)
    means = per_seed.groupby(["method", "smooth"])["event_f1"].mean().reset_index()
    means = means.sort_values(["method", "event_f1", "smooth"], ascending=[True, False, True])
    windows = means.groupby("method")["smooth"].first().astype(int).to_dict()
    chosen = per_seed[per_seed["smooth"] == per_seed["method"].map(windows)]
    thresholds = chosen.groupby("method")["threshold"].first().to_dict()
    return per_seed, windows, thresholds


def apply_fixed(outputs, windows, thresholds, split, scenario):
    """Seçilmiş pencere ve eşikleri bir kümeye/senaryoya aynen uygular (seçim yok)."""
    runs = make_runs(outputs, windows)
    rows = []
    for method in windows:
        rows += per_seed_rows(runs, method, thresholds[method], split, scenario, windows[method])
    return pd.DataFrame(rows)


def choose_final(per_seed, order, se_multiplier):
    """Final yöntem: val_dev ana senaryo olay F1'i; gürültü düzeyindekiler arasından en basiti.

    Kural (revizyon 2026-10-31): ortalama olay F1'i en yüksek yöntem referanstır;
    ona karşı eşleştirilmiş seed farkı için |ort| ≤ k · std / √n ise (k = 2, n =
    seed sayısı; standart hata tabanlı) fark gürültü düzeyindedir. Referans +
    bunlar arasından `order`'da (basitlik sırası) en önde olan seçilir.
    Neden: Eski ölçüt |ort| ≤ std seed sayısı arttıkça gevşer (n=10'da t ≲ 3,2).
    Çıktı: (final yöntem, referans yöntem, eşleştirilmiş fark tablosu).
    """
    means = per_seed.groupby("method")["event_f1"].mean().sort_values(ascending=False)
    best = means.index[0]
    n_seeds = per_seed.loc[per_seed["method"] == best, "seed"].nunique()
    paired = paired_differences(per_seed, best, metrics=("event_f1",), rule=RULE)
    paired["noise_bound"] = se_multiplier * paired["event_f1_diff_std"] / np.sqrt(n_seeds)
    paired["noise_level"] = paired["event_f1_diff_mean"].abs() <= paired["noise_bound"]
    candidates = [best] + paired.loc[paired["noise_level"], "method"].tolist()
    return min(candidates, key=order.index), best, paired


def std_comparison(splits, config, fitted):
    """In-sample ve OOF artık std'lerinin saat bazında karşılaştırması (gün ortalaması).

    Sütunlar: tüm train in-sample, OOF dönemiyle sınırlı in-sample, OOF ve oran.
    Neden: OOF dönemi train'in yalnız son 12 ayı; aynı dönemle sınırlanmış
    in-sample sütunu, farkın dönemden değil OOF'tan geldiğini gösterir.
    """
    target = config["preprocess"]["target"]
    residual = config["anomaly"]["residual"]
    train = splits["train"]
    predictions = fitted["train_predictions"]
    in_sample = {"A3-s2s": predictions[residual["seq2seq"]],
                 "A3-ens": combine_predictions(predictions, residual["ensemble"])}
    oof_start = train.index[~np.isnan(fitted["oof_predictions"][residual["seq2seq"]].to_numpy())][0]
    rows = []
    for name in ("A3-s2s", "A3-ens"):
        tables = {
            "in_sample_full": fitted["in_sample_std_tables"][name],
            "in_sample_oof_period": residual_std_table(train.loc[oof_start:],
                                                       in_sample[name].loc[oof_start:], target),
            "oof": fitted["std_tables"][name],
        }
        hourly = pd.DataFrame({key: table.groupby("hour")["std"].mean()
                               for key, table in tables.items()})
        hourly["ratio_oof_to_in_sample_period"] = hourly["oof"] / hourly["in_sample_oof_period"]
        rows.append(hourly.reset_index().assign(model=name))
    return pd.concat(rows, ignore_index=True), oof_start


def save_std_figure(comparison, oof_start, figures_dir):
    """In-sample vs OOF std (saat bazında, 7 günün ortalaması) — TRAIN artıkları."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for axis, (name, part) in zip(axes, comparison.groupby("model", sort=False)):
        axis.plot(part["hour"], part["in_sample_full"], marker="o", label="In-sample, tüm train")
        axis.plot(part["hour"], part["in_sample_oof_period"], marker="s",
                  label=f"In-sample, {oof_start:%Y-%m} – 2009-09")
        axis.plot(part["hour"], part["oof"], marker="^", label="OOF (3 kat)")
        axis.set_xlabel("Saat")
        axis.set_title(f"{name}: h=1 artık std'si – TRAIN (gün ortalaması)")
        axis.set_xticks(range(0, 24, 3))
    axes[0].set_ylabel("Artık std'si (kW)")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figures_dir / "P_train_residual_std_in_sample_vs_oof.png", dpi=200)
    plt.close(figure)


def side_by_side(summaries):
    """Yöntem × (küme, senaryo) tablosu: olay F1/R/P, nokta F1, PR-AUC (ort ± std)."""
    rows = []
    for method in summaries[0]["method"]:
        row = {"method": method}
        for summary in summaries:
            part = summary.set_index("method").loc[method]
            tag = f"{part['split']}_{part['scenario']}"
            for metric in ("event_f1", "event_recall", "event_precision", "f1", "pr_auc"):
                row[f"{tag}_{metric}"] = f"{part[metric + '_mean']:.3f}±{part[metric + '_std']:.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def write_markdown(frame, path):
    """DataFrame'i basit bir Markdown tablosu olarak yazar."""
    table = frame.astype(str)
    lines = ["| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
    lines += ["| " + " | ".join(row) + " |" for row in table.to_numpy()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_event_f1_figure(summaries, final, figures_dir):
    """Olay F1 (ort ± std) — val_dev / val_hold × ana / zor, aynı eşikler."""
    methods = list(summaries[0]["method"])
    width = 0.8 / len(summaries)
    figure, axis = plt.subplots(figsize=(12, 4.8))
    for k, summary in enumerate(summaries):
        part = summary.set_index("method").loc[methods]
        tag = f"{part['split'].iloc[0]} / {part['scenario'].iloc[0]}"
        axis.bar(np.arange(len(methods)) + k * width, part["event_f1_mean"], width,
                 yerr=part["event_f1_std"], capsize=2, label=tag)
    axis.set_xticks(np.arange(len(methods)) + width * (len(summaries) - 1) / 2,
                    [f"{m}{' (final)' if m == final else ''}" for m in methods], fontsize=8)
    axis.set_ylabel("Olay F1, ort ± std (10 seed)")
    axis.set_title("Olay F1 – VAL-DEV (seçim) ve VAL-HOLD (bağımsız), ana ve zor senaryo; "
                   "eşik ve pencere val-dev ana senaryodan")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figures_dir / "P_val_event_f1_dev_vs_hold.png", dpi=200)
    plt.close(figure)


def top_clean_events(splits, config, fitted, device, combination, method, window, threshold, k=10):
    """Enjeksiyonsuz test verisinde final yöntemin en yüksek skorlu k alarm bloğu.

    Çıktı: [başlangıç, bitiş, süre (saat), en yüksek skor, gerçek ve h=1 tahmini
    ortalama kW, gün adı, tatil] tablosu. Neden: Gerçek etiket yok; bu bloklar
    nitel olarak incelenir (açıklamalar hipotezdir, doğrulanmış değildir).
    """
    target = config["preprocess"]["target"]
    test = splits["test"]
    scores = score_frame(with_context(splits, "test", test, config), config, fitted, device)
    add_combinations(scores, combination)
    smoothed = smooth_scores(scores[[method]], window)[method].loc[test.index]
    model, checkpoint = fitted["forecasters"][config["anomaly"]["residual"]["seq2seq"]]
    predicted = one_step_predictions(model, checkpoint, with_context(splits, "test", test, config),
                                     device).loc[test.index]
    flagged = (smoothed >= threshold).to_numpy() & (test["is_imputed"].to_numpy() == 0)
    rows = []
    for start, end in runs(flagged):
        hours = test.index[start:end]
        rows.append({
            "start": hours[0], "end": hours[-1], "hours": end - start,
            "max_score": smoothed.iloc[start:end].max(),
            "actual_kw_mean": test[target].iloc[start:end].mean(),
            "predicted_h1_kw_mean": predicted.iloc[start:end].mean(),
            "weekday": hours[0].day_name(), "is_holiday": int(test["is_holiday"].iloc[start:end].max()),
        })
    table = pd.DataFrame(rows).sort_values("max_score", ascending=False)
    return table.head(k).reset_index(drop=True), len(table)


def run_test(splits, config, fitted, device, combination, sigma_table, tables_dir):
    """Final değerlendirme: pencere, eşik ve final yöntem val_dev tablosundan; test'te seçim yok."""
    chosen = pd.read_csv(tables_dir / "anomaly_protocol_selection.csv")
    windows = dict(zip(chosen["method"], chosen["smooth"].astype(int)))
    thresholds = dict(zip(chosen["method"], chosen["threshold"]))
    final = chosen.loc[chosen["final"], "method"].iloc[0]
    results, paired_rows = [], []
    for scenario in ("main", "hard"):
        outputs = score_copies(splits, config, "test", scenario, fitted, device,
                               combination, sigma_table)
        table = apply_fixed(outputs, windows, thresholds, "test", scenario)
        results.append(table)
        paired = paired_differences(table, "A0", metrics=("event_f1", "f1", "pr_auc", "event_recall"),
                                    rule=RULE)
        paired_rows.append(paired.assign(split="test", scenario=scenario))
    per_seed = pd.concat(results, ignore_index=True)
    per_seed.to_csv(tables_dir / "anomaly_protocol_test_per_seed.csv", index=False)
    summary = summarize(per_seed, ("split", "scenario", "method", "rule", "smooth"))
    summary["final"] = summary["method"] == final
    summary.to_csv(tables_dir / "anomaly_protocol_test.csv", index=False)
    pd.concat(paired_rows, ignore_index=True).to_csv(
        tables_dir / "anomaly_protocol_test_paired_vs_A0.csv", index=False)
    events, n_blocks = top_clean_events(splits, config, fitted, device, combination, final,
                                        windows[final], thresholds[final])
    events.to_csv(tables_dir / "anomaly_test_top_events_clean.csv", index=False)
    print(summary[["scenario", "method", "final", "event_f1_mean", "event_f1_std", "f1_mean",
                   "pr_auc_mean"]].round(3).to_string(index=False))
    print(f"\nEnjeksiyonsuz testte {final} alarm bloğu: {n_blocks}")
    print(events.round(3).to_string(index=False))


def main():
    """Varsayılan: val_dev seçimi + val_hold ölçümü. --split test yalnız final değerlendirmede."""
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
    target = config["preprocess"]["target"]
    protocol = config["protocol"]

    splits = protocol_splits(df, config)
    fitted = fit_detectors(splits, config, device, std_source=protocol["std_source"])
    sigma_table = train_sigma_table(splits["train"], target)        # yalnız train
    combination = fit_combination(splits, config, fitted, device,
                                  protocol["combination_components"], reference_split="val_dev")

    if args.split == "test":
        run_test(splits, config, fitted, device, combination, sigma_table, tables_dir)
        return

    # 1. Std karşılaştırması (train; in-sample vs OOF).
    comparison, oof_start = std_comparison(splits, config, fitted)
    comparison.to_csv(tables_dir / "anomaly_std_in_sample_vs_oof.csv", index=False)
    save_std_figure(comparison, oof_start, figures_dir)

    # 2. val_dev ana senaryo: seçim.
    dev_main = score_copies(splits, config, "val_dev", "main", fitted, device, combination, sigma_table)
    all_windows, windows, thresholds = select_on_dev(dev_main, config)
    all_windows.to_csv(tables_dir / "anomaly_protocol_dev_windows_per_seed.csv", index=False)
    per_seed = {("val_dev", "main"): all_windows[all_windows["smooth"]
                                                 == all_windows["method"].map(windows)]}
    final, best, paired_best = choose_final(per_seed[("val_dev", "main")], protocol["simplicity_order"],
                                             protocol["noise_se_multiplier"])
    selection = pd.DataFrame({"method": list(windows), "smooth": list(windows.values()),
                              "threshold": [thresholds[m] for m in windows]})
    selection["final"] = selection["method"] == final
    selection.to_csv(tables_dir / "anomaly_protocol_selection.csv", index=False)
    paired_best.to_csv(tables_dir / "anomaly_protocol_dev_paired_vs_best.csv", index=False)

    # 3. Aynı pencere ve eşiklerle: val_dev zor, val_hold ana ve zor (seçim yok).
    for split, scenario in (("val_dev", "hard"), ("val_hold", "main"), ("val_hold", "hard")):
        outputs = score_copies(splits, config, split, scenario, fitted, device, combination, sigma_table)
        per_seed[(split, scenario)] = apply_fixed(outputs, windows, thresholds, split, scenario)

    summaries, paired_rows = [], []
    for (split, scenario), table in per_seed.items():
        summaries.append(summarize(table, ("split", "scenario", "method", "rule", "smooth")))
        paired = paired_differences(table, "A0", metrics=("event_f1", "f1", "pr_auc", "event_recall"),
                                    rule=RULE)
        paired_rows.append(paired.assign(split=split, scenario=scenario))
    all_per_seed = pd.concat(per_seed.values(), ignore_index=True)
    all_per_seed.to_csv(tables_dir / "anomaly_protocol_val_per_seed.csv", index=False)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(tables_dir / "anomaly_protocol_val.csv", index=False)
    pd.concat(paired_rows, ignore_index=True).to_csv(
        tables_dir / "anomaly_protocol_val_paired_vs_A0.csv", index=False)
    table = side_by_side(summaries)
    table.to_csv(tables_dir / "anomaly_protocol_side_by_side.csv", index=False)
    write_markdown(table, tables_dir / "anomaly_protocol_side_by_side.md")
    save_event_f1_figure(summaries, final, figures_dir)

    rows = []
    for method in protocol["methods"]:
        parts = []
        for part in summaries:
            r = part.set_index("method").loc[method]
            parts.append(f"{r['split']}/{r['scenario']} olay F1={r['event_f1_mean']:.3f}±"
                         f"{r['event_f1_std']:.3f}, nokta F1={r['f1_mean']:.3f}, PR-AUC={r['pr_auc_mean']:.3f}")
        rows.append({
            "exp_id": f"P-{method}", "date": date.today().isoformat(),
            "model": PROTOCOL_LABELS.get(method, METHOD_LABELS.get(method, method)),
            "seed": config["seed"], "lookback": windows[method],
            "val_mae": "", "val_rmse": "", "test_mae": "", "test_rmse": "",
            "note": (f"Final protokol (OOF std; val_dev seçim, val_hold bağımsız); w={windows[method]}, "
                     f"olay-F1 eşiği={thresholds[method]:.4g} (val_dev ana); " + "; ".join(parts)
                     + ("; FINAL YÖNTEM" if method == final else "")),
        })
    upsert_csv(resolve_path(PROJECT_ROOT, config["paths"]["experiments_file"]), rows, ["exp_id"])

    columns = ["split", "scenario", "method", "smooth", "threshold", "event_f1_mean", "event_f1_std",
               "event_recall_mean", "event_precision_mean", "f1_mean", "pr_auc_mean", "roc_auc_mean"]
    print(summary[columns].round(3).to_string(index=False))
    print(f"\nReferans (en yüksek val_dev olay F1): {best}; FINAL: {final}")
    print(paired_best.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
