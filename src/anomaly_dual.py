"""A6: test sonrası geliştirme denemesi (FİNAL SONUÇ DEĞİLDİR; test kümesi kullanılmaz).

A6-dual       : yumuşatılmış A3-s2s-norm ≥ T1 VEYA ham A3-s2s-norm ≥ T2.
A6-dual-floor : A6-dual VEYA tüketim, train'in 0,5. yüzdeliğinin altında en az k saat.

--stage select : T2 ve k yalnız val_dev ana senaryoda, birleşik olay F1 ile seçilir.
--stage hold   : seçilenler val_hold'da BİR KEZ ölçülür; sonuç dosyası varsa çalışmaz.

Neden: Yumuşatma 1–2 saatlik sıçramaları bastırıyor; ham skor kolu bunu, taban kuralı da
modelin göremediği ani düşüşleri hedefler. Bedeli yanlış alarmdır; ayrıca raporlanır.
"""

import argparse
from datetime import date

import numpy as np
import pandas as pd
import torch

from src.anomaly_protocol import protocol_splits, score_copies
from src.data.inject_anomalies import ANOMALY_TYPES, train_sigma_table
from src.data.load import PROJECT_ROOT, load_config
from src.evaluate_anomaly import fit_combination, fit_detectors, make_runs, valid_hours
from src.utils.anomaly_metrics import event_metrics
from src.utils.console import setup_console
from src.utils.results_io import resolve_path
from src.utils.seed import set_seed

BASE_METHOD = "A3-s2s-norm"
REFERENCE_METHOD = "A5-mean-z"
DUAL = "A6-dual"
DUAL_FLOOR = "A6-dual-floor"
STAGES = {"select": "val_dev", "hold": "val_hold"}     # test bilerek yok
SELECTION_FILE = "a6_selection.csv"


def floor_threshold(train, target, percentile):
    """Taban eşiği: yalnız train'in doldurulmamış saatlerinden yüzdelik (kW)."""
    values = train.loc[train["is_imputed"].to_numpy() == 0, target].to_numpy(dtype=np.float64)
    return float(np.percentile(values, percentile))


def floor_alarm(values, floor, min_hours):
    """Tüketim tabanın altında en az min_hours saattir sürüyorsa True döndürür.

    Girdi: values [N] kW, taban (kW), k saat. Çıktı: [N] bool.
    Neden: Yalnız geçmişe bakar: t anındaki karar t−k+1 … t saatlerine dayanır,
    bu yüzden alarm k. saatte başlar. Gelecek bilgisi kullanılmaz.
    """
    below = np.asarray(values, dtype=np.float64) < floor
    run_length = np.zeros(len(below), dtype=int)
    current = 0
    for index, is_below in enumerate(below):
        current = current + 1 if is_below else 0
        run_length[index] = current
    return run_length >= min_hours


def dual_alarm(smoothed, raw, t1, t2):
    """A6-dual kuralı: yumuşatılmış ≥ T1 veya ham ≥ T2. Girdi/çıktı: [N]. NaN alarm üretmez."""
    with np.errstate(invalid="ignore"):
        return (np.asarray(smoothed) >= t1) | (np.asarray(raw) >= t2)


def build_copies(outputs, config, windows):
    """score_copies çıktısını seçim ve ölçümün kullandığı dizilere çevirir.

    Çıktı: kopya başına seed, tipler, A3 ve A5'in yumuşatılmış skorları, A3 ham skoru,
    enjekte tüketim ve geçerli saat maskeleri. Hepsi [N], küme saatleriyle hizalı.
    """
    target = config["preprocess"]["target"]
    runs = make_runs(outputs, windows)
    copies = []
    for (seed, scores, labels, injected), run in zip(outputs, runs):
        copies.append({
            "seed": seed,
            "types": labels["anomaly_type"].to_numpy(),
            "base_smoothed": run["scores"][BASE_METHOD].to_numpy(),
            "base_raw": scores[BASE_METHOD].loc[injected.index].to_numpy(),
            "reference_smoothed": run["scores"][REFERENCE_METHOD].to_numpy(),
            "values": injected[target].to_numpy(dtype=np.float64),
            "base_valid": valid_hours(run, BASE_METHOD),
            "reference_valid": valid_hours(run, REFERENCE_METHOD),
        })
    return copies


def method_flags(copy, method, settings):
    """Bir kopyada bir yöntemin alarm saatlerini [N] bool olarak hesaplar.

    Doldurulmuş saatler hiçbir kuralda alarm üretmez (final protokolle aynı).
    """
    if method == REFERENCE_METHOD:
        return (copy["reference_smoothed"] >= settings["t1"][method]) & copy["reference_valid"]
    flags = copy["base_smoothed"] >= settings["t1"][BASE_METHOD]
    if method in (DUAL, DUAL_FLOOR):
        flags = dual_alarm(copy["base_smoothed"], copy["base_raw"], settings["t1"][BASE_METHOD], settings[method]["t2"])
    if method == DUAL_FLOOR:
        flags = flags | floor_alarm(copy["values"], settings["floor_kw"], settings[method]["k"])
    return flags & copy["base_valid"]


def pooled_event_f1(pairs):
    """Birleşik olay F1: yakalananlar ve doğru bloklar kopyalar üzerinden toplanır.

    Girdi: [(tipler [N], alarm [N]), ...]. Neden: Final eşik seçimiyle aynı ölçüt
    (src.utils.anomaly_metrics.event_f1_max_threshold).
    """
    detected = n_events = correct = n_blocks = 0.0
    for types, flags in pairs:
        metrics = event_metrics(types, flags)
        if metrics["n_events"]:
            n_events += metrics["n_events"]
            detected += metrics["event_recall"] * metrics["n_events"]
        n_blocks += metrics["n_pred_blocks"]
        correct += metrics["event_precision"] * metrics["n_pred_blocks"]
    recall = detected / n_events if n_events else 0.0
    precision = correct / n_blocks if n_blocks else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def t2_candidates(copies, quantile_settings):
    """T2 ızgarası: birleşik ham skorun yüzdelikleri (final eşik ızgarasıyla aynı aralık)."""
    pooled = np.concatenate([copy["base_raw"][copy["base_valid"]] for copy in copies])
    quantiles = np.linspace(quantile_settings["low"], quantile_settings["high"], quantile_settings["n"])
    return np.unique(np.quantile(pooled, quantiles))


def select_settings(copies, t1, floor_kw, candidates, k_values):
    """T2 ve k'yı val_dev kopyalarında birleşik olay F1 ile seçer.

    Çıktı: {DUAL: {t2, f1}, DUAL_FLOOR: {t2, k, f1}}.
    Neden: Eşitlikte yüksek T2 ve büyük k seçilir, yani daha az alarm (final kuralıyla aynı).
    """
    base = [dict(copy, smoothed_flags=copy["base_smoothed"] >= t1) for copy in copies]
    best_dual = {"t2": float(candidates[-1]), "f1": -1.0}
    best_floor = {"t2": float(candidates[-1]), "k": max(k_values), "f1": -1.0}
    floors = {k: [floor_alarm(copy["values"], floor_kw, k) for copy in copies] for k in k_values}
    for t2 in candidates:
        dual = [(copy["smoothed_flags"] | (copy["base_raw"] >= t2)) & copy["base_valid"] for copy in base]
        f1 = pooled_event_f1([(copy["types"], flags) for copy, flags in zip(base, dual)])
        if f1 >= best_dual["f1"]:
            best_dual = {"t2": float(t2), "f1": f1}
        for k in sorted(k_values):
            combined = [
                flags | (floor & copy["base_valid"])
                for copy, flags, floor in zip(base, dual, floors[k])
            ]
            f1 = pooled_event_f1([(copy["types"], flags) for copy, flags in zip(base, combined)])
            if f1 >= best_floor["f1"]:
                best_floor = {"t2": float(t2), "k": int(k), "f1": f1}
    return {DUAL: best_dual, DUAL_FLOOR: best_floor}


def evaluate_methods(copies, settings, split):
    """Dört yöntemin kopya başına olay metrikleri ve yanlış alarm sayısı."""
    rows = []
    for method in (BASE_METHOD, REFERENCE_METHOD, DUAL, DUAL_FLOOR):
        for copy in copies:
            metrics = event_metrics(copy["types"], method_flags(copy, method, settings))
            blocks = metrics["n_pred_blocks"]
            row = {
                "split": split, "scenario": "main", "seed": copy["seed"], "method": method,
                "event_f1": metrics["event_f1"], "event_recall": metrics["event_recall"],
                "event_precision": metrics["event_precision"], "n_events": metrics["n_events"],
                "n_alarms": blocks,
                "false_alarms": blocks - round(metrics["event_precision"] * blocks),
            }
            for kind in ANOMALY_TYPES:
                row[f"event_recall_{kind}"] = metrics[f"event_recall_{kind}"]
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_methods(per_seed):
    """Kopya başına tablodan ortalama ± std ve A3'e göre yanlış alarm artışı üretir."""
    columns = ["event_f1", "event_recall", "event_precision", *(f"event_recall_{k}" for k in ANOMALY_TYPES),
               "n_alarms", "false_alarms"]
    rows = []
    base = per_seed[per_seed["method"] == BASE_METHOD].set_index("seed")["false_alarms"]
    for (split, method), group in per_seed.groupby(["split", "method"], sort=False):
        row = {"split": split, "method": method, "n_seeds": len(group)}
        for column in columns:
            row[f"{column}_mean"] = group[column].mean()
            row[f"{column}_std"] = group[column].std(ddof=1)
        difference = group.set_index("seed")["false_alarms"] - base
        row["false_alarm_increase_vs_A3_mean"] = difference.mean()
        row["false_alarm_increase_vs_A3_std"] = difference.std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def check_against_saved(per_seed, tables_dir, split):
    """A3 ve A5 sonuçları kayıtlı final protokol değerleriyle aynı olmalı; değilse durur.

    Neden: Karşılaştırmanın temeli protokolden sapmışsa A6 sayıları yorumlanamaz.
    """
    saved = pd.read_csv(tables_dir / "anomaly_protocol_val_per_seed.csv")
    saved = saved[(saved["split"] == split) & (saved["scenario"] == "main")]
    for method in (BASE_METHOD, REFERENCE_METHOD):
        ours = per_seed[per_seed["method"] == method].set_index("seed")["event_f1"].sort_index()
        theirs = saved[saved["method"] == method].set_index("seed")["event_f1"].sort_index()
        if not np.allclose(ours.to_numpy(), theirs.to_numpy(), atol=1e-6):
            raise RuntimeError(f"{split}/{method}: kayıtlı protokol sonucuyla uyuşmuyor.")


def base_settings(config, tables_dir, train):
    """Mevcut eşikleri, pencereleri ve train'den taban eşiğini okur (seçim yapılmaz)."""
    chosen = pd.read_csv(tables_dir / "anomaly_protocol_selection.csv").set_index("method")
    methods = [BASE_METHOD, REFERENCE_METHOD]
    return {
        "windows": chosen.loc[methods, "smooth"].astype(int).to_dict(),
        "t1": chosen.loc[methods, "threshold"].astype(float).to_dict(),
        "floor_kw": floor_threshold(train, config["preprocess"]["target"],
                                    config["protocol"]["a6"]["floor_percentile"]),
    }


def score_split(config, stage):
    """Seçilen doğrulama parçası için ana senaryo kopyalarını protokolle puanlar.

    Girdi: config, "select" | "hold". Test kümesine giden kod yolu yoktur.
    """
    split = STAGES[stage]
    device = torch.device("cpu")
    frame = pd.read_parquet(resolve_path(PROJECT_ROOT, config["paths"]["processed_file"]))
    splits = protocol_splits(frame, config)
    protocol = config["protocol"]
    fitted = fit_detectors(splits, config, device, std_source=protocol["std_source"])
    combination = fit_combination(splits, config, fitted, device,
                                  protocol["combination_components"], reference_split="val_dev")
    sigma_table = train_sigma_table(splits["train"], config["preprocess"]["target"])
    return splits, score_copies(splits, config, split, "main", fitted, device, combination, sigma_table)


def run_select(config, tables_dir):
    """T2 ve k'yı val_dev'de seçer, seçimi ve val_dev tablosunu yazar."""
    splits, outputs = score_split(config, "select")
    settings = base_settings(config, tables_dir, splits["train"])
    copies = build_copies(outputs, config, settings["windows"])
    a6 = config["protocol"]["a6"]
    candidates = t2_candidates(copies, config["protocol"]["threshold_quantiles"])
    settings.update(select_settings(copies, settings["t1"][BASE_METHOD], settings["floor_kw"],
                                    candidates, a6["k_values"]))
    # Demoda tek deneysel varyant: val_dev'de iyi olan; eşitlikte daha basit olan.
    demo = DUAL_FLOOR if settings[DUAL_FLOOR]["f1"] > settings[DUAL]["f1"] else DUAL
    pd.DataFrame([
        {"method": name, "window": settings["windows"][BASE_METHOD], "t1": settings["t1"][BASE_METHOD],
         "t2": settings[name]["t2"], "k": settings[name].get("k", ""), "floor_kw": settings["floor_kw"],
         "val_dev_pooled_event_f1": settings[name]["f1"], "demo": name == demo}
        for name in (DUAL, DUAL_FLOOR)
    ]).to_csv(tables_dir / SELECTION_FILE, index=False)
    per_seed = evaluate_methods(copies, settings, "val_dev")
    check_against_saved(per_seed, tables_dir, "val_dev")
    per_seed.to_csv(tables_dir / "a6_val_dev_per_seed.csv", index=False)
    return settings, per_seed


def load_selection(tables_dir, train, config):
    """Kayıtlı seçimi okur; val_hold ölçümü seçim yapmaz."""
    settings = base_settings(config, tables_dir, train)
    chosen = pd.read_csv(tables_dir / SELECTION_FILE).set_index("method")
    settings[DUAL] = {"t2": float(chosen.loc[DUAL, "t2"])}
    settings[DUAL_FLOOR] = {"t2": float(chosen.loc[DUAL_FLOOR, "t2"]), "k": int(chosen.loc[DUAL_FLOOR, "k"])}
    return settings


def run_hold(config, tables_dir):
    """Seçilmiş ayarları val_hold'da bir kez ölçer; sonuç dosyası varsa reddeder."""
    output = tables_dir / "a6_val_hold_per_seed.csv"
    if output.exists():
        raise SystemExit(f"{output.name} zaten var: val_hold bir kez ölçülür.")
    if not (tables_dir / SELECTION_FILE).exists():
        raise SystemExit("Önce --stage select çalıştırılmalı.")
    splits, outputs = score_split(config, "hold")
    settings = load_selection(tables_dir, splits["train"], config)
    copies = build_copies(outputs, config, settings["windows"])
    per_seed = evaluate_methods(copies, settings, "val_hold")
    check_against_saved(per_seed, tables_dir, "val_hold")
    per_seed.to_csv(output, index=False)
    return per_seed


def write_experiment_rows(config, summary):
    """experiments.csv sonuna iki A6 satırı ekler; test sütunları boş kalır.

    Neden: Dosya baştan yazılırsa (upsert) mevcut final kayıtlarının son ondalıkları
    yuvarlanarak değişir. Final kayıtlara dokunulmaması için yalnız ekleme yapılır.
    """
    rows = []
    for method in (DUAL, DUAL_FLOOR):
        parts = summary[summary["method"] == method]
        notes = "; ".join(
            f"{row['split']} olay F1={row['event_f1_mean']:.3f}±{row['event_f1_std']:.3f}"
            for _, row in parts.iterrows()
        )
        rows.append({
            "exp_id": method, "date": date.today().isoformat(), "model": "Test sonrası deneme (final değil)",
            "seed": config["seed"], "val_mae": "", "val_rmse": "", "test_mae": "", "test_rmse": "",
            "note": f"Test sonrası geliştirme, test kullanılmadı; {notes}",
        })
    path = resolve_path(PROJECT_ROOT, config["paths"]["experiments_file"])
    columns = pd.read_csv(path, nrows=0).columns
    pd.DataFrame(rows).reindex(columns=columns).to_csv(path, mode="a", header=False, index=False)


def main():
    """select veya hold aşamasını çalıştırır ve tabloyu basar."""
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    args = parser.parse_args()
    config = load_config()
    set_seed(config["seed"])
    tables_dir = resolve_path(PROJECT_ROOT, config["paths"]["tables_dir"])
    if args.stage == "select":
        settings, per_seed = run_select(config, tables_dir)
        print({name: settings[name] for name in (DUAL, DUAL_FLOOR)}, "taban (kW):", round(settings["floor_kw"], 4))
    else:
        per_seed = run_hold(config, tables_dir)
    frames = [per_seed] if args.stage == "select" else [pd.read_csv(tables_dir / "a6_val_dev_per_seed.csv"), per_seed]
    summary = summarize_methods(pd.concat(frames, ignore_index=True))
    summary.to_csv(tables_dir / "a6_val_summary.csv", index=False)
    if args.stage == "hold":
        write_experiment_rows(config, summary)     # ölçüm bir kez yapıldığı için günlüğe de bir kez yazılır
    pd.set_option("display.width", 250)
    columns = ["split", "method", "event_f1_mean", "event_f1_std", "event_recall_mean", "event_precision_mean",
               *(f"event_recall_{k}_mean" for k in ANOMALY_TYPES), "false_alarms_mean",
               "false_alarm_increase_vs_A3_mean"]
    print(summary[columns].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
