"""Kanal başına ayrı yumuşatma: (A3-s2s-hmean, w=1, ≥ T1) VEYA (C1-L2, w=6, ≥ T2). KEŞİFSEL.

Protokol: docs/anomaly_or_channels.md. Birleşimden sonra yumuşatma yok; eşdeğer skor
r = max(a/T1, c/T2), alarm r ≥ 1. Kümeler kış-dev + yaz-dev, orta senaryo; val_hold ve test
KULLANILMAZ. Tahmin modeli eğitilmez. Eşikler aynı geliştirme kopyalarında seçildi (seçim yanlılığı).
Komut: python -m src.anomaly_or_channels
"""

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from src.anomaly_candidates import S0_SCALED_THRESHOLD, decide, evaluation_arrays, prepare, summarize
from src.anomaly_dev import SET_KEYS
from src.anomaly_e1_sustained import MANIFEST_ITEMS, Log, copy_metrics, paired
from src.anomaly_eval_audit import sha256
from src.data.inject_anomalies import ANOMALY_TYPES
from src.data.load import PROJECT_ROOT, load_config
from src.utils.anomaly_metrics import event_metrics, runs
from src.utils.console import setup_console
from src.utils.results_io import resolve_path, upsert_csv
from src.utils.seed import set_seed

T1 = S0_SCALED_THRESHOLD                 # S0 (Aşama 3), w = 1
T2 = 7.052471410803624                   # C1-L2 tek başına seçilen eşik, w = 6
C1_WINDOW = 6
STAGE_B_MULTIPLIERS = (1.25, 1.5, 2.0)   # yalnız Aşama A kabul edilmezse
DEADLINE = datetime(2026, 11, 9, 1, 15)  # bundan sonra yeni aşamaya geçilmez


def channel_arrays(item, scales):
    """Bir kopya (veya temiz küme) için (a, c, geçersiz_maske) — kopya ekseninde."""
    a, _ = evaluation_arrays(item, scales, "S0-yerel", 1)
    c, _ = evaluation_arrays(item, scales, "C1-L2", C1_WINDOW)
    imputed = item["frame"]["is_imputed"].to_numpy() == 1
    return a, c, imputed


def or_score(a, c, imputed, t1, t2):
    """r = max(a/T1, c/T2) (NaN kanal diğerini engellemez); geçerli = sonlu ve doldurulmamış."""
    with np.errstate(invalid="ignore"):
        score = np.fmax(a / t1, c / t2)
    valid = np.isfinite(score) & ~imputed
    return score, valid


def single(item, scales, method):
    """Tek kanal yöntemleri (kayıtlı ayarla): (skor, geçerli, eşik)."""
    a, c, imputed = channel_arrays(item, scales)
    score, threshold = (a, T1) if method == "S0-yerel" else (c, T2)
    return score, np.isfinite(score) & ~imputed, threshold


def evaluate(data, method, t2=None):
    """Kopya başına metrikler; OR için kanal katkısı ve S0'a eklenen yanlış alarmlar da."""
    rows, events, added = [], [], []
    for set_name, item in data.items():
        scales = item["scales"]
        if method.startswith("OR"):
            a, c, imputed = channel_arrays(item["background"], scales)
            clean_score, clean_valid = or_score(a, c, imputed, T1, t2)
            background = (clean_score >= 1.0) & clean_valid
        else:
            clean_score, clean_valid, threshold = single(item["background"], scales, method)
            background = (clean_score >= threshold) & clean_valid
        for copy in item["copies"]:
            labels = copy["labels"]
            types = labels["anomaly_type"].to_numpy()
            if method.startswith("OR"):
                a, c, imputed = channel_arrays(copy, scales)
                score, valid = or_score(a, c, imputed, T1, t2)
                threshold = 1.0
                events += attribution(set_name, copy["seed"], types, (a >= T1) & valid, (c >= t2) & valid)
                added.append(added_false_alarms(set_name, copy["seed"], types, (score >= 1.0) & valid,
                                                (a >= T1) & valid, background))
            else:
                score, valid, threshold = single(copy, scales, method)
            rows.append({"set": set_name, "method": method, "seed": copy["seed"], "threshold": threshold,
                         **copy_metrics(types, labels["is_anomaly"].to_numpy(), score, valid, threshold,
                                        background)})
    return rows, events, added


def attribution(set_name, seed, types, flag_a, flag_c):
    """Her olay: yalnız A3 / yalnız C1 / ikisi / hiçbiri (tip yalnız raporlama için, seçimde yok)."""
    rows = []
    for kind in ANOMALY_TYPES:
        for start, end in runs(types == kind):
            hit_a, hit_c = flag_a[start:end].any(), flag_c[start:end].any()
            who = "ikisi" if hit_a and hit_c else "yalnız A3" if hit_a else "yalnız C1" if hit_c else "hiçbiri"
            rows.append({"set": set_name, "seed": seed, "type": kind, "caught_by": who})
    return rows


def added_false_alarms(set_name, seed, types, flag_or, flag_s0, background):
    """OR'un S0'a eklediği yanlış bloklar: anomaliye ve S0 alarmına dokunmayan OR blokları."""
    anomalous = types != "none"
    false_blocks = [(s, e) for s, e in runs(flag_or) if not anomalous[s:e].any()]
    new = [(s, e) for s, e in false_blocks if not flag_s0[s:e].any()]
    return {"set": set_name, "seed": seed, "or_false_blocks": len(false_blocks), "added_vs_s0": len(new),
            "added_in_background": sum(background[s:e].any() for s, e in new)}


def run_stage(data, name, t2, log):
    rows, events, added = evaluate(data, name, t2)
    log(f"değerlendirildi: {name} (T2 = {t2:.3f})")
    return rows, events, added


def main():
    setup_console()
    argparse.ArgumentParser(description=__doc__).parse_args()
    config = load_config()
    set_seed(config["seed"])
    tables_dir = resolve_path(PROJECT_ROOT, config["paths"]["tables_dir"])
    log = Log(resolve_path(PROJECT_ROOT, "results/logs/anomaly_or_channels.log"))

    manifest = pd.read_csv(tables_dir / "repro_manifest.csv").set_index("item")
    for item in MANIFEST_ITEMS:
        if sha256(PROJECT_ROOT / manifest.loc[item, "path"]) != manifest.loc[item, "sha256"]:
            raise RuntimeError(f"{item} hash'i manifestoyla uyuşmuyor.")
    log("[1/5] manifest denetimi tamam")

    df = pd.read_parquet(resolve_path(PROJECT_ROOT, config["paths"]["processed_file"]))
    data = {}
    for set_name in SET_KEYS:
        data[set_name] = prepare(set_name, df, config)
        log(f"[2/5] hazırlandı: {set_name}")

    rows, events, added = [], [], []
    for method in ("S0-yerel", "C1-L2"):
        part, _, _ = evaluate(data, method)
        rows += part
    part, ev, ad = run_stage(data, "OR-A", T2, log)
    rows, events, added = rows + part, events + [dict(r, method="OR-A") for r in ev], added + [dict(r, method="OR-A") for r in ad]
    per_seed = pd.DataFrame(rows)
    summary = summarize(per_seed)
    decisions = [extended_decision(per_seed, summary, "OR-A")]
    log(f"[3/5] Aşama A kabul: {decisions[-1]['accepted']}")

    stage_b = []
    if not decisions[-1]["accepted"] and datetime.now() < DEADLINE:
        # Aşama B: T2 ∈ {1,25; 1,5; 2,0} × T2, 20 kopyada birleşik block_f1 ile (eşitlikte yüksek T2).
        for multiplier in STAGE_B_MULTIPLIERS:
            t2 = multiplier * T2
            detected = n_events = correct = n_blocks = 0
            for item in data.values():
                for copy in item["copies"]:
                    a, c, imputed = channel_arrays(copy, item["scales"])
                    score, valid = or_score(a, c, imputed, T1, t2)
                    m = event_metrics(copy["labels"]["anomaly_type"].to_numpy(), (score >= 1.0) & valid)
                    n_events += m["n_events"]
                    detected += m["event_recall"] * m["n_events"] if m["n_events"] else 0
                    n_blocks += m["n_pred_blocks"]
                    correct += m["event_precision"] * m["n_pred_blocks"]
            recall, precision = detected / n_events, (correct / n_blocks if n_blocks else 0.0)
            pooled = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            stage_b.append({"multiplier": multiplier, "t2": t2, "pooled_block_f1": pooled})
            log(f"[4/5] Aşama B aday ×{multiplier}: birleşik block_f1 {pooled:.4f}")
        best = sorted(stage_b, key=lambda r: (r["pooled_block_f1"], r["t2"]), reverse=True)[0]
        part, ev, ad = run_stage(data, "OR-B", best["t2"], log)
        rows += part
        events += [dict(r, method="OR-B") for r in ev]
        added += [dict(r, method="OR-B") for r in ad]
        per_seed = pd.DataFrame(rows)
        summary = summarize(per_seed)
        decisions.append(extended_decision(per_seed, summary, "OR-B") | {"t2": best["t2"],
                                                                          "selection_bias": True})
        log(f"[4/5] Aşama B (T2 = {best['t2']:.3f}) kabul: {decisions[-1]['accepted']}")
    elif not decisions[-1]["accepted"]:
        log("SÜRE: Aşama B'ye geçilmedi")

    paired_rows = [{"method": m, **r} for m in per_seed["method"].unique() if m != "S0-yerel"
                   for metric in ("block_f1", "count_f1", "block_precision", "count_precision", "event_recall",
                                  "false_alarms_per_day", "detection_delay_mean",
                                  *[f"event_recall_{k}" for k in ANOMALY_TYPES])
                   for r in paired(per_seed, m, metric)]
    events = pd.DataFrame(events)
    attribution_table = (events.groupby(["method", "set", "type", "caught_by"]).size()
                         .unstack("caught_by", fill_value=0).reset_index())
    added_table = pd.DataFrame(added).groupby(["method", "set"])[
        ["or_false_blocks", "added_vs_s0", "added_in_background"]].mean().reset_index()

    prefix = tables_dir / "anomaly_or_channels"
    per_seed.to_csv(f"{prefix}_per_seed.csv", index=False)
    summary.to_csv(f"{prefix}_summary.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(f"{prefix}_paired.csv", index=False)
    pd.DataFrame(decisions).to_csv(f"{prefix}_decision.csv", index=False)
    attribution_table.to_csv(f"{prefix}_attribution.csv", index=False)
    added_table.to_csv(f"{prefix}_added_false_alarms.csv", index=False)
    if stage_b:
        pd.DataFrame(stage_b).to_csv(f"{prefix}_stage_b.csv", index=False)
    upsert_csv(resolve_path(PROJECT_ROOT, config["paths"]["experiments_file"]), [
        {"exp_id": f"ORCH-{m}", "date": datetime.now().date().isoformat(),
         "model": f"anomali {m} (KEŞİFSEL; docs/anomaly_or_channels.md)", "seed": config["seed"],
         "note": "orta senaryo block_f1 kış {:.3f} / yaz {:.3f}; eşikler aynı dev kopyalarında seçildi "
                 "(seçim yanlılığı); val_hold/test yok".format(
                     *summary.set_index(["method", "set"]).loc[[(m, "winter_dev"), (m, "summer_dev")], "block_f1"])}
        for m in per_seed["method"].unique() if m.startswith("OR")], ["exp_id"])
    log(f"[5/5] tablolar yazıldı; kabul: {[(d['method'], d['accepted']) for d in decisions]}")

    columns = ["method", "set", "block_f1", "count_f1", "block_precision", "count_precision", "event_recall",
               "event_recall_spike", "event_recall_drop", "event_recall_level_shift", "event_recall_pattern",
               "false_alarms_per_day", "detection_delay_mean", "pr_auc"]
    print(summary[columns].round(3).to_string(index=False))
    print(pd.DataFrame(decisions).round(4).to_string(index=False))
    print(attribution_table.to_string(index=False))
    print(added_table.round(2).to_string(index=False))


def extended_decision(per_seed, summary, method):
    """c1–c5 (anomaly_candidates.decide) + c6 yaz desen recall'ü artışı + c7 spike kaybı ≤ 0,05."""
    result = decide(per_seed, summary, method)
    s = summary.set_index(["method", "set"])
    result["c6_summer_pattern_recall_up"] = (s.loc[(method, "summer_dev"), "event_recall_pattern"]
                                             > s.loc[("S0-yerel", "summer_dev"), "event_recall_pattern"])
    result["c7_spike_loss_le_0.05"] = all(
        s.loc[("S0-yerel", k), "event_recall_spike"] - s.loc[(method, k), "event_recall_spike"] <= 0.05
        for k in SET_KEYS)
    result["accepted"] = bool(result["accepted"] and result["c6_summer_pattern_recall_up"]
                              and result["c7_spike_loss_le_0.05"])
    return result


if __name__ == "__main__":
    main()
