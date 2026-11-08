"""Anomali aday yöntemleri C1 (sağlam profil sapması) ve C2 (C1 + A3) – docs/anomaly_candidates.md.

C1: s_t = |y_t − medyan(R_t)| / max(1,4826·MAD(R_t), taban(saat, gün tipi)); R_t = son L haftanın
aynı saat ve aynı gün tipindeki (hafta içi / hafta sonu-tatil) gözlemleri, yalnız geçmiş.
C2: max(A3-s2s-hmean / q_A3, C1 / q_C1).
Referanslar S0-yerel, A0, A3-s2s-norm aynı kopyalarda, kayıtlı ayarlarıyla. Kümeler kış-dev + yaz-dev,
orta senaryo; val_hold ve test KULLANILMAZ. Tahmin modeli eğitilmez.
Komut: python -m src.anomaly_candidates
"""

import argparse
import time
import warnings

import numpy as np
import pandas as pd

from src.anomaly_dev import SET_KEYS, prepare_set, score_set
from src.anomaly_e1_sustained import MANIFEST_ITEMS, Log, copy_metrics, paired
from src.anomaly_eval_audit import sha256
from src.data.inject_anomalies import ANOMALY_TYPES
from src.data.load import PROJECT_ROOT, load_config
from src.evaluate_anomaly import score_frame, smooth_scores, with_context
from src.utils.anomaly_metrics import event_f1_max_threshold, event_metrics
from src.utils.console import setup_console
from src.utils.results_io import resolve_path, upsert_csv
from src.utils.seed import set_seed

WEEKS = (2, 4)                    # C1 aday bütçesi
WINDOWS = (1, 3, 6)               # mevcut yumuşatma seçimi
MIN_REFERENCES = 4
FLOOR_QUANTILE = 0.25
REFERENCE_QUANTILE = 0.99
PILOT_SEEDS = 3
PILOT_MARGIN = 0.03
BUDGET_S = 75 * 60
# Referansların kayıtlı ayarları (yeniden seçilmez): yöntem → (ham skor sütunu, pencere, eşik).
REFERENCES = {"S0-yerel": ("A3-s2s-hmean", 1, None),      # eşik = 1,1636 · q_A3 (Aşama 3)
              "A0": ("A0", 6, 3.9893783062624975),
              "A3-s2s-norm": ("A3-s2s-norm", 1, 4.240649915881419)}
S0_SCALED_THRESHOLD = 1.1636431513354957


def day_type(frame):
    """0 = hafta içi, 1 = hafta sonu veya resmî tatil (takvimden; önceden bilinir)."""
    return ((frame["is_weekend"] == 1) | (frame["is_holiday"] == 1)).astype(int)


def robust_profile(frame, target, weeks):
    """Her saat için geçmiş referansın medyanı, 1,4826·MAD'ı ve referans sayısı.

    Girdi: saatlik tablo (geçmiş bağlam dahil). Referans: t − 24k (k = 1…7·weeks) saatlerinden
    gün tipi t ile aynı olanlar. Saatin kendisi ve gelecek referansa girmez.
    Çıktı: (medyan [T], ölçek [T], sayı [T]) Series'leri.
    """
    values, kind = frame[target], day_type(frame)
    references = np.column_stack([
        values.shift(24 * k).where(kind.shift(24 * k) == kind).to_numpy(dtype=float)
        for k in range(1, 7 * weeks + 1)
    ])                                                             # [T, 7·weeks]
    count = np.isfinite(references).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # referanssız satır: tümü NaN
        median = np.nanmedian(references, axis=1)
        mad = np.nanmedian(np.abs(references - median[:, None]), axis=1)
    index = frame.index
    return pd.Series(median, index), pd.Series(1.4826 * mad, index), pd.Series(count, index)


def fit_floor(train, target, weeks):
    """Taban(saat, gün tipi) = train'deki ölçeğin 25. yüzdeliği (yalnız train, doldurulmamış saat)."""
    _, scale, count = robust_profile(train, target, weeks)
    ok = (count >= MIN_REFERENCES) & (train["is_imputed"] == 0)
    keys = [train.index.hour[ok], day_type(train)[ok]]
    return scale[ok].groupby(keys).quantile(FLOOR_QUANTILE)


def c1_score(frame, target, weeks, floor):
    """C1 ham skoru [T]: |y − medyan| / max(MAD ölçeği, taban); referans < 4 ise NaN."""
    median, scale, count = robust_profile(frame, target, weeks)
    keys = pd.MultiIndex.from_arrays([frame.index.hour, day_type(frame)])
    scale = np.maximum(scale.to_numpy(), floor.reindex(keys).to_numpy())
    score = (frame[target] - median).abs() / scale
    return score.where(count >= MIN_REFERENCES)


def history_frame(train, frame, weeks):
    """Değerlendirilen kopyanın önüne train'in son 7·weeks gün + 1 gününü (ENJEKSİYONSUZ) ekler."""
    return pd.concat((train.tail(24 * (7 * weeks + 1)), frame))


def prepare(set_name, df, config):
    """Bir küme: kopyalar, referans ham skorları, C1 ham skorları (iki L), temiz skorlar, ölçekler."""
    prepared = prepare_set(df, config, set_name, "cpu")
    cfg, splits, key = prepared["config"], prepared["splits"], prepared["key"]
    target = cfg["preprocess"]["target"]
    train, clean = splits["train"], splits[key]
    floors = {w: fit_floor(train, target, w) for w in WEEKS}
    scales = {f"C1-L{w}": float(c1_score(train, target, w, floors[w])[train["is_imputed"] == 0]
                                .dropna().quantile(REFERENCE_QUANTILE)) for w in WEEKS}
    scales["A3"] = float(pd.read_csv(resolve_path(PROJECT_ROOT, "results/tables/E1_sustained_scales.csv"))
                         .set_index("set").loc[set_name, "A3"])

    def raw_scores(frame, reference_scores):
        """[T] ham skorlar: referans sütunları (bağlamlı) + C1 (kendi geçmiş bağlamıyla), kopya ekseninde."""
        result = {name: reference_scores[name] for name in ("A0", "A3-s2s-norm", "A3-s2s-hmean")}
        for w in WEEKS:
            result[f"C1-L{w}"] = c1_score(history_frame(train, frame, w), target, w, floors[w])
        return result

    copies = []
    for seed, scores, labels, injected in score_set(prepared, set_name, "medium", "cpu"):
        copies.append({"seed": seed, "labels": labels, "frame": injected, "raw": raw_scores(injected, scores)})
    clean_scores = score_frame(with_context(splits, key, clean, cfg), cfg, prepared["fitted"], "cpu")
    background = {"frame": clean, "raw": raw_scores(clean, clean_scores)}
    return {"copies": copies, "background": background, "scales": scales}


def method_score(raw, scales, method, window):
    """Yöntem + pencere → yumuşatılmış [T] skor (bağlamlı; sonra kopya eksenine kesilir)."""
    if method in REFERENCES:
        series = raw[REFERENCES[method][0]]
        if method == "S0-yerel":
            series = series / scales["A3"]
    elif method.startswith("C2"):
        weeks = method.split("-L")[1]
        series = pd.concat([raw["A3-s2s-hmean"] / scales["A3"],
                            raw[f"C1-L{weeks}"] / scales[f"C1-L{weeks}"]], axis=1).max(axis=1, skipna=False)
    else:
        series = raw[method]
    return smooth_scores(series.to_frame("s"), window)["s"]


def evaluation_arrays(item, scales, method, window):
    """Bir kopya (veya temiz küme) için kopya ekseninde (skor, geçerli) dizileri."""
    frame = item["frame"]
    score = method_score(item["raw"], scales, method, window).reindex(frame.index).to_numpy(dtype=float)
    valid = ~np.isnan(score) & (frame["is_imputed"].to_numpy() == 0)
    return score, valid


def select(data, method, seeds_per_set, quantiles):
    """Pencere × eşik seçimi (mevcut kural): seçilen kopyalarda birleşik block_f1 eşiği, ort. block_f1 ile pencere."""
    rows = []
    for window in WINDOWS:
        copies = []
        for set_name, item in data.items():
            for copy in item["copies"][:seeds_per_set]:
                score, valid = evaluation_arrays(copy, item["scales"], method, window)
                copies.append((copy["labels"]["anomaly_type"].to_numpy(), score, valid))
        threshold, _ = event_f1_max_threshold(copies, quantiles)
        f1 = float(np.mean([event_metrics(t, (s >= threshold) & v)["event_f1"] for t, s, v in copies]))
        rows.append({"method": method, "window": window, "threshold": threshold, "selection_block_f1": f1})
    table = pd.DataFrame(rows).sort_values(["selection_block_f1", "window"], ascending=[False, True])
    return table.iloc[0].to_dict(), table


def evaluate(data, method, window, threshold, seeds_per_set):
    """Kopya başına metrikler (+ arka plan alarmları: aynı ayarla enjeksiyonsuz kümede)."""
    rows = []
    for set_name, item in data.items():
        clean_score, clean_valid = evaluation_arrays(item["background"], item["scales"], method, window)
        background = (clean_score >= threshold) & clean_valid
        for copy in item["copies"][:seeds_per_set]:
            score, valid = evaluation_arrays(copy, item["scales"], method, window)
            labels = copy["labels"]
            rows.append({"set": set_name, "method": method, "seed": copy["seed"], "window": window,
                         "threshold": threshold,
                         **copy_metrics(labels["anomaly_type"].to_numpy(), labels["is_anomaly"].to_numpy(),
                                        score, valid, threshold, background)})
    return rows


def reference_threshold(method):
    return S0_SCALED_THRESHOLD if method == "S0-yerel" else REFERENCES[method][2]


def both(per_seed, method, metric):
    """İki mevsim ortalaması ve eşleştirilmiş fark (S0-yerel'e karşı): ort, se."""
    rows = {r["set"]: r for r in paired(per_seed, method, metric)}
    mean = (rows["winter_dev"]["diff_mean"] + rows["summer_dev"]["diff_mean"]) / 2
    se = np.sqrt(rows["winter_dev"]["diff_se"] ** 2 + rows["summer_dev"]["diff_se"] ** 2) / 2
    return mean, se, rows


def decide(per_seed, summary, method):
    """Protokol kabul ölçütü (5 koşul, S0-yerel'e karşı)."""
    s = summary.set_index(["method", "set"])
    f1_mean, f1_se, f1_rows = both(per_seed, method, "block_f1")
    count_mean, _, _ = both(per_seed, method, "count_f1")
    _, _, fa_rows = both(per_seed, method, "false_alarms_per_day")
    checks = {
        "c1_block_f1_gain_beyond_2se": f1_mean > 2 * f1_se,
        "c2_block_f1_gain_both_seasons": all(f1_rows[k]["diff_mean"] > 0 for k in SET_KEYS),
        "c3_false_alarm_increase_le_0.05": all(fa_rows[k]["diff_mean"] <= 0.05 for k in SET_KEYS),
        "c4_level_pattern_loss_le_0.05": all(
            s.loc[("S0-yerel", k), f"event_recall_{t}"] - s.loc[(method, k), f"event_recall_{t}"] <= 0.05
            for k in SET_KEYS for t in ("level_shift", "pattern")),
        "c5_count_f1_gain": count_mean > 0,
    }
    return {"method": method, "both_block_f1_diff": f1_mean, "both_block_f1_diff_se": f1_se,
            "both_count_f1_diff": count_mean, **checks, "accepted": all(checks.values())}


def summarize(per_seed):
    metrics = [c for c in per_seed.columns if c not in ("set", "method", "seed", "window", "threshold")]
    summary = per_seed.groupby(["method", "set"])[metrics].mean()
    stds = per_seed.groupby(["method", "set"])[["block_f1", "count_f1"]].std().add_suffix("_std")
    return summary.join(stds).reset_index()


def main():
    setup_console()
    argparse.ArgumentParser(description=__doc__).parse_args()
    config = load_config()
    set_seed(config["seed"])
    tables_dir = resolve_path(PROJECT_ROOT, config["paths"]["tables_dir"])
    log = Log(resolve_path(PROJECT_ROOT, "results/logs/anomaly_candidates.log"))

    manifest = pd.read_csv(tables_dir / "repro_manifest.csv").set_index("item")
    for item in MANIFEST_ITEMS:
        if sha256(PROJECT_ROOT / manifest.loc[item, "path"]) != manifest.loc[item, "sha256"]:
            raise RuntimeError(f"{item} hash'i manifestoyla uyuşmuyor.")
    log("[1/6] manifest denetimi tamam")

    df = pd.read_parquet(resolve_path(PROJECT_ROOT, config["paths"]["processed_file"]))
    data, runtime = {}, {}
    for i, set_name in enumerate(SET_KEYS, start=1):
        start = time.perf_counter()
        data[set_name] = prepare(set_name, df, config)
        runtime[set_name] = time.perf_counter() - start
        log(f"[2/6] hazırlandı {i}/2 {set_name}: {len(data[set_name]['copies'])} kopya, ölçekler "
            f"{ {k: round(v, 3) for k, v in data[set_name]['scales'].items()} }")

    settings = config["protocol"]["threshold_quantiles"]
    quantiles = np.linspace(settings["low"], settings["high"], settings["n"])
    n_full = len(data["winter_dev"]["copies"])

    # Pilot: ilk 3 seed / küme; referans S0-yerel kayıtlı ayarla, adaylar pilot kopyalarda seçilir.
    pilot_rows, pilot_selection, timing = [], [], {}
    pilot_rows += evaluate(data, "S0-yerel", 1, S0_SCALED_THRESHOLD, PILOT_SEEDS)
    candidates = [f"C1-L{w}" for w in WEEKS]
    for method in candidates:
        start = time.perf_counter()
        chosen, _ = select(data, method, PILOT_SEEDS, quantiles)
        pilot_selection.append(chosen)
        pilot_rows += evaluate(data, method, int(chosen["window"]), chosen["threshold"], PILOT_SEEDS)
        timing[method] = time.perf_counter() - start
        log(f"[3/6] pilot {method}: w={int(chosen['window'])}, eşik {chosen['threshold']:.3f}")
    pilot = pd.DataFrame(pilot_rows)
    pilot_means = pilot.groupby(["method", "set"])[["block_f1", "count_f1"] +
                                                   [f"event_recall_{k}" for k in ANOMALY_TYPES]].mean()
    s0_pilot = pilot_means.loc["S0-yerel", "block_f1"].mean()
    survivors = []
    for method in candidates:
        value = pilot_means.loc[method, "block_f1"].mean()
        keep = value >= s0_pilot - PILOT_MARGIN
        log(f"      pilot {method}: block_f1 ort {value:.3f} (S0 {s0_pilot:.3f}) → {'devam' if keep else 'ELENDİ'}")
        if keep:
            survivors.append(method)
    # En iyi C1 (pilot block_f1); C2 koşulu: C1 elenmediyse veya S0'ın kaçırdığı tipte C1 recall ≥ 0,3.
    best_c1 = max(candidates, key=lambda m: pilot_means.loc[m, "block_f1"].mean())
    s0_types = pilot_means.loc["S0-yerel"]
    c1_types = pilot_means.loc[best_c1]
    # Herhangi bir mevsimde S0'ın kaçırdığı (< 0,3) bir tipte C1 recall'ü ≥ 0,3 mü?
    complements = any(((c1_types[f"event_recall_{k}"] >= 0.3) & (s0_types[f"event_recall_{k}"] < 0.3)).any()
                      for k in ANOMALY_TYPES)
    c2 = f"C2-L{best_c1.split('-L')[1]}"
    run_c2 = best_c1 in survivors or bool(complements)
    log(f"      C2 ({c2}) {'çalıştırılacak' if run_c2 else 'çalıştırılmayacak'} (tamamlayıcı tip: {bool(complements)})")
    if run_c2:
        start = time.perf_counter()
        chosen, _ = select(data, c2, PILOT_SEEDS, quantiles)
        pilot_rows_c2 = evaluate(data, c2, int(chosen["window"]), chosen["threshold"], PILOT_SEEDS)
        timing[c2] = time.perf_counter() - start
        value = pd.DataFrame(pilot_rows_c2)["block_f1"].mean()
        pilot_rows += pilot_rows_c2
        pilot_selection.append(chosen)
        keep = value >= s0_pilot - PILOT_MARGIN
        log(f"[3/6] pilot {c2}: w={int(chosen['window'])}, block_f1 ort {value:.3f} → {'devam' if keep else 'ELENDİ'}")
        if keep:
            survivors.append(c2)
    pd.DataFrame(pilot_rows).to_csv(tables_dir / "anomaly_candidates_pilot_per_seed.csv", index=False)
    pd.DataFrame(pilot_selection).to_csv(tables_dir / "anomaly_candidates_pilot_selection.csv", index=False)
    if log.over_budget():
        log("SÜRE DOLDU: pilot sonrası durduruldu")
        return

    # Tam değerlendirme: 10 seed / küme; adaylar 20 kopyada yeniden seçilir, referanslar kayıtlı ayarla.
    rows, selections = [], []
    for method in REFERENCES:
        rows += evaluate(data, method, REFERENCES[method][1], reference_threshold(method), n_full)
    for method in survivors:
        start = time.perf_counter()
        chosen, table = select(data, method, n_full, quantiles)
        selections.append(table)
        rows += evaluate(data, method, int(chosen["window"]), chosen["threshold"], n_full)
        timing[method] = timing.get(method, 0) + time.perf_counter() - start
        log(f"[4/6] tam {method}: w={int(chosen['window'])}, eşik {chosen['threshold']:.3f}")
        if log.over_budget():
            log("SÜRE DOLDU: tam değerlendirme yarıda bırakıldı")
            break
    per_seed = pd.DataFrame(rows)
    summary = summarize(per_seed)
    summary["runtime_s"] = summary["method"].map(timing)
    decisions = pd.DataFrame([decide(per_seed, summary, m) for m in survivors]) if survivors else pd.DataFrame()
    paired_rows = [{"method": m, **r} for m in per_seed["method"].unique() if m != "S0-yerel"
                   for metric in ("block_f1", "count_f1", "block_precision", "count_precision", "event_recall",
                                  "false_alarms_per_day", "detection_delay_mean",
                                  *[f"event_recall_{k}" for k in ANOMALY_TYPES])
                   for r in paired(per_seed, m, metric)]
    log(f"[5/6] değerlendirildi: {per_seed['method'].nunique()} yöntem × {len(per_seed) // per_seed['method'].nunique()} kopya")

    per_seed.to_csv(tables_dir / "anomaly_candidates_per_seed.csv", index=False)
    summary.to_csv(tables_dir / "anomaly_candidates_summary.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(tables_dir / "anomaly_candidates_paired.csv", index=False)
    decisions.to_csv(tables_dir / "anomaly_candidates_decision.csv", index=False)
    if selections:
        pd.concat(selections).to_csv(tables_dir / "anomaly_candidates_selection.csv", index=False)
    upsert_csv(resolve_path(PROJECT_ROOT, config["paths"]["experiments_file"]), [
        {"exp_id": f"CAND-{m}", "date": pd.Timestamp.today().date().isoformat(),
         "model": f"anomali {m} (geliştirme; docs/anomaly_candidates.md)", "seed": config["seed"],
         "note": "orta senaryo block_f1 kış {:.3f} / yaz {:.3f}; yerel ağırlıklar; val_hold/test yok".format(
             *summary.set_index(["method", "set"]).loc[[(m, "winter_dev"), (m, "summer_dev")], "block_f1"])}
        for m in per_seed["method"].unique()], ["exp_id"])
    accepted = decisions.loc[decisions["accepted"], "method"].tolist() if len(decisions) else []
    log(f"[6/6] kabul edilen: {accepted or 'yok'}")

    columns = ["method", "set", "block_f1", "count_f1", "block_precision", "count_precision", "event_recall",
               "event_recall_spike", "event_recall_drop", "event_recall_level_shift", "event_recall_pattern",
               "false_alarms_per_day", "detection_delay_mean", "pr_auc"]
    print(summary[columns].round(3).to_string(index=False))
    if len(decisions):
        print(decisions.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
