"""E1: A3-s2s-hmean + sürekli işaretli artık (docs/anomaly_e1_sustained.md; geliştirme, final değil).

Skor: E1-w = max(A3 / q_A3, u^(w) / q_u), u^(w)_t = |ort(z̄_{t−w+1}, …, z̄_t)|, z̄ = işaretli
ortalama normalize artık (= spike − drop). q'lar train OOF saatlerinin 99. yüzdeliği.
Adaylar w ∈ {3, 6}; referans S0-yerel (A3-s2s-hmean, Aşama 3 eşiği, bu makinedeki ağırlıklar).
Kümeler kış-dev + yaz-dev, orta senaryo; val_hold ve test KULLANILMAZ. Tahmin modeli eğitilmez.
Komut: python -m src.anomaly_e1_sustained
"""

import argparse
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score

from src.anomaly_dev import SET_KEYS, prepare_set
from src.anomaly_eval_audit import sha256
from src.anomaly_short_event import change_table, score_copies, signal_frame
from src.data.inject_anomalies import ANOMALY_TYPES
from src.data.load import PROJECT_ROOT, load_config
from src.evaluate_anomaly import with_context
from src.models.residual_detector import multi_step_predictions
from src.utils.anomaly_metrics import (
    alarm_diagnostics,
    event_count_metrics,
    event_f1_max_threshold,
    event_metrics,
)
from src.utils.console import setup_console
from src.utils.results_io import resolve_path, upsert_csv
from src.utils.seed import set_seed

WINDOWS = (3, 6)                       # en fazla iki aday (protokol)
REFERENCE_QUANTILE = 0.99
S0_THRESHOLD = 1.1636431513354957      # Aşama 3'te seçilen S0 eşiği (ölçekli; anomaly_short_event_summary.csv)
BUDGET_S = 50 * 60                     # 50 dk sonra yeni adıma geçilmez (toplam bütçe 60 dk)
MANIFEST_ITEMS = ("data", "config", "F10-opt", "F10-opt-oof1", "F10-opt-oof2", "F10-opt-oof3",
                  "S-F10-L48", "S-F10-L48-oof1", "S-F10-L48-oof2")


class Log:
    """Zaman damgalı ilerleme kaydı (ekran + results/logs/E1_sustained.log)."""

    def __init__(self, path):
        self.path, self.start = path, time.perf_counter()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def elapsed(self):
        return time.perf_counter() - self.start

    def __call__(self, message):
        line = f"{datetime.now():%H:%M:%S} [+{self.elapsed() / 60:4.1f} dk] {message}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as file:
            file.write(line + "\n")

    def over_budget(self):
        return self.elapsed() > BUDGET_S


def sustained(signals, window):
    """u^(w)_t = |ort(z̄_{t−w+1..t})|; z̄ = spike − drop. Yalnız geçmişe bakar, eksikte NaN."""
    mean_signed = signals["spike"] - signals["drop"]                  # [T] işaretli z̄
    return mean_signed.rolling(window, min_periods=window).mean().abs()


def scores_for(signals, scales, window):
    """Yöntem → [T] skor: S0-yerel ve E1-w (ölçekli; birleşik = max, NaN korunur)."""
    a3 = signals["A3"] / scales["A3"]
    result = {"S0-yerel": a3}
    if window is not None:
        u = sustained(signals, window) / scales[f"u{window}"]
        result[f"E1-w{window}"] = pd.concat([a3, u], axis=1).max(axis=1, skipna=False)
    return result


def train_scales(prepared, change_tab):
    """q99'lar yalnız train OOF saatlerinden: A3 ve her w için u^(w)."""
    train = prepared["splits"]["train"]
    target = prepared["config"]["preprocess"]["target"]
    fitted = prepared["fitted"]
    signals = signal_frame(train, fitted["train_multi_predictions"], fitted["multi_std"], change_tab, target)
    clean = (train["is_imputed"] == 0).to_numpy()
    scales = {"A3": float(signals["A3"][clean].dropna().quantile(REFERENCE_QUANTILE))}
    for window in WINDOWS:
        u = sustained(signals, window)
        scales[f"u{window}"] = float(u[clean].dropna().quantile(REFERENCE_QUANTILE))
    return scales


def copy_metrics(types, labels, score, valid, threshold, background):
    """Bir kopya: blok ve olay sayımlı metrikler, tip recall'ü, yanlış alarm, gecikme, PR-AUC."""
    with np.errstate(invalid="ignore"):
        flagged = (score >= threshold) & valid
    block = event_metrics(types, flagged)
    counted = event_count_metrics(types, flagged)
    alarms = alarm_diagnostics(types, flagged, int(valid.sum()))
    anomalous = types != "none"
    false_blocks = [(a, b) for a, b in _runs(flagged) if not anomalous[a:b].any()]
    row = {"block_f1": block["event_f1"], "block_precision": block["event_precision"],
           "event_recall": block["event_recall"], **counted,
           "false_alarms_per_day": alarms["false_alarms_per_day"],
           "false_blocks_in_background": sum(background[a:b].any() for a, b in false_blocks),
           "detection_delay_mean": alarms["detection_delay_mean"],
           "pr_auc": average_precision_score(labels[valid], score[valid])}
    row.update({f"event_recall_{k}": block[f"event_recall_{k}"] for k in ANOMALY_TYPES})
    return row


def _runs(mask):
    padded = np.concatenate(([0], np.asarray(mask, dtype=int), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2], edges[1::2]))


def paired(per_seed, method, metric):
    """S0-yerel'e karşı aynı seed (aynı enjeksiyon) eşleştirilmiş fark: küme başına ort, se, %95 GA."""
    rows = []
    for set_name in SET_KEYS:
        part = per_seed[per_seed["set"] == set_name].pivot(index="seed", columns="method", values=metric)
        diffs = (part[method] - part["S0-yerel"]).dropna()
        n = len(diffs)
        mean, se = float(diffs.mean()), float(diffs.std(ddof=1) / np.sqrt(n))
        t = stats.t.ppf(0.975, n - 1)
        rows.append({"set": set_name, "metric": metric, "diff_mean": mean, "diff_se": se, "n": n,
                     "ci_low": mean - t * se, "ci_high": mean + t * se})
    return rows


def decide(summary, paired_table, method):
    """Protokol §1 kabul ölçütü (6 koşul)."""
    s = summary.set_index(["method", "set"])
    p = paired_table.set_index(["metric", "set"])
    w, y = "winter_dev", "summer_dev"
    both = lambda metric: (p.loc[(metric, w), "diff_mean"] + p.loc[(metric, y), "diff_mean"]) / 2
    both_se = np.sqrt(p.loc[("block_f1", w), "diff_se"] ** 2 + p.loc[("block_f1", y), "diff_se"] ** 2) / 2
    checks = {
        "c1_block_f1_gain_beyond_2se": both("block_f1") > 2 * both_se,
        "c2_block_f1_gain_both_seasons": all(p.loc[("block_f1", k), "diff_mean"] > 0 for k in (w, y)),
        "c3_false_alarm_increase_le_0.05": all(p.loc[("false_alarms_per_day", k), "diff_mean"] <= 0.05 for k in (w, y)),
        "c4_level_pattern_loss_le_0.05": all(
            s.loc[("S0-yerel", k), f"event_recall_{t}"] - s.loc[(method, k), f"event_recall_{t}"] <= 0.05
            for k in (w, y) for t in ("level_shift", "pattern")),
        "c5_count_f1_gain": both("count_f1") > 0,
        "c6_summer_pattern_recall_up": s.loc[(method, y), "event_recall_pattern"] > s.loc[("S0-yerel", y), "event_recall_pattern"],
    }
    return {"method": method, "both_block_f1_diff": both("block_f1"), "both_block_f1_diff_se": both_se,
            "both_count_f1_diff": both("count_f1"), **checks, "accepted": all(checks.values())}


def main():
    setup_console()
    argparse.ArgumentParser(description=__doc__).parse_args()
    config = load_config()
    set_seed(config["seed"])
    tables_dir = resolve_path(PROJECT_ROOT, config["paths"]["tables_dir"])
    log = Log(resolve_path(PROJECT_ROOT, "results/logs/E1_sustained.log"))

    # 1. Ortak sabit başlangıç: ağırlık/veri/config hash'leri manifestoyla aynı olmalı.
    manifest = pd.read_csv(tables_dir / "repro_manifest.csv").set_index("item")
    for item in MANIFEST_ITEMS:
        path = PROJECT_ROOT / manifest.loc[item, "path"]
        if sha256(path) != manifest.loc[item, "sha256"]:
            raise RuntimeError(f"{item} hash'i manifestoyla uyuşmuyor; ortak başlangıç bozuldu.")
    log(f"[1/5] manifest denetimi tamam ({len(MANIFEST_ITEMS)} dosya)")

    df = pd.read_parquet(resolve_path(PROJECT_ROOT, config["paths"]["processed_file"]))
    target = config["preprocess"]["target"]
    data, scales = {}, {}
    for i, set_name in enumerate(SET_KEYS, start=1):
        prepared = prepare_set(df, config, set_name, "cpu")
        change_tab = change_table(prepared["splits"]["train"], target)
        scales[set_name] = train_scales(prepared, change_tab)
        outputs = score_copies(prepared, set_name, "medium", change_tab, "cpu")
        # Enjeksiyonsuz kopya: etiketsiz arka plan alarmları için (metrikten çıkarılmaz).
        cfg, splits, key = prepared["config"], prepared["splits"], prepared["key"]
        model, checkpoint = prepared["fitted"]["forecasters"][cfg["anomaly"]["residual"]["seq2seq"]]
        full = with_context(splits, key, splits[key], cfg)
        clean = signal_frame(full, multi_step_predictions(model, checkpoint, full, "cpu"),
                             prepared["fitted"]["multi_std"], change_tab, target)
        data[set_name] = {"outputs": outputs, "clean": clean, "frame": splits[key]}
        log(f"[2/5] puanlandı {i}/{len(SET_KEYS)}: {set_name}, {len(outputs)} kopya; ölçek {scales[set_name]}")
        if log.over_budget():
            log("SÜRE DOLDU: puanlama sonrası durduruldu")
            return

    def series_for(method, window):
        """Her küme için [(seed, types, labels, score, valid)] (yalnız değerlendirilen saatler ekseni)."""
        result = {}
        for set_name, item in data.items():
            rows = []
            for seed, signals, labels, injected in item["outputs"]:
                score = scores_for(signals, scales[set_name], window)[method].loc[injected.index].to_numpy(dtype=float)
                valid = ~np.isnan(score) & (injected["is_imputed"].to_numpy() == 0)
                rows.append((seed, labels["anomaly_type"].to_numpy(), labels["is_anomaly"].to_numpy(), score, valid))
            result[set_name] = rows
        return result

    # 2. Adayların eşiği (iki kümenin 20 kopyası birlikte, blok tanımlı olay F1) ve aday seçimi.
    settings = config["protocol"]["threshold_quantiles"]
    quantiles = np.linspace(settings["low"], settings["high"], settings["n"])
    candidates = []
    for k, window in enumerate(WINDOWS, start=1):
        method = f"E1-w{window}"
        series = series_for(method, window)
        copies = [(t, s, v) for rows in series.values() for _, t, _, s, v in rows]
        threshold, _ = event_f1_max_threshold(copies, quantiles)
        f1s = [event_metrics(t, (s >= threshold) & v)["event_f1"] for t, s, v in copies]
        candidates.append({"method": method, "window": window, "threshold": threshold,
                           "selection_block_f1": float(np.mean(f1s))})
        log(f"[3/5] aday {k}/{len(WINDOWS)} {method}: eşik {threshold:.4f}, seçim block_f1 {np.mean(f1s):.4f}")
    candidates = pd.DataFrame(candidates).sort_values(["selection_block_f1", "window"], ascending=[False, True])
    chosen = candidates.iloc[0]
    log(f"seçilen aday: {chosen['method']}")
    if log.over_budget():
        log("SÜRE DOLDU: aday seçimi sonrası durduruldu")
        candidates.to_csv(tables_dir / "E1_sustained_candidates.csv", index=False)
        return

    # 3. Kopya başına değerlendirme: S0-yerel, iki aday (seçim yalnız yukarıda).
    rows = []
    plans = [("S0-yerel", None, S0_THRESHOLD)] + [(r["method"], int(r["window"]), r["threshold"])
                                                  for _, r in candidates.iterrows()]
    for method, window, threshold in plans:
        series = series_for(method, window)
        for set_name, copies in series.items():
            item = data[set_name]
            clean_score = scores_for(item["clean"], scales[set_name], window)[method].loc[item["frame"].index]
            clean_valid = clean_score.notna().to_numpy() & (item["frame"]["is_imputed"].to_numpy() == 0)
            background = (clean_score.to_numpy() >= threshold) & clean_valid
            for seed, types, labels, score, valid in copies:
                rows.append({"set": set_name, "method": method, "seed": seed, "threshold": threshold,
                             **copy_metrics(types, labels, score, valid, threshold, background)})
    per_seed = pd.DataFrame(rows)
    log(f"[4/5] değerlendirildi: {len(per_seed)} kopya × yöntem")

    metrics = [c for c in per_seed.columns if c not in ("set", "method", "seed", "threshold")]
    summary = per_seed.groupby(["method", "set"])[metrics].mean().reset_index()
    stds = per_seed.groupby(["method", "set"])[["block_f1", "count_f1"]].std().add_suffix("_std").reset_index()
    summary = summary.merge(stds, on=["method", "set"])
    paired_rows = []
    for method in candidates["method"]:
        for metric in ("block_f1", "count_f1", "block_precision", "count_precision", "event_recall",
                       "false_alarms_per_day", "detection_delay_mean",
                       *[f"event_recall_{k}" for k in ANOMALY_TYPES]):
            paired_rows += [{"method": method, **r} for r in paired(per_seed, method, metric)]
    paired_table = pd.DataFrame(paired_rows)
    decision = pd.DataFrame([decide(summary, paired_table[paired_table["method"] == m], m)
                             for m in candidates["method"]])
    decision["chosen_candidate"] = decision["method"] == chosen["method"]

    per_seed.to_csv(tables_dir / "E1_sustained_per_seed.csv", index=False)
    summary.to_csv(tables_dir / "E1_sustained_summary.csv", index=False)
    paired_table.to_csv(tables_dir / "E1_sustained_paired.csv", index=False)
    candidates.to_csv(tables_dir / "E1_sustained_candidates.csv", index=False)
    decision.to_csv(tables_dir / "E1_sustained_decision.csv", index=False)
    pd.DataFrame([{"set": s, **v} for s, v in scales.items()]).to_csv(tables_dir / "E1_sustained_scales.csv", index=False)

    experiment_rows = []
    for method, part in summary.groupby("method"):
        f1 = part.set_index("set")
        experiment_rows.append({
            "exp_id": f"E1-{method}" if method == "S0-yerel" else method, "date": datetime.now().date().isoformat(),
            "model": f"anomali {method} (A3-s2s-hmean tabanlı; geliştirme)", "seed": config["seed"],
            "note": (f"E1 sürekli işaretli artık (docs/anomaly_e1_sustained.md); orta senaryo block_f1 kış "
                     f"{f1.loc['winter_dev', 'block_f1']:.3f} / yaz {f1.loc['summer_dev', 'block_f1']:.3f}; "
                     f"count_f1 kış {f1.loc['winter_dev', 'count_f1']:.3f} / yaz {f1.loc['summer_dev', 'count_f1']:.3f}; "
                     "yerel ağırlıklar (repro_manifest.csv); val_hold/test kullanılmadı")})
    upsert_csv(resolve_path(PROJECT_ROOT, config["paths"]["experiments_file"]), experiment_rows, ["exp_id"])
    log(f"[5/5] tablolar yazıldı; kabul: {decision.set_index('method')['accepted'].to_dict()}")

    columns = ["method", "set", "block_f1", "count_f1", "block_precision", "count_precision", "event_recall",
               "event_recall_spike", "event_recall_drop", "event_recall_level_shift", "event_recall_pattern",
               "false_alarms_per_day", "detection_delay_mean"]
    print(summary[columns].round(3).to_string(index=False))
    print(decision.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
