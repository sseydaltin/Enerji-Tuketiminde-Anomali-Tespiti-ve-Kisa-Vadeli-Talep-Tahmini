"""Değerlendirme denetimi: iki olay metriği, etiketsiz arka plan alarmları, yeniden üretim manifestosu.

Seçim YAPMAZ. Kayıtlı pencere/eşik/ağırlıklar aynen uygulanır:
- Aşama 2 yöntemleri (A0, A1, A2, A3-*, A5): results/tables/dev_baseline_selection.csv
- Aşama 3 deneyleri (S0–S5): results/tables/anomaly_short_event_summary.csv (orta senaryo satırları)
Kümeler kış-dev ve yaz-dev, orta senaryo (seçim senaryosu); val_hold ve test kullanılmaz.

Ölçülenler (kopya başına):
- blok tanımı: event_metrics (mevcut "olay F1"; precision = doğru alarm bloğu / alarm bloğu),
- olay-eşleşmeli tanım: event_matched_metrics (precision = yakalanan olay / (olay + yanlış blok)),
- yanlış blokların kaçının ENJEKSİYONSUZ kopyada da aynı saatte alarm verdiği ("arka plan").
  Enjeksiyonsuz veri anomaliden arınmış KABUL EDİLMEZ; bu alarmlar "etiketsiz arka plan alarmı"
  olarak ayrıca sayılır ve metriklerden çıkarılmaz.
Ek teşhis (yalnız S0): bir anomali tipini kusursuz yakalayan "kâhin" eklenseydi olay F1 ne olurdu.

Komut: python -m src.anomaly_eval_audit
Çıktılar: results/tables/anomaly_eval_audit_{per_seed,summary,background,oracle}.csv,
          results/tables/repro_manifest.csv
"""

import argparse
import hashlib
import platform
import sys

import numpy as np
import pandas as pd
import sklearn
import torch

from src.anomaly_dev import SET_KEYS, prepare_set, score_set
from src.anomaly_short_event import (
    EXPERIMENTS,
    change_table,
    combined_score,
    score_copies,
    signal_frame,
)
from src.data.inject_anomalies import ANOMALY_TYPES
from src.data.load import PROJECT_ROOT, load_config
from src.evaluate_anomaly import add_combinations, score_frame, smooth_scores, with_context
from src.models.residual_detector import multi_step_predictions
from src.utils.anomaly_metrics import event_matched_metrics, event_metrics, fill_alarm_gaps, runs
from src.utils.console import setup_console
from src.utils.results_io import resolve_path
from src.utils.seed import set_seed

SCENARIO = "medium"


def alarm_mask(score, valid, threshold, max_gap):
    """[N] alarm maskesi: skor ≥ eşik (geçerli saatler), sonra alarm birleştirme."""
    with np.errstate(invalid="ignore"):
        flagged = (score >= threshold) & valid
    return fill_alarm_gaps(flagged, max_gap) & valid


def copy_row(types, flagged, background):
    """Bir kopyanın iki tanımdaki metrikleri ve yanlış blokların arka plan payı.

    background [N] bool: aynı yöntemin ENJEKSİYONSUZ kopyadaki alarm maskesi.
    """
    block = event_metrics(types, flagged)
    matched = event_matched_metrics(types, flagged)
    anomalous = types != "none"
    false_blocks = [(a, b) for a, b in runs(flagged) if not anomalous[a:b].any()]
    in_background = sum(background[a:b].any() for a, b in false_blocks)
    row = {"block_f1": block["event_f1"], "block_precision": block["event_precision"],
           "event_recall": block["event_recall"], "n_events": block["n_events"],
           "alarm_blocks": block["n_pred_blocks"], **matched,
           "false_blocks_in_background": in_background,
           "false_blocks_new": len(false_blocks) - in_background}
    for kind in ANOMALY_TYPES:
        row[f"event_recall_{kind}"] = block[f"event_recall_{kind}"]
    return row


def stage2_rows(prepared, set_name, selection):
    """Aşama 2 yöntemleri: kayıtlı pencere/eşikle kopya başına satırlar + arka plan alarm sayısı."""
    cfg, splits, key = prepared["config"], prepared["splits"], prepared["key"]
    frame = splits[key]
    clean = score_frame(with_context(splits, key, frame, cfg), cfg, prepared["fitted"], "cpu")
    add_combinations(clean, prepared["combination"])
    rows, background_rows = [], []
    outputs = score_set(prepared, set_name, SCENARIO, "cpu")
    for method, settings in selection.iterrows():
        window, threshold = int(settings["smooth"]), float(settings["threshold"])
        clean_score = smooth_scores(clean[[method]], window)[method].loc[frame.index].to_numpy(dtype=float)
        clean_valid = ~np.isnan(clean_score) & (frame["is_imputed"].to_numpy() == 0)
        background = alarm_mask(clean_score, clean_valid, threshold, 0)
        background_rows.append(background_row(set_name, method, background, frame, clean_valid))
        for seed, scores, labels, injected in outputs:
            score = smooth_scores(scores[[method]], window)[method].loc[injected.index].to_numpy(dtype=float)
            valid = ~np.isnan(score) & (injected["is_imputed"].to_numpy() == 0)
            flagged = alarm_mask(score, valid, threshold, 0)
            rows.append({"set": set_name, "stage": "Aşama 2", "method": method, "seed": seed,
                         **copy_row(labels["anomaly_type"].to_numpy(), flagged, background)})
    return rows, background_rows


def stage3_rows(prepared, set_name, settings, scales):
    """Aşama 3 deneyleri S0–S5: kayıtlı ağırlık/pencere/boşluk/eşikle (ölçek = train OOF q99)."""
    cfg, splits, key = prepared["config"], prepared["splits"], prepared["key"]
    target = cfg["preprocess"]["target"]
    frame = splits[key]
    change_tab = change_table(splits["train"], target)
    model, checkpoint = prepared["fitted"]["forecasters"][cfg["anomaly"]["residual"]["seq2seq"]]
    full = with_context(splits, key, frame, cfg)
    clean_signals = signal_frame(full, multi_step_predictions(model, checkpoint, full, "cpu"),
                                 prepared["fitted"]["multi_std"], change_tab, target)
    outputs = score_copies(prepared, set_name, SCENARIO, change_tab, "cpu")
    rows, background_rows, oracle_rows = [], [], []
    for experiment, row in settings.iterrows():
        weights = {name: float(row[f"w_{name}"]) for name in EXPERIMENTS[experiment]}
        window, gap, threshold = int(row["smooth"]), int(row["gap"]), float(row["threshold"])
        clean_score = smooth_scores(combined_score(clean_signals, scales, weights).to_frame("s"),
                                    window)["s"].loc[frame.index].to_numpy(dtype=float)
        clean_valid = ~np.isnan(clean_score) & (frame["is_imputed"].to_numpy() == 0)
        background = alarm_mask(clean_score, clean_valid, threshold, gap)
        background_rows.append(background_row(set_name, experiment, background, frame, clean_valid))
        for seed, signals, labels, injected in outputs:
            score = smooth_scores(combined_score(signals, scales, weights).to_frame("s"),
                                  window)["s"].loc[injected.index].to_numpy(dtype=float)
            valid = ~np.isnan(score) & (injected["is_imputed"].to_numpy() == 0)
            flagged = alarm_mask(score, valid, threshold, gap)
            types = labels["anomaly_type"].to_numpy()
            rows.append({"set": set_name, "stage": "Aşama 3", "method": experiment, "seed": seed,
                         **copy_row(types, flagged, background)})
            if experiment == "S0":
                oracle_rows += oracle(set_name, seed, types, flagged)
    return rows, background_rows, oracle_rows


def background_row(set_name, method, background, frame, valid):
    """Enjeksiyonsuz kopyadaki alarm blokları: sayı, günlük oran, başlangıç saatleri."""
    blocks = runs(background)
    return {"set": set_name, "method": method, "background_blocks": len(blocks),
            "background_per_day": len(blocks) / (valid.sum() / 24),
            "starts": ";".join(str(frame.index[a]) for a, _ in blocks)}


def oracle(set_name, seed, types, flagged):
    """Teşhis (seçim değil): S0 alarmlarına bir tipin bütün saatleri eklenseydi iki tanımda F1."""
    rows = []
    for kind in ("none", *ANOMALY_TYPES):
        extended = flagged | (types == kind) if kind != "none" else flagged
        rows.append({"set": set_name, "seed": seed, "added_type": kind,
                     "block_f1": event_metrics(types, extended)["event_f1"],
                     "matched_f1": event_matched_metrics(types, extended)["matched_f1"]})
    return rows


def summarize(per_seed):
    """Küme × yöntem ortalamaları, iki mevsim ortalaması ve iki tanımdaki sıra."""
    metrics = ["block_f1", "matched_f1", "block_precision", "matched_precision", "event_recall",
               "alarm_blocks", "detected_events", "false_alarm_blocks", "false_blocks_in_background",
               "false_blocks_new"] + [f"event_recall_{k}" for k in ANOMALY_TYPES]
    means = per_seed.groupby(["stage", "method", "set"])[metrics].mean().unstack("set")
    means.columns = [f"{s}_{m}" for m, s in means.columns]
    for metric in ("block_f1", "matched_f1"):
        means[f"both_{metric}"] = means[[f"{s}_{metric}" for s in SET_KEYS]].mean(axis=1)
    means = means.reset_index()
    for metric in ("block_f1", "matched_f1"):
        means[f"rank_{metric}"] = means.groupby("stage")[f"both_{metric}"].rank(ascending=False, method="min")
    return means


def sha256(path):
    """Dosyanın SHA-256 özeti (yoksa "YOK")."""
    if not path.is_file():
        return "YOK"
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest(config):
    """Bu koşunun girdileri: veri, config, kullanılan ağırlıklar ve paket sürümleri."""
    models_dir = resolve_path(PROJECT_ROOT, config["paths"]["models_dir"])
    files = {"data": resolve_path(PROJECT_ROOT, config["paths"]["processed_file"]),
             "config": PROJECT_ROOT / "configs/config.yaml"}
    names = ["F10-opt", "F5-opt", "A2", "S-F10-L48", "S-F5-L48", "S-A2"]
    names += [f"{n}-oof{k}" for n in ("F10-opt", "F5-opt") for k in (1, 2, 3)]
    names += [f"{n}-oof{k}" for n in ("S-F10-L48", "S-F5-L48") for k in (1, 2)]
    files.update({name: models_dir / f"{name}_best.pt" for name in names})
    rows = [{"item": k, "path": str(v.relative_to(PROJECT_ROOT)), "sha256": sha256(v)} for k, v in files.items()]
    for item, version in (("python", platform.python_version()), ("pandas", pd.__version__),
                          ("numpy", np.__version__), ("torch", torch.__version__),
                          ("scikit-learn", sklearn.__version__), ("platform", sys.platform)):
        rows.append({"item": item, "path": "", "sha256": version})
    return pd.DataFrame(rows)


def main():
    setup_console()
    argparse.ArgumentParser(description=__doc__).parse_args()
    config = load_config()
    set_seed(config["seed"])
    df = pd.read_parquet(resolve_path(PROJECT_ROOT, config["paths"]["processed_file"]))
    tables_dir = resolve_path(PROJECT_ROOT, config["paths"]["tables_dir"])
    selection = pd.read_csv(tables_dir / "dev_baseline_selection.csv").set_index("method")
    summary3 = pd.read_csv(tables_dir / "anomaly_short_event_summary.csv")
    settings3 = summary3[summary3["scenario"] == SCENARIO].set_index("experiment")
    scales = pd.read_csv(tables_dir / "anomaly_short_event_scales.csv").set_index("set")

    rows, background, oracles = [], [], []
    for set_name in SET_KEYS:
        prepared = prepare_set(df, config, set_name, "cpu")
        part, back = stage2_rows(prepared, set_name, selection)
        rows += part
        background += back
        part, back, orc = stage3_rows(prepared, set_name, settings3, scales.loc[set_name].to_dict())
        rows += part
        background += back
        oracles += orc
        print(f"denetlendi: {set_name}")

    per_seed = pd.DataFrame(rows)
    summary = summarize(per_seed)
    oracle_table = pd.DataFrame(oracles).groupby(["set", "added_type"])[["block_f1", "matched_f1"]].mean()
    per_seed.to_csv(tables_dir / "anomaly_eval_audit_per_seed.csv", index=False)
    summary.to_csv(tables_dir / "anomaly_eval_audit_summary.csv", index=False)
    pd.DataFrame(background).to_csv(tables_dir / "anomaly_eval_audit_background.csv", index=False)
    oracle_table.reset_index().to_csv(tables_dir / "anomaly_eval_audit_oracle.csv", index=False)
    manifest(config).to_csv(tables_dir / "repro_manifest.csv", index=False)

    columns = ["stage", "method", "both_block_f1", "both_matched_f1", "rank_block_f1", "rank_matched_f1",
               "winter_dev_block_f1", "winter_dev_matched_f1", "summer_dev_block_f1", "summer_dev_matched_f1"]
    print(summary[columns].round(3).to_string(index=False))
    print(pd.DataFrame(background)[["set", "method", "background_blocks", "background_per_day"]]
          .round(3).to_string(index=False))
    print(oracle_table.round(3).to_string())


if __name__ == "__main__":
    main()
