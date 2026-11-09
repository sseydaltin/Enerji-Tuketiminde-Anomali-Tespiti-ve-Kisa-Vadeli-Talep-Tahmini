"""S0 + fiziksel taban (kesinti) kanalı – docs/anomaly_floor_channel.md (geliştirme, final değil).

f_t = (y_t < taban) VE (y_{t−1} < taban); taban = train 0,5. yüzdeliği (yalnız train).
Alarm = (A3-s2s-hmean / q_A3 ≥ 1,1636) VEYA f_t. Orta senaryo + drop duyarlılık senaryosu
(drop değeri = train 5. yüzdeliği). Hiçbir eşik seçilmez. val_hold ve test KULLANILMAZ.
Komut: python -m src.anomaly_floor_channel
"""

import argparse
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd

from src.anomaly_candidates import S0_SCALED_THRESHOLD, decide, summarize
from src.anomaly_dev import SET_KEYS, prepare_set
from src.anomaly_e1_sustained import MANIFEST_ITEMS, Log, copy_metrics, paired
from src.anomaly_eval_audit import sha256
from src.anomaly_short_event import change_table, score_copies, signal_frame
from src.data.inject_anomalies import ANOMALY_TYPES
from src.data.load import PROJECT_ROOT, load_config
from src.evaluate_anomaly import with_context
from src.models.residual_detector import multi_step_predictions
from src.utils.console import setup_console
from src.utils.results_io import resolve_path, upsert_csv
from src.utils.seed import set_seed

FLOOR_QUANTILE = 0.005             # A6 ile aynı taban tanımı (train)
SENSITIVITY_QUANTILE = 0.05        # duyarlılık senaryosu drop değeri (train)
SCENARIOS = ("medium", "drop_sensitivity")


def floor_flags(values, floor, index=None, imputed=None):
    """f_t = (y_t < taban) VE (y_{t−1} < taban); ilk saatin geçmişi yoksa False. Yalnız geçmiş.

    İsteğe bağlı korumalar (v2 ve demo kullanır; varsayılan çağrı eski davranışla aynı):
    - index [N] DatetimeIndex: iki kayıt arası tam 1 saat değilse (zaman boşluğu) önceki kayıt
      "ardışık saat" sayılmaz;
    - imputed [N] bool: doldurulmuş saatin değeri ölçüm olmadığı için "düşük" sayılmaz.
    """
    low = np.asarray(values, dtype=float) < floor                        # NaN → False
    if imputed is not None:
        low &= ~np.asarray(imputed, dtype=bool)
    previous = np.concatenate(([False], low[:-1]))
    if index is not None:
        index = pd.DatetimeIndex(index)
        consecutive = np.concatenate(([False], np.diff(index) == pd.Timedelta(hours=1)))
        previous &= consecutive
    return low & previous


def train_floor(train, target):
    """Taban = train'in doldurulmamış saatlerinin 0,5. yüzdeliği (pandas quantile 0,005; yalnız train)."""
    return float(train.loc[train["is_imputed"] == 0, target].quantile(FLOOR_QUANTILE))


def combined(a, flags, imputed):
    """r = max(a / T1, 2·f); alarm r ≥ 1 ⇔ S0 alarmı VEYA taban alarmı."""
    with np.errstate(invalid="ignore"):
        score = np.fmax(a / S0_SCALED_THRESHOLD, 2.0 * flags)
    return score, np.isfinite(score) & ~imputed


def with_sensitivity(config, train_by_set):
    """Duyarlılık senaryosu: orta senaryo + drop değeri = train 5. yüzdeliği (kümeye göre)."""
    config = deepcopy(config)
    scenarios = config["development"]["scenarios"]
    scenarios["drop_sensitivity"] = deepcopy(scenarios["medium"])
    scenarios["drop_sensitivity"]["injection"]["drop"] = dict(scenarios["medium"]["injection"]["drop"])
    return config


def main():
    setup_console()
    argparse.ArgumentParser(description=__doc__).parse_args()
    base_config = load_config()
    set_seed(base_config["seed"])
    tables_dir = resolve_path(PROJECT_ROOT, base_config["paths"]["tables_dir"])
    log = Log(resolve_path(PROJECT_ROOT, "results/logs/anomaly_floor_channel.log"))
    manifest = pd.read_csv(tables_dir / "repro_manifest.csv").set_index("item")
    for item in MANIFEST_ITEMS:
        if sha256(PROJECT_ROOT / manifest.loc[item, "path"]) != manifest.loc[item, "sha256"]:
            raise RuntimeError(f"{item} hash'i manifestoyla uyuşmuyor.")
    log("[1/4] manifest denetimi tamam")

    df = pd.read_parquet(resolve_path(PROJECT_ROOT, base_config["paths"]["processed_file"]))
    target = base_config["preprocess"]["target"]
    q_a3 = pd.read_csv(tables_dir / "E1_sustained_scales.csv").set_index("set")["A3"]
    rows = {s: [] for s in SCENARIOS}
    levels = []
    for set_name in SET_KEYS:
        config = with_sensitivity(base_config, None)
        prepared = prepare_set(df, config, set_name, "cpu")
        cfg, splits, key = prepared["config"], prepared["splits"], prepared["key"]
        train = splits["train"]
        clean_train = train.loc[train["is_imputed"] == 0, target]
        floor = float(clean_train.quantile(FLOOR_QUANTILE))
        sensitivity_value = float(clean_train.quantile(SENSITIVITY_QUANTILE))
        cfg["development"]["scenarios"]["drop_sensitivity"]["injection"]["drop"]["value_kw"] = sensitivity_value
        levels.append({"set": set_name, "floor_kw": floor, "sensitivity_drop_kw": sensitivity_value,
                       "medium_drop_kw": cfg["development"]["scenarios"]["medium"]["injection"]["drop"]["value_kw"]})
        change_tab = change_table(train, target)
        # Enjeksiyonsuz küme: etiketsiz arka plan alarmları (metrikten çıkarılmaz).
        model, checkpoint = prepared["fitted"]["forecasters"][cfg["anomaly"]["residual"]["seq2seq"]]
        frame = splits[key]
        full = with_context(splits, key, frame, cfg)
        clean = signal_frame(full, multi_step_predictions(model, checkpoint, full, "cpu"),
                             prepared["fitted"]["multi_std"], change_tab, target)
        clean_a = (clean["A3"] / q_a3[set_name]).loc[frame.index].to_numpy(dtype=float)
        clean_imputed = frame["is_imputed"].to_numpy() == 1
        background = {"S0-yerel": (clean_a >= S0_SCALED_THRESHOLD) & np.isfinite(clean_a) & ~clean_imputed}
        score, valid = combined(clean_a, floor_flags(frame[target], floor), clean_imputed)
        background["S0+taban"] = (score >= 1.0) & valid
        log(f"[2/4] {set_name}: taban {floor:.3f} kW, duyarlılık drop değeri {sensitivity_value:.3f} kW; "
            f"temiz veride taban alarm bloğu {int(np.diff(np.r_[0, floor_flags(frame[target], floor).astype(int), 0]).clip(0).sum())}")
        for scenario in SCENARIOS:
            for seed, signals, labels, injected in score_copies(prepared, set_name, scenario, change_tab, "cpu"):
                a = (signals["A3"] / q_a3[set_name]).loc[injected.index].to_numpy(dtype=float)
                imputed = injected["is_imputed"].to_numpy() == 1
                types, is_anomaly = labels["anomaly_type"].to_numpy(), labels["is_anomaly"].to_numpy()
                s0_valid = np.isfinite(a) & ~imputed
                rows[scenario].append({"set": set_name, "method": "S0-yerel", "seed": seed, **copy_metrics(
                    types, is_anomaly, a, s0_valid, S0_SCALED_THRESHOLD, background["S0-yerel"])})
                score, valid = combined(a, floor_flags(injected[target], floor), imputed)
                rows[scenario].append({"set": set_name, "method": "S0+taban", "seed": seed, **copy_metrics(
                    types, is_anomaly, score, valid, 1.0, background["S0+taban"])})
            log(f"[3/4] puanlandı: {set_name} / {scenario}")

    outputs, decision = {}, None
    for scenario in SCENARIOS:
        per_seed = pd.DataFrame(rows[scenario])
        summary = summarize(per_seed)
        paired_rows = [{"method": "S0+taban", **r}
                       for metric in ("block_f1", "count_f1", "block_precision", "count_precision", "event_recall",
                                      "false_alarms_per_day", "detection_delay_mean",
                                      *[f"event_recall_{k}" for k in ANOMALY_TYPES])
                       for r in paired(per_seed, "S0+taban", metric)]
        outputs[scenario] = (per_seed, summary, pd.DataFrame(paired_rows))
        per_seed.to_csv(tables_dir / f"anomaly_floor_{scenario}_per_seed.csv", index=False)
        summary.to_csv(tables_dir / f"anomaly_floor_{scenario}_summary.csv", index=False)
        pd.DataFrame(paired_rows).to_csv(tables_dir / f"anomaly_floor_{scenario}_paired.csv", index=False)

    per_seed, summary, _ = outputs["medium"]
    decision = decide(per_seed, summary, "S0+taban")
    s = summary.set_index(["method", "set"])
    decision["c7_spike_loss_le_0.05"] = all(
        s.loc[("S0-yerel", k), "event_recall_spike"] - s.loc[("S0+taban", k), "event_recall_spike"] <= 0.05
        for k in SET_KEYS)
    sens = outputs["drop_sensitivity"][2]
    sens_f1 = sens[sens["metric"] == "block_f1"].set_index("set")["diff_mean"]
    decision["c8_sensitivity_no_harm"] = bool((sens_f1 >= -0.01).all())
    decision["accepted"] = bool(decision["accepted"] and decision["c7_spike_loss_le_0.05"]
                                and decision["c8_sensitivity_no_harm"])
    pd.DataFrame([decision]).to_csv(tables_dir / "anomaly_floor_decision.csv", index=False)
    pd.DataFrame(levels).to_csv(tables_dir / "anomaly_floor_levels.csv", index=False)
    upsert_csv(resolve_path(PROJECT_ROOT, base_config["paths"]["experiments_file"]), [
        {"exp_id": f"FLOOR-{scenario}", "date": datetime.now().date().isoformat(),
         "model": "anomali S0+taban (deneysel; docs/anomaly_floor_channel.md)", "seed": base_config["seed"],
         "note": "{} block_f1 kış {:.3f} / yaz {:.3f} (S0 {:.3f} / {:.3f}); val_hold/test yok".format(
             scenario, *outputs[scenario][1].set_index(["method", "set"]).loc[
                 [("S0+taban", "winter_dev"), ("S0+taban", "summer_dev"),
                  ("S0-yerel", "winter_dev"), ("S0-yerel", "summer_dev")], "block_f1"])}
        for scenario in SCENARIOS], ["exp_id"])
    log(f"[4/4] kabul: {decision['accepted']}")

    columns = ["method", "set", "block_f1", "count_f1", "block_precision", "count_precision", "event_recall",
               "event_recall_spike", "event_recall_drop", "event_recall_level_shift", "event_recall_pattern",
               "false_alarms_per_day", "detection_delay_mean", "false_blocks_in_background"]
    for scenario in SCENARIOS:
        print(f"== {scenario}")
        print(outputs[scenario][1][columns].round(3).to_string(index=False))
    print(pd.DataFrame([decision]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
