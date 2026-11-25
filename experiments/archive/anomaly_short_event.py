"""Aşama 3: A3-s2s-hmean + kısa olay sinyalleri (docs/anomaly_short_event_experiments.md).

Deneyler (birleşik skor = max(A3', w·sinyal', ...), ' = train OOF 99. yüzdeliğine bölünmüş):
  S0 A3 · S1 A3+spike · S2 A3+drop · S3 A3+change · S4 A3+spike+drop · S5 A3+spike+drop+change

Kurallar:
- Kümeler: kış-dev ve yaz-dev (src/anomaly_dev.py ile aynı modeller, std tabloları, enjeksiyonlar).
  val_hold ve test bu modülde HİÇ kullanılmaz.
- Ağırlık, yumuşatma penceresi, alarm birleştirme boşluğu ve eşik yalnız ORTA senaryoda,
  iki kümenin 20 kopyası birlikte seçilir; kolay ve zor aynı ayarlarla ek rapordur.
- Ölçek (99. yüzdelik) ve değişim tablosu yalnız train / OOF saatlerinden.
Komut: python -m src.anomaly_short_event
"""

import argparse
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import average_precision_score

from src.anomaly_dev import SCENARIOS, SET_KEYS, dev_scenario_settings, prepare_set
from src.data.inject_anomalies import ANOMALY_TYPES, inject_anomalies, sigma_for
from src.data.load import PROJECT_ROOT, load_config
from src.evaluate_anomaly import make_runs, multi_horizon_scores, valid_hours, with_context
from src.models.residual_detector import multi_step_predictions, normalized_scores
from src.utils.anomaly_metrics import (
    alarm_diagnostics,
    event_f1_max_threshold,
    event_metrics,
    fill_alarm_gaps,
)
from src.utils.console import setup_console
from src.utils.results_io import resolve_path, upsert_csv
from src.utils.seed import set_seed

# Deney → A3'e eklenen sinyaller (sıra = basitlik sırası).
EXPERIMENTS = {
    "S0": [],
    "S1": ["spike"],
    "S2": ["drop"],
    "S3": ["change"],
    "S4": ["spike", "drop"],
    "S5": ["spike", "drop", "change"],
}
SIGNALS = ("A3", "spike", "drop", "change")
WEIGHTS = (0.5, 0.75, 1.0, 1.5, 2.0)        # ek sinyal ağırlık ızgarası (A3'ün ağırlığı 1)
GAPS = (0, 1, 2)                            # alarm birleştirme boşluğu (saat)
REFERENCE_QUANTILE = 0.99                   # ortak ölçek: train OOF 99. yüzdeliği
SIMPLICITY = {"S0": 0, "S1": 1, "S2": 1, "S3": 1, "S4": 2, "S5": 3}
# Önceden sabitlenen kabul ölçütleri (belge §1 "Karar kuralı").
MAX_FALSE_ALARM_INCREASE = 0.05             # günlük yanlış alarm artışı üst sınırı (her mevsim)
MAX_OTHER_RECALL_LOSS = 0.05                # level_shift / pattern recall kaybı üst sınırı


def change_table(train, target):
    """Saatlik değişimin (Δy_t = y_t − y_{t−1}) saat × gün ortalaması ve std'si (yalnız train).

    Girdi: train tablosu (enjeksiyonsuz), hedef sütunu. Çıktı: [hour, dow, mean, std] (168 satır).
    Neden: Sabah ve akşam tüketim olağan olarak hızla artar/azalır; değişimi kendi saatinin
    olağan değişimine göre ölçmek gerekir. Δy_t yalnız t ve t−1'e bakar (gelecek yok).
    t veya t−1 doldurulmuşsa saat dışarıda kalır.
    """
    delta = train[target].diff()
    imputed = train["is_imputed"].astype(bool)
    valid = delta.notna() & ~imputed & ~imputed.shift(1, fill_value=True)
    delta = delta[valid]
    grouped = delta.groupby([delta.index.hour, delta.index.dayofweek])
    table = pd.DataFrame({"mean": grouped.mean(), "std": grouped.std(ddof=1)})
    table.index.names = ["hour", "dow"]
    return table.reset_index()


def change_scores(frame, table, target):
    """Beklenmeyen değişim: |Δy_t − μ_Δ(saat, gün)| / σ_Δ(saat, gün). İlk saat NaN."""
    delta = frame[target].diff()
    lookup = table.set_index(["hour", "dow"])
    keys = pd.MultiIndex.from_arrays([frame.index.hour, frame.index.dayofweek])
    mean = lookup["mean"].reindex(keys).to_numpy()
    std = lookup["std"].reindex(keys).to_numpy()
    return (delta - mean).abs() / std


def signal_frame(frame, predictions, multi_std, change_tab, target):
    """Bir saatlik tablo için dört ham sinyal: A3, spike, drop, change ([T, 4]).

    Girdi: frame (hedef + is_imputed), predictions [T, 24] (sütun h = t saatinin h adım önceki
    tahmini), multi_std {h: saat×gün std tablosu}, change_tab (change_table çıktısı).
    Neden: z_{h,t} = (y_t − ŷ_{h,t}) / σ_h işaretli normalize artıktır. A3 = ort_h |z_{h,t}|
    (mevcut A3-s2s-hmean ile aynı); z̄_t = ort_h z_{h,t} işareti korur: büyük pozitif → spike,
    büyük negatif → drop. Bir ufuk eksikse sinyal NaN (A3 ile aynı kural).
    """
    signed = pd.DataFrame({
        h: normalized_scores(frame[target] - predictions[h], multi_std[h]) for h in predictions.columns
    })                                                       # [T, 24] işaretli z
    mean_signed = signed.mean(axis=1, skipna=False)          # [T] z̄_t
    return pd.DataFrame({
        "A3": signed.abs().mean(axis=1, skipna=False),
        "spike": mean_signed.clip(lower=0),
        "drop": (-mean_signed).clip(lower=0),
        "change": change_scores(frame, change_tab, target),
    }, index=frame.index)


def reference_scales(prepared, change_tab):
    """Ortak ölçek: her sinyalin train OOF saatlerindeki 99. yüzdeliği ({sinyal: q99}).

    Neden: Sinyaller farklı ölçekte (ör. drop'un kuyruğu spike'tan ince). Referans yalnız
    train'in kat dışı (OOF) saatleri, doldurulmamış ve enjeksiyonsuz; dev kümeleri kullanılmaz.
    """
    train = prepared["splits"]["train"]
    fitted = prepared["fitted"]
    target = prepared["config"]["preprocess"]["target"]
    signals = signal_frame(train, fitted["train_multi_predictions"], fitted["multi_std"], change_tab, target)
    clean = signals[(train["is_imputed"] == 0).to_numpy()].dropna()
    return {name: float(clean[name].quantile(REFERENCE_QUANTILE)) for name in SIGNALS}


def score_copies(prepared, set_name, scenario, change_tab, device):
    """Bir küme × senaryonun bütün kopyaları için ham sinyaller: [(seed, sinyaller, etiketler, kopya)].

    Enjeksiyon src/anomaly_dev.score_set ile aynıdır (aynı ayar, seed, σ). İlk kopyada A3'ün
    mevcut multi_horizon_scores çıktısıyla birebir aynı olduğu denetlenir.
    """
    cfg, splits, key = prepared["config"], prepared["splits"], prepared["key"]
    target = cfg["preprocess"]["target"]
    settings = dev_scenario_settings(cfg, scenario, splits["train"])
    frame = splits[key]
    sigma = sigma_for(frame, prepared["sigma_table"])
    model, checkpoint = prepared["fitted"]["forecasters"][cfg["anomaly"]["residual"]["seq2seq"]]
    outputs = []
    for seed in cfg["development"]["scenarios"][scenario]["seeds"][set_name]:
        injected, labels = inject_anomalies(frame, target, settings, seed, sigma)
        full = with_context(splits, key, injected, cfg)                  # 8 gün enjeksiyonsuz bağlam
        predictions = multi_step_predictions(model, checkpoint, full, device)   # [T, 24]
        signals = signal_frame(full, predictions, prepared["fitted"]["multi_std"], change_tab, target)
        if not outputs:
            existing = multi_horizon_scores(full, cfg, prepared["fitted"], device)["A3-s2s-hmean"]
            gap = (signals["A3"] - existing).abs().max()
            if not gap < 1e-9:
                raise ValueError(f"A3 sinyali A3-s2s-hmean ile aynı değil (fark {gap}).")
        outputs.append((seed, signals, labels, injected))
    return outputs


def combined_score(signals, scales, weights):
    """Birleşik skor = max(A3 / q_A3, w_k · sinyal_k / q_k, ...). Bir bileşen NaN ise NaN.

    Girdi: signals [T, 4], scales {sinyal: q99}, weights {ek sinyal: w} (S0'da boş).
    """
    parts = [signals["A3"] / scales["A3"]]
    parts += [weight * signals[name] / scales[name] for name, weight in weights.items()]
    return pd.concat(parts, axis=1).max(axis=1, skipna=False)


def variant_outputs(outputs_by_set, scales_by_set, weights):
    """Her kümenin kopyalarına bir ağırlık bileşimi uygular: {küme: [(seed, skor tablosu, etiketler, kopya)]}.

    Skor tablosu tek sütunludur ("score") ki mevcut make_runs / valid_hours aynen kullanılsın.
    """
    result = {}
    for set_name, outputs in outputs_by_set.items():
        scales = scales_by_set[set_name]
        result[set_name] = [(seed, combined_score(signals, scales, weights).to_frame("score"), labels, injected)
                            for seed, signals, labels, injected in outputs]
    return result


def weight_grid(experiment):
    """Bir deneyin bütün ağırlık bileşimleri: [{sinyal: w}, ...] (S0: [{}])."""
    names = EXPERIMENTS[experiment]
    return [dict(zip(names, values)) for values in itertools.product(WEIGHTS, repeat=len(names))]


def alarm_mask(run, threshold, max_gap):
    """Alarm maskesi [N]: skor ≥ eşik (geçerli saatlerde), sonra fill_alarm_gaps."""
    scores = run["scores"]["score"].to_numpy(dtype=float)
    valid = valid_hours(run, "score")
    with np.errstate(invalid="ignore"):
        flagged = (scores >= threshold) & valid
    return fill_alarm_gaps(flagged, max_gap) & valid


def copy_metrics(run, threshold, max_gap):
    """Bir kopyada olay metrikleri, tip recall'ü, günlük yanlış alarm, gecikme ve PR-AUC.

    Olay tanımı event_metrics'tekiyle aynı; PR-AUC eşikten ve birleştirmeden bağımsızdır.
    """
    types = run["labels"]["anomaly_type"].to_numpy()
    labels = run["labels"]["is_anomaly"].to_numpy()
    scores = run["scores"]["score"].to_numpy(dtype=float)
    valid = valid_hours(run, "score")
    flagged = alarm_mask(run, threshold, max_gap)
    metrics = event_metrics(types, flagged)
    alarms = alarm_diagnostics(types, flagged, int(valid.sum()))
    metrics.update({k: v for k, v in alarms.items() if k != "delays"})
    metrics["pr_auc"] = average_precision_score(labels[valid], scores[valid])
    return metrics


def select_settings(outputs_by_set, scales_by_set, experiment, config):
    """Orta senaryo, iki küme birlikte: ağırlık × pencere × boşluk ızgarası ve her noktada olay-F1 eşiği.

    Çıktı: (seçilen satır sözlüğü, ızgara tablosu). Ölçüt kopya başına olay F1 ortalaması;
    eşitlikte küçük pencere, küçük boşluk, Σ|log w| küçük.
    """
    settings = config["protocol"]["threshold_quantiles"]
    quantiles = np.linspace(settings["low"], settings["high"], settings["n"])
    rows = []
    for weights in weight_grid(experiment):
        variant = variant_outputs(outputs_by_set, scales_by_set, weights)
        for window in config["anomaly"]["smoothing_windows"]:
            runs_ = [run for outputs in variant.values() for run in make_runs(outputs, {"score": window})]
            copies = [(run["labels"]["anomaly_type"].to_numpy(), run["scores"]["score"].to_numpy(),
                       valid_hours(run, "score")) for run in runs_]
            for max_gap in GAPS:
                threshold, _ = event_f1_max_threshold(copies, quantiles, max_gap)
                f1s = [event_metrics(run["labels"]["anomaly_type"].to_numpy(),
                                     alarm_mask(run, threshold, max_gap))["event_f1"] for run in runs_]
                rows.append({"experiment": experiment, **{f"w_{k}": v for k, v in weights.items()},
                             "smooth": window, "gap": max_gap, "threshold": threshold,
                             "event_f1_mean": float(np.mean(f1s)),
                             "weight_distance": float(sum(abs(np.log(v)) for v in weights.values()))})
    table = pd.DataFrame(rows).sort_values(["event_f1_mean", "smooth", "gap", "weight_distance"],
                                           ascending=[False, True, True, True])
    return table.iloc[0].to_dict(), table


def weights_of(selected, experiment):
    """Seçilen satırdan ağırlık sözlüğü."""
    return {name: float(selected[f"w_{name}"]) for name in EXPERIMENTS[experiment]}


def evaluate_fixed(outputs_by_set, scales_by_set, weights, window, max_gap, threshold, experiment, scenario):
    """Sabit ayarlarla her kümenin her kopyası için metrik satırları (kopya başına)."""
    rows = []
    for set_name, outputs in variant_outputs(outputs_by_set, scales_by_set, weights).items():
        for run in make_runs(outputs, {"score": window}):
            rows.append({"experiment": experiment, "set": set_name, "scenario": scenario, "seed": run["seed"],
                         **copy_metrics(run, threshold, max_gap)})
    return pd.DataFrame(rows)


METRICS = ["event_f1", "event_precision", "event_recall", "pr_auc", "false_alarms_per_day",
           "false_alarms", "detection_delay_mean", "detection_delay_median"] + \
          [f"event_recall_{kind}" for kind in ANOMALY_TYPES]


def paired_ci(diffs):
    """Eşleştirilmiş farkların ortalaması, std'si, se'si ve %95 t güven aralığı."""
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    mean, std = float(diffs.mean()), float(diffs.std(ddof=1))
    se = std / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    return {"diff_mean": mean, "diff_std": std, "diff_se": se, "n": n,
            "ci_low": mean - t * se, "ci_high": mean + t * se}


def paired_table(per_seed, metrics):
    """S0'a karşı aynı seed (= aynı enjeksiyon) üzerinde eşleştirilmiş farklar (küme × senaryo × deney × metrik)."""
    reference = per_seed[per_seed["experiment"] == "S0"].set_index(["set", "scenario", "seed"])
    rows = []
    for (experiment, set_name, scenario), part in per_seed.groupby(["experiment", "set", "scenario"]):
        part = part.set_index(["set", "scenario", "seed"])
        for metric in metrics:
            diffs = part[metric] - reference.loc[part.index, metric]
            rows.append({"experiment": experiment, "set": set_name, "scenario": scenario, "metric": metric,
                         **paired_ci(diffs)})
    return pd.DataFrame(rows)


def both_season_diff(paired, scenario, metric):
    """İki mevsimin eşit ağırlıklı fark ortalaması: ort = (m_k + m_y)/2, se = √(se_k² + se_y²)/2."""
    part = paired[(paired["scenario"] == scenario) & (paired["metric"] == metric)].set_index(["experiment", "set"])
    rows = []
    for experiment in EXPERIMENTS:
        winter, summer = part.loc[(experiment, "winter_dev")], part.loc[(experiment, "summer_dev")]
        mean = (winter["diff_mean"] + summer["diff_mean"]) / 2
        se = np.sqrt(winter["diff_se"] ** 2 + summer["diff_se"] ** 2) / 2
        rows.append({"experiment": experiment, "diff_mean": mean, "diff_se": se,
                     "ci_low": mean - 1.96 * se, "ci_high": mean + 1.96 * se})
    return pd.DataFrame(rows).set_index("experiment")


def decide(by_season, paired, scenario):
    """Önceden sabitlenen karar kuralı (belge §1). Çıktı: deney başına ölçüt tablosu ve seçilen deney."""
    means = by_season[by_season["scenario"] == scenario].set_index(["experiment", "set"])
    f1_diff = both_season_diff(paired, scenario, "event_f1")
    rows = []
    for experiment in EXPERIMENTS:
        season = {s: means.loc[(experiment, s)] for s in SET_KEYS}
        base = {s: means.loc[("S0", s)] for s in SET_KEYS}
        diff = f1_diff.loc[experiment]
        row = {
            "experiment": experiment,
            "both_f1": np.mean([season[s]["event_f1_mean"] for s in SET_KEYS]),
            "both_f1_diff": diff["diff_mean"], "both_f1_diff_se": diff["diff_se"],
            "c1_gain_beyond_noise": diff["diff_mean"] > 2 * diff["diff_se"],
            "c2_gain_both_seasons": all(season[s]["event_f1_mean"] > base[s]["event_f1_mean"] for s in SET_KEYS),
            "c3_false_alarms_ok": all(season[s]["false_alarms_per_day_mean"] - base[s]["false_alarms_per_day_mean"]
                                      <= MAX_FALSE_ALARM_INCREASE for s in SET_KEYS),
            "c4_other_types_ok": all(base[s][f"event_recall_{k}_mean"] - season[s][f"event_recall_{k}_mean"]
                                     <= MAX_OTHER_RECALL_LOSS
                                     for s in SET_KEYS for k in ("level_shift", "pattern")),
        }
        row["accepted"] = experiment != "S0" and all(row[c] for c in row if c.startswith("c"))
        rows.append(row)
    table = pd.DataFrame(rows)
    accepted = table[table["accepted"]]
    if accepted.empty:
        return table, "S0"
    best = accepted.sort_values("both_f1", ascending=False).iloc[0]
    # Daha basit bir kabul edilmiş deneyle farkı gürültü içindeyse (≤ 2·se) basit olan seçilir.
    for _, row in accepted.sort_values("experiment", key=lambda c: c.map(SIMPLICITY)).iterrows():
        if SIMPLICITY[row["experiment"]] >= SIMPLICITY[best["experiment"]]:
            break
        gap = best["both_f1_diff"] - row["both_f1_diff"]
        if abs(gap) <= 2 * max(best["both_f1_diff_se"], row["both_f1_diff_se"]):
            return table, row["experiment"]
    return table, best["experiment"]


def save_figures(by_season, by_type, paired, figures_dir, scenario):
    """Dört figür: olay F1 (mevsim), tip recall'ü, yanlış alarm–F1, S0'a karşı eşleştirilmiş F1 farkı."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    part = by_season[by_season["scenario"] == scenario]
    experiments = list(EXPERIMENTS)

    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.38
    for i, set_name in enumerate(SET_KEYS):
        rows = part[part["set"] == set_name].set_index("experiment").loc[experiments]
        ax.bar(np.arange(len(experiments)) + (i - 0.5) * width, rows["event_f1_mean"], width,
               yerr=rows["event_f1_std"], capsize=3, label=set_name)
    ax.set_xticks(range(len(experiments)), experiments)
    ax.set_xlabel("Deney")
    ax.set_ylabel("Olay F1 (10 seed ort. ± std)")
    ax.set_title(f"Kısa olay sinyalleri – olay F1, kış-dev ve yaz-dev, {scenario} senaryo (geliştirme, test değil)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(figures_dir / f"event_f1_by_season_{scenario}.png", dpi=200)
    plt.close(fig)

    types = by_type[by_type["scenario"] == scenario]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, set_name in zip(axes, SET_KEYS):
        pivot = types[types["set"] == set_name].pivot_table(index="experiment", columns="type",
                                                            values="recall_mean").loc[experiments]
        pivot.plot.bar(ax=ax, rot=0)
        ax.set_title(f"{set_name}, {scenario} senaryo")
        ax.set_xlabel("Deney")
        ax.set_ylabel("Olay recall'ü (10 seed ort.)")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Tip bazında olay recall'ü – kısa olay sinyalleri (geliştirme kümeleri)")
    fig.tight_layout()
    fig.savefig(figures_dir / f"recall_by_type_{scenario}.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for set_name, marker in zip(SET_KEYS, ("o", "s")):
        rows = part[part["set"] == set_name]
        ax.scatter(rows["false_alarms_per_day_mean"], rows["event_f1_mean"], marker=marker, label=set_name)
        for _, row in rows.iterrows():
            ax.annotate(row["experiment"], (row["false_alarms_per_day_mean"], row["event_f1_mean"]),
                        fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Günlük yanlış alarm (alarm/gün)")
    ax.set_ylabel("Olay F1")
    ax.set_title(f"Yanlış alarm – olay F1, {scenario} senaryo (geliştirme)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / f"false_alarms_vs_f1_{scenario}.png", dpi=200)
    plt.close(fig)

    diffs = paired[(paired["scenario"] == scenario) & (paired["metric"] == "event_f1")
                   & (paired["experiment"] != "S0")]
    fig, ax = plt.subplots(figsize=(8, 4))
    others = experiments[1:]
    for i, set_name in enumerate(SET_KEYS):
        rows = diffs[diffs["set"] == set_name].set_index("experiment").loc[others]
        x = np.arange(len(others)) + (i - 0.5) * 0.3
        ax.errorbar(x, rows["diff_mean"], yerr=[rows["diff_mean"] - rows["ci_low"], rows["ci_high"] - rows["diff_mean"]],
                    fmt="o", capsize=4, label=set_name)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(others)), others)
    ax.set_xlabel("Deney")
    ax.set_ylabel("Olay F1 farkı − S0 (%95 GA)")
    ax.set_title(f"S0'a karşı eşleştirilmiş olay F1 farkı, {scenario} senaryo (aynı enjeksiyonlar)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(figures_dir / f"paired_f1_diff_{scenario}.png", dpi=200)
    plt.close(fig)


def check_reproduction(outputs_by_set, scales_by_set, tables_dir):
    """S0-kayıtlı: Aşama 2'nin pencere/eşiğiyle (dev_baseline_selection) olay F1 kayıtlı kopya F1'leriyle aynı mı?

    Ölçek monoton olduğu için kayıtlı eşik q_A3'e bölünerek uygulanır. Çıktı: kopya başına tablo, en büyük fark.
    """
    selection = pd.read_csv(tables_dir / "dev_baseline_selection.csv").set_index("method").loc["A3-s2s-hmean"]
    saved = pd.read_csv(tables_dir / "dev_baseline_per_seed.csv")
    saved = saved[(saved["method"] == "A3-s2s-hmean") & (saved["scenario"] == "medium")]
    rows = []
    for set_name, outputs in variant_outputs(outputs_by_set, scales_by_set, {}).items():
        threshold = selection["threshold"] / scales_by_set[set_name]["A3"]
        for run in make_runs(outputs, {"score": int(selection["smooth"])}):
            rows.append({"set": set_name, "seed": run["seed"], **copy_metrics(run, threshold, 0)})
    table = pd.DataFrame(rows)
    merged = table.merge(saved[["set", "seed", "event_f1"]], on=["set", "seed"], suffixes=("", "_saved"))
    gap = float((merged["event_f1"] - merged["event_f1_saved"]).abs().max())
    return table, gap, len(merged) == len(table)


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
    figures_dir = resolve_path(PROJECT_ROOT, config["paths"]["figures_dir"]) / "anomaly_short_event"
    target = config["preprocess"]["target"]
    selection_scenario = config["development"]["selection_scenario"]

    outputs, scales = {}, {}
    for set_name in SET_KEYS:
        prepared = prepare_set(df, config, set_name, device)
        change_tab = change_table(prepared["splits"]["train"], target)   # yalnız bu kümenin train kesimi
        scales[set_name] = reference_scales(prepared, change_tab)
        for scenario in SCENARIOS:
            outputs[(set_name, scenario)] = score_copies(prepared, set_name, scenario, change_tab, device)
            print(f"puanlandı: {set_name} / {scenario}")
    print("Ölçek (train OOF 99. yüzdelik):", scales)

    by_scenario = {c: {s: outputs[(s, c)] for s in SET_KEYS} for c in SCENARIOS}
    reproduction, gap, complete = check_reproduction(by_scenario[selection_scenario], scales, tables_dir)
    print(f"S0-kayıtlı = Aşama 2 A3-s2s-hmean: en büyük olay F1 farkı {gap:.1e}, tüm kopyalar eşleşti: {complete}")

    selected, grids = {}, []
    for experiment in EXPERIMENTS:
        selected[experiment], grid = select_settings(by_scenario[selection_scenario], scales, experiment, config)
        grids.append(grid)
        print(f"{experiment}: seçilen {selected[experiment]}")

    per_seed = []
    for experiment, row in selected.items():
        weights = weights_of(row, experiment)
        for scenario in SCENARIOS:
            per_seed.append(evaluate_fixed(by_scenario[scenario], scales, weights, int(row["smooth"]),
                                           int(row["gap"]), row["threshold"], experiment, scenario))
    per_seed = pd.concat(per_seed, ignore_index=True)

    grouped = per_seed.groupby(["experiment", "set", "scenario"])[METRICS]
    by_season = grouped.mean().add_suffix("_mean").join(grouped.std().add_suffix("_std")).reset_index()
    paired = paired_table(per_seed, METRICS)
    by_type = []
    for kind in ANOMALY_TYPES:
        column = f"event_recall_{kind}"
        part = by_season[["experiment", "set", "scenario", f"{column}_mean", f"{column}_std"]]
        part = part.rename(columns={f"{column}_mean": "recall_mean", f"{column}_std": "recall_std"})
        diff = paired[paired["metric"] == column][["experiment", "set", "scenario", "diff_mean", "ci_low", "ci_high"]]
        by_type.append(part.merge(diff, on=["experiment", "set", "scenario"]).assign(type=kind))
    by_type = pd.concat(by_type, ignore_index=True)

    decision, chosen = decide(by_season, paired, selection_scenario)
    settings = pd.DataFrame([{"experiment": e, "signals": "+".join(["A3"] + EXPERIMENTS[e]),
                              **{f"w_{k}": v for k, v in weights_of(r, e).items()},
                              "smooth": int(r["smooth"]), "gap": int(r["gap"]), "threshold": r["threshold"],
                              "selection_f1": r["event_f1_mean"]} for e, r in selected.items()])
    summary = []
    for scenario in SCENARIOS:
        part = by_season[by_season["scenario"] == scenario].set_index(["experiment", "set"])
        f1_diff = both_season_diff(paired, scenario, "event_f1")
        for experiment in EXPERIMENTS:
            row = {"scenario": scenario, "experiment": experiment}
            for set_name, short in (("winter_dev", "winter"), ("summer_dev", "summer")):
                values = part.loc[(experiment, set_name)]
                for metric in ("event_f1", "event_precision", "event_recall", "pr_auc", "false_alarms_per_day",
                               "detection_delay_mean", "event_recall_spike", "event_recall_drop"):
                    row[f"{short}_{metric}"] = values[f"{metric}_mean"]
            row["both_event_f1"] = (row["winter_event_f1"] + row["summer_event_f1"]) / 2
            row.update({f"both_f1_{k}": v for k, v in f1_diff.loc[experiment].items()})
            summary.append(row)
    summary = settings.merge(pd.DataFrame(summary), on="experiment")
    summary = summary.merge(decision[["experiment", "accepted"]].assign(scenario=selection_scenario),
                            on=["experiment", "scenario"], how="left")
    false_alarms = by_season[["experiment", "set", "scenario", "false_alarms_mean", "false_alarms_per_day_mean",
                              "false_alarms_per_day_std", "event_precision_mean", "detection_delay_mean_mean",
                              "detection_delay_median_mean"]]
    fa_diff = paired[paired["metric"] == "false_alarms_per_day"][["experiment", "set", "scenario", "diff_mean",
                                                                   "ci_low", "ci_high"]]
    false_alarms = false_alarms.merge(fa_diff, on=["experiment", "set", "scenario"])

    prefix = tables_dir / "anomaly_short_event"
    summary.to_csv(f"{prefix}_summary.csv", index=False)
    by_type.to_csv(f"{prefix}_by_type.csv", index=False)
    by_season.to_csv(f"{prefix}_by_season.csv", index=False)
    false_alarms.to_csv(f"{prefix}_false_alarms.csv", index=False)
    per_seed.to_csv(f"{prefix}_per_seed.csv", index=False)
    paired.to_csv(f"{prefix}_paired.csv", index=False)
    pd.concat(grids, ignore_index=True).to_csv(f"{prefix}_grid.csv", index=False)
    decision.to_csv(f"{prefix}_decision.csv", index=False)
    reproduction.to_csv(f"{prefix}_s0_reproduction.csv", index=False)
    pd.DataFrame([{"set": s, **v} for s, v in scales.items()]).to_csv(f"{prefix}_scales.csv", index=False)
    for scenario in SCENARIOS:
        save_figures(by_season, by_type, paired, figures_dir, scenario)

    medium = summary[summary["scenario"] == selection_scenario]
    rows = [{"exp_id": f"SE-{r['experiment']}", "date": pd.Timestamp.today().date().isoformat(),
             "model": f"anomali {r['signals']} (max, train OOF q99 ölçek)", "feature_set": "", "lookback": "",
             "hidden": "", "layers": "", "dropout": "", "lr": "", "batch": "", "epochs_run": "", "best_epoch": "",
             "seed": config["seed"], "val_mae": "", "val_rmse": "", "test_mae": "", "test_rmse": "",
             "train_time_s": "",
             "note": (f"Aşama 3 kısa olay (geliştirme, test değil); orta senaryo olay F1 kış "
                      f"{r['winter_event_f1']:.3f} / yaz {r['summer_event_f1']:.3f}; smooth={r['smooth']}, "
                      f"gap={r['gap']}; val_hold/test kullanılmadı")} for _, r in medium.iterrows()]
    upsert_csv(resolve_path(PROJECT_ROOT, config["paths"]["experiments_file"]), rows, ["exp_id"])

    print(settings.round(3).to_string(index=False))
    print(medium[["experiment", "winter_event_f1", "summer_event_f1", "both_event_f1", "both_f1_diff_mean",
                  "both_f1_diff_se", "winter_false_alarms_per_day", "summer_false_alarms_per_day"]]
          .round(3).to_string(index=False))
    print(decision.round(3).to_string(index=False))
    print(f"Seçilen: {chosen}")


if __name__ == "__main__":
    main()
