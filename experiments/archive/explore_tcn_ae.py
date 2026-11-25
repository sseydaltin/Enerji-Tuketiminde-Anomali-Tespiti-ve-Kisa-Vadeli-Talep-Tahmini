"""A7 KEŞİF DENEYİ: TCN-Autoencoder'ı final protokolün val aşamasıyla değerlendirir.

Bu deney final test SONRASINDA eklendi. Bu yüzden:
- Yalnız val kullanılır (val_dev: pencere + eşik seçimi, val_hold: bağımsız ölçüm);
  bu betikte bilerek --split test seçeneği YOKTUR.
- Final yöntem (A3-s2s-norm) ve final tablolar değişmez; çıktılar ayrı
  `A7_explore_*` dosyalarına yazılır.
Kıyas yöntemleri: A0 (referans), A2 (LSTM-AE, doğrudan karşılığı), A3-s2s-norm (final).
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

from src.anomaly_protocol import (
    RULE,
    apply_fixed,
    protocol_splits,
    scenario_settings,
    select_on_dev,
    side_by_side,
    write_markdown,
)
from src.core.inject_anomalies import inject_anomalies, sigma_for, train_sigma_table
from src.core.config import PROJECT_ROOT, load_config
from src.evaluate_anomaly import (
    autoencoder_scores,
    fit_detectors,
    paired_differences,
    score_frame,
    summarize,
    with_context,
)
from src.core.console import setup_console
from src.core.results_io import resolve_path, upsert_csv
from src.core.seed import get_device, set_seed

METHODS = ["A0", "A2", "A3-s2s-norm", "A7"]
EXP_ID = "A7"
# Markdown tablosunun başlığı ve notu betikten yazılır (yeniden çalışınca silinmesin).
TABLE_TITLE = "Keşif deneyi (test sonrası, yalnız val) — final sonuç değildir"
TABLE_NOTE = ("**Not (2026-11-25):** Val-hold sütunları yalnızca bilgi amaçlıdır. "
              "Geliştirme protokolüne (§1) göre yeni yöntemler val-hold'da ölçülmemeliydi. "
              "Bu nedenle A7'nin seçimi ve yorumu yalnızca val-dev'e dayanır.")


def score_copies_with_a7(splits, config, split, scenario, fitted, device, sigma_table, a7_path):
    """anomaly_protocol.score_copies ile aynı; skorlara A7 (TCN-AE, ölçekli kare hata) eklenir.

    Çıktı: [(seed, skorlar [bağlam + küme, METHODS], etiketler, enjekte kopya)].
    Neden: Seed'ler, enjeksiyon ayarları ve bağlam final protokolle birebir aynı;
    böylece A7 diğer yöntemlerle aynı kopyalarda karşılaştırılır.
    """
    target = config["preprocess"]["target"]
    settings = scenario_settings(config, scenario)
    frame = splits[split]
    sigma = sigma_for(frame, sigma_table)
    outputs = []
    for seed in config["protocol"]["scenarios"][scenario]["seeds"][split]:
        injected, labels = inject_anomalies(frame, target, settings, seed, sigma)   # ayrı kopya
        full = with_context(splits, split, injected, config)
        scores = score_frame(full, config, fitted, device)
        scores["A7"], _ = autoencoder_scores(
            a7_path, full[target], config["anomaly"]["ae_tcn"]["window"],
            config["anomaly"]["relative_eps_kw"], device,
        )
        outputs.append((seed, scores[METHODS], labels, injected))
    return outputs


def save_figure(summary, figures_dir):
    """Küme × senaryo başına olay F1 (ort ± std, 10 enjeksiyon seed'i) çubuk grafiği."""
    groups = summary[["split", "scenario"]].drop_duplicates().to_records(index=False)
    figure, axis = plt.subplots(figsize=(9, 5))
    width = 0.8 / len(METHODS)
    for i, method in enumerate(METHODS):
        rows = [summary[(summary["split"] == s) & (summary["scenario"] == c)
                        & (summary["method"] == method)].iloc[0] for s, c in groups]
        axis.bar(np.arange(len(groups)) + i * width, [r["event_f1_mean"] for r in rows], width,
                 yerr=[r["event_f1_std"] for r in rows], capsize=3, label=method)
    axis.set_xticks(np.arange(len(groups)) + width * (len(METHODS) - 1) / 2)
    axis.set_xticklabels([f"{s} / {c}" for s, c in groups])
    axis.set_ylabel("Olay F1 (ort ± std, 10 seed)")
    axis.set_xlabel("Küme / enjeksiyon senaryosu (yalnız val)")
    axis.set_title("A7 TCN-AE keşif deneyi: A0, A2 LSTM-AE, A3-s2s-norm ile kıyas (val)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures_dir / "A7_explore_event_f1.png", dpi=200)
    plt.close(figure)


def main():
    """Yalnız val: val_dev ana senaryoda seçim, aynı pencere/eşikle val_dev zor ve val_hold."""
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = get_device(args.device)
    df = pd.read_parquet(resolve_path(PROJECT_ROOT, config["paths"]["processed_file"]))
    tables_dir = resolve_path(PROJECT_ROOT, config["paths"]["tables_dir"])
    figures_dir = resolve_path(PROJECT_ROOT, config["paths"]["figures_dir"])
    a7_path = resolve_path(PROJECT_ROOT, config["paths"]["models_dir"]) / f"{EXP_ID}_best.pt"
    if not a7_path.exists():
        raise FileNotFoundError(f"{a7_path} yok; önce: python -m src.train_autoencoder --arch tcn --exp A7")

    # select_on_dev yöntem listesini config'ten okur; kopyada yalnız kıyas yöntemleri.
    config = deepcopy(config)
    config["protocol"]["methods"] = METHODS
    splits = protocol_splits(df, config)
    fitted = fit_detectors(splits, config, device, std_source=config["protocol"]["std_source"])
    sigma_table = train_sigma_table(splits["train"], config["preprocess"]["target"])   # yalnız train

    # 1. val_dev ana senaryo: her yöntem için pencere + olay-F1 eşiği (protokolle aynı kural).
    dev_main = score_copies_with_a7(splits, config, "val_dev", "main", fitted, device, sigma_table, a7_path)
    all_windows, windows, thresholds = select_on_dev(dev_main, config)
    per_seed = {("val_dev", "main"): all_windows[all_windows["smooth"]
                                                 == all_windows["method"].map(windows)]}

    # 2. Aynı pencere ve eşiklerle diğer küme/senaryolar (seçim yok).
    for split, scenario in (("val_dev", "hard"), ("val_hold", "main"), ("val_hold", "hard")):
        outputs = score_copies_with_a7(splits, config, split, scenario, fitted, device,
                                       sigma_table, a7_path)
        per_seed[(split, scenario)] = apply_fixed(outputs, windows, thresholds, split, scenario)

    summaries, paired_rows = [], []
    for (split, scenario), table in per_seed.items():
        summaries.append(summarize(table, ("split", "scenario", "method", "rule", "smooth")))
        for reference in ("A2", "A0", "A3-s2s-norm"):
            paired = paired_differences(table, reference, metrics=("event_f1", "pr_auc"), rule=RULE)
            paired_rows.append(paired.assign(split=split, scenario=scenario, reference=reference))
    summary = pd.concat(summaries, ignore_index=True)
    paired = pd.concat(paired_rows, ignore_index=True)
    paired = paired[paired["method"] == "A7"]      # yalnız A7'nin farkları
    pd.concat(per_seed.values(), ignore_index=True).to_csv(
        tables_dir / "A7_explore_val_per_seed.csv", index=False)
    summary.to_csv(tables_dir / "A7_explore_val.csv", index=False)
    paired.to_csv(tables_dir / "A7_explore_val_paired.csv", index=False)
    table = side_by_side(summaries)
    table.to_csv(tables_dir / "A7_explore_side_by_side.csv", index=False)
    write_markdown(table, tables_dir / "A7_explore_side_by_side.md", title=TABLE_TITLE, note=TABLE_NOTE)
    save_figure(summary, figures_dir)

    parts = []
    for part in summaries:
        r = part.set_index("method").loc["A7"]
        parts.append(f"{r['split']}/{r['scenario']} olay F1={r['event_f1_mean']:.3f}±"
                     f"{r['event_f1_std']:.3f}, PR-AUC={r['pr_auc_mean']:.3f}")
    upsert_csv(resolve_path(PROJECT_ROOT, config["paths"]["experiments_file"]), [{
        "exp_id": "P-A7-explore", "date": date.today().isoformat(),
        "model": "TCN-AE (keşif, ölçekli kare hata)", "seed": config["seed"],
        "lookback": windows["A7"], "val_mae": "", "val_rmse": "", "test_mae": "", "test_rmse": "",
        "note": (f"KEŞİF: final test sonrası eklendi, yalnız val; final yöntem değişmez. "
                 f"w={windows['A7']}, olay-F1 eşiği={thresholds['A7']:.4g} (val_dev ana); "
                 + "; ".join(parts)),
    }], ["exp_id"])

    columns = ["split", "scenario", "method", "smooth", "threshold", "event_f1_mean", "event_f1_std",
               "f1_mean", "pr_auc_mean", "roc_auc_mean"]
    print(summary[columns].round(3).to_string(index=False))
    print("\nA7 eşleştirilmiş farklar (A7 − referans):")
    print(paired.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
