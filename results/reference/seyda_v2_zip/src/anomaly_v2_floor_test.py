"""v2 (TEST SONRASI, final değil): A3-s2s-norm + fiziksel taban kanalı, test kümesinde TEK koşu.

Protokol koşudan önce yazıldı: docs/anomaly_v2_floor_test.md.
- Kısa kanal: A3-s2s-norm, val-dev'de seçilmiş pencere ve eşik (anomaly_protocol_selection.csv).
- Taban kanalı: f_t = (y_t < taban) VE (y_{t-1} < taban); taban = train'in 0,5. yüzdeliği.
- Alarm = kısa kanal VEYA taban kanalı. Testte hiçbir şey seçilmez.
Resmî test tablolarının (anomaly_protocol_test*.csv) ÜZERİNE YAZMAZ; çıktılar anomaly_v2_floor_test*.csv.
Komut: python -m src.anomaly_v2_floor_test --device cpu
"""

import argparse

import numpy as np
import pandas as pd
import torch

from src.anomaly_floor_channel import FLOOR_QUANTILE, floor_flags
from src.anomaly_protocol import protocol_splits, score_copies
from src.data.inject_anomalies import ANOMALY_TYPES, train_sigma_table
from src.data.load import PROJECT_ROOT, load_config
from src.evaluate_anomaly import fit_combination, fit_detectors, make_runs, valid_hours
from src.utils.anomaly_metrics import evaluate_threshold, runs
from src.utils.console import setup_console
from src.utils.results_io import resolve_path
from src.utils.seed import set_seed

BASE = "A3-s2s-norm"            # resmî final yöntem
OFFICIAL_MAIN_F1 = 0.3835       # anomaly_protocol_test.csv, ana senaryo (yeniden üretim denetimi)


def main():
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_config()
    set_seed(config["seed"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    target = config["preprocess"]["target"]
    tables_dir = resolve_path(PROJECT_ROOT, config["paths"]["tables_dir"])
    df = pd.read_parquet(resolve_path(PROJECT_ROOT, config["paths"]["processed_file"]))

    # Resmî koşuyla aynı hazırlık: hepsi yalnız train (ve enjeksiyonsuz val-dev referansı) ile.
    splits = protocol_splits(df, config)
    fitted = fit_detectors(splits, config, device, std_source=config["protocol"]["std_source"])
    sigma_table = train_sigma_table(splits["train"], target)
    combination = fit_combination(splits, config, fitted, device,
                                  config["protocol"]["combination_components"], reference_split="val_dev")

    # Dondurulmuş pencere ve eşik (val-dev seçimi); testte seçim yok.
    chosen = pd.read_csv(tables_dir / "anomaly_protocol_selection.csv").set_index("method")
    window, threshold = int(chosen.loc[BASE, "smooth"]), float(chosen.loc[BASE, "threshold"])

    # Taban: train'in doldurulmamış saatlerinin 0,5. yüzdeliği (yalnız train → sızıntı yok).
    train = splits["train"]
    floor = float(train.loc[train["is_imputed"] == 0, target].quantile(FLOOR_QUANTILE))
    clean_flags = floor_flags(splits["test"][target].to_numpy(), floor)
    print(f"Taban = {floor:.3f} kW; enjeksiyonsuz testte taban alarm bloğu: {len(runs(clean_flags))}")

    rows = []
    for scenario in ("main", "hard"):
        outputs = score_copies(splits, config, "test", scenario, fitted, device, combination, sigma_table)
        for (seed, _, labels, injected), run in zip(outputs, make_runs(outputs, {BASE: window})):
            base_score = run["scores"][BASE].to_numpy(dtype=float)       # [N] yumuşatılmış A3 skoru
            valid = valid_hours(run, BASE)                               # resmî koşuyla aynı saatler
            flags = floor_flags(injected[target].to_numpy(), floor)      # [N] bool
            with np.errstate(invalid="ignore"):
                v2_score = np.fmax(base_score / threshold, 2.0 * flags)  # r ≥ 1 ⇔ A3 alarmı VEYA taban
            is_anomaly = labels["is_anomaly"].to_numpy()
            types = labels["anomaly_type"].to_numpy()
            for method, score, thr in ((BASE, base_score, threshold), ("v2-A3+taban", v2_score, 1.0)):
                metrics = evaluate_threshold(is_anomaly, score, types, thr, valid)
                rows.append({"scenario": scenario, "seed": seed, "method": method, **metrics})

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(tables_dir / "anomaly_v2_floor_test_per_seed.csv", index=False)
    metrics = ["event_f1", "event_recall", "event_precision"] + [f"event_recall_{k}" for k in ANOMALY_TYPES]
    summary = per_seed.groupby(["scenario", "method"])[metrics].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(tables_dir / "anomaly_v2_floor_test_summary.csv", index=False)

    # Eşleştirilmiş fark (aynı seed, aynı kopya): v2 − A3-s2s-norm.
    paired = []
    for scenario, part in per_seed.groupby("scenario"):
        wide = part.pivot(index="seed", columns="method", values="event_f1")
        diff = wide["v2-A3+taban"] - wide[BASE]
        paired.append({"scenario": scenario, "n": len(diff), "event_f1_diff_mean": diff.mean(),
                       "event_f1_diff_std": diff.std(), "n_seeds_improved": int((diff > 0).sum())})
    pd.DataFrame(paired).to_csv(tables_dir / "anomaly_v2_floor_test_paired.csv", index=False)

    reproduced = summary.set_index(["scenario", "method"]).loc[("main", BASE), "event_f1_mean"]
    print(f"Yeniden üretim: A3-s2s-norm ana olay F1 = {reproduced:.4f} (resmî {OFFICIAL_MAIN_F1}); "
          f"{'TAMAM' if abs(reproduced - OFFICIAL_MAIN_F1) <= 0.005 else 'YENİDEN ÜRETİLEMEDİ'}")
    print(summary.round(3).to_string(index=False))
    print(pd.DataFrame(paired).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
