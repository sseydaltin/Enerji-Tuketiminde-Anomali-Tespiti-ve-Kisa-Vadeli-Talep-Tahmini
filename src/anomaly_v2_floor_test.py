"""v2 (TEST SONRASI, final değil): A3-s2s-norm + fiziksel taban kanalı, test kümesinde tek koşu.

Kaynak: seyda_v2_guncelleme.zip (orijinali results/reference/seyda_v2_zip/src/). Uyarlamalar:
- Kural src/anomaly_v2.v2_score'dan gelir (demo ile aynı uygulama); taban bayrağı zaman boşluğu ve
  doldurulmuş saat korumasıyla hesaplanır ve korumasız (zip) sürümle birebir aynı olduğu denetlenir.
- --out-dir: çıktılar önce ayrı bir dizine yazılabilir (zip referansıyla karşılaştırma için).
- Enjeksiyonsuz testte taban, A3 ve v2 alarm bloğu sayıları ayrı yazılır (taban kanalının 0 alarmı,
  bütün sistemin 0 yanlış alarmı demek değildir).
- Kullanılan veri/ağırlık/tablo dosyalarının SHA-256 özetleri yazılır.
Kısa kanal: A3-s2s-norm, val-dev'de seçilmiş pencere ve eşik (anomaly_protocol_selection.csv).
Resmî test tablolarının (anomaly_protocol_test*.csv) ÜZERİNE YAZMAZ.
Komut: python -m src.anomaly_v2_floor_test --device cpu [--out-dir results/tables]
"""

import argparse

import numpy as np
import pandas as pd
import torch

from src.anomaly_eval_audit import sha256
from src.anomaly_floor_channel import floor_flags
from src.anomaly_protocol import protocol_splits, score_copies
from src.anomaly_v2 import BASE_METHOD, V2_METHOD, train_floor, v2_score
from src.data.inject_anomalies import ANOMALY_TYPES, train_sigma_table
from src.data.load import PROJECT_ROOT, load_config
from src.evaluate_anomaly import (
    add_combinations,
    fit_combination,
    fit_detectors,
    make_runs,
    score_frame,
    smooth_scores,
    valid_hours,
    with_context,
)
from src.utils.anomaly_metrics import evaluate_threshold, runs
from src.utils.console import setup_console
from src.utils.results_io import resolve_path
from src.utils.seed import set_seed

OFFICIAL_MAIN_F1 = 0.3835       # anomaly_protocol_test.csv, ana senaryo (yeniden üretim denetimi)
TOLERANCE = 0.005
USED_FILES = ["data/processed/hourly.parquet", "configs/config.yaml",
              "results/tables/anomaly_protocol_selection.csv", "models/F10-opt_best.pt",
              "models/F10-opt-oof1_best.pt", "models/F10-opt-oof2_best.pt", "models/F10-opt-oof3_best.pt",
              "models/F5-opt_best.pt", "models/F5-opt-oof1_best.pt", "models/F5-opt-oof2_best.pt",
              "models/F5-opt-oof3_best.pt", "models/A2_best.pt"]


def clean_alarm_counts(splits, config, fitted, device, combination, window, threshold, floor, target):
    """Enjeksiyonsuz testte alarm blokları: yalnız taban, yalnız A3, v2 (A3 VEYA taban)."""
    test = splits["test"]
    scores = score_frame(with_context(splits, "test", test, config), config, fitted, device)
    add_combinations(scores, combination)
    smoothed = smooth_scores(scores[[BASE_METHOD]], window)[BASE_METHOD].loc[test.index].to_numpy(dtype=float)
    valid = np.isfinite(smoothed) & (test["is_imputed"].to_numpy() == 0)
    score, flags = v2_score(smoothed, threshold, test[target].to_numpy(), floor, test.index,
                            test["is_imputed"].to_numpy() == 1)
    with np.errstate(invalid="ignore"):
        a3 = (smoothed >= threshold) & valid
    return {"floor_blocks": len(runs(flags & valid)), "floor_blocks_all_hours": len(runs(flags)),
            "a3_blocks": len(runs(a3)), "v2_blocks": len(runs((score >= 1.0) & valid)), "n_hours": int(valid.sum())}


def main():
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default=None, help="Varsayılan: config tables_dir (results/tables).")
    args = parser.parse_args()

    config = load_config()
    set_seed(config["seed"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    target = config["preprocess"]["target"]
    tables_dir = resolve_path(PROJECT_ROOT, config["paths"]["tables_dir"])
    out_dir = resolve_path(PROJECT_ROOT, args.out_dir) if args.out_dir else tables_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(resolve_path(PROJECT_ROOT, config["paths"]["processed_file"]))

    hashes = pd.DataFrame([{"path": p, "sha256": sha256(PROJECT_ROOT / p)} for p in USED_FILES])
    hashes.to_csv(out_dir / "anomaly_v2_floor_test_inputs.csv", index=False)
    missing = hashes[hashes["sha256"] == "YOK"]
    if len(missing):
        raise FileNotFoundError(f"Eksik girdi: {missing['path'].tolist()}")
    print(f"Girdiler (SHA-256) yazıldı: {len(hashes)} dosya")

    # Resmî koşuyla aynı hazırlık: hepsi yalnız train (ve enjeksiyonsuz val-dev referansı) ile.
    splits = protocol_splits(df, config)
    fitted = fit_detectors(splits, config, device, std_source=config["protocol"]["std_source"])
    sigma_table = train_sigma_table(splits["train"], target)
    combination = fit_combination(splits, config, fitted, device,
                                  config["protocol"]["combination_components"], reference_split="val_dev")

    # Dondurulmuş pencere ve eşik (val-dev seçimi); testte seçim yok.
    chosen = pd.read_csv(tables_dir / "anomaly_protocol_selection.csv").set_index("method")
    window, threshold = int(chosen.loc[BASE_METHOD, "smooth"]), float(chosen.loc[BASE_METHOD, "threshold"])
    floor = train_floor(splits["train"], target)
    print(f"A3-s2s-norm: pencere {window}, eşik {threshold:.4f}; taban (train %0,5) = {floor:.4f} kW")
    clean = clean_alarm_counts(splits, config, fitted, device, combination, window, threshold, floor, target)
    print(f"Enjeksiyonsuz test ({clean['n_hours']} saat): taban kanalı {clean['floor_blocks']} blok, "
          f"A3 {clean['a3_blocks']} blok, v2 {clean['v2_blocks']} blok")

    rows, guard_mismatch = [], 0
    for scenario in ("main", "hard"):
        outputs = score_copies(splits, config, "test", scenario, fitted, device, combination, sigma_table)
        for (seed, _, labels, injected), run in zip(outputs, make_runs(outputs, {BASE_METHOD: window})):
            base_score = run["scores"][BASE_METHOD].to_numpy(dtype=float)       # [N] yumuşatılmış A3
            valid = valid_hours(run, BASE_METHOD)                               # iki yöntemde aynı maske
            values = injected[target].to_numpy()
            v2, flags = v2_score(base_score, threshold, values, floor, injected.index,
                                 injected["is_imputed"].to_numpy() == 1)
            guard_mismatch += int((flags != floor_flags(values, floor)).sum())   # zip (korumasız) ile fark
            is_anomaly = labels["is_anomaly"].to_numpy()
            types = labels["anomaly_type"].to_numpy()
            for method, score, thr in ((BASE_METHOD, base_score, threshold), (V2_METHOD, v2, 1.0)):
                metrics = evaluate_threshold(is_anomaly, score, types, thr, valid)
                rows.append({"scenario": scenario, "seed": seed, "method": method, **metrics})
    print(f"Korumalı ve korumasız taban bayrağı arasındaki fark (tüm kopyalar): {guard_mismatch} saat")

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out_dir / "anomaly_v2_floor_test_per_seed.csv", index=False)
    metrics = ["event_f1", "event_recall", "event_precision"] + [f"event_recall_{k}" for k in ANOMALY_TYPES]
    summary = per_seed.groupby(["scenario", "method"])[metrics].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out_dir / "anomaly_v2_floor_test_summary.csv", index=False)

    paired = []
    for scenario, part in per_seed.groupby("scenario"):
        wide = part.pivot(index="seed", columns="method", values="event_f1")
        diff = wide[V2_METHOD] - wide[BASE_METHOD]
        paired.append({"scenario": scenario, "n": len(diff), "event_f1_diff_mean": diff.mean(),
                       "event_f1_diff_std": diff.std(), "n_seeds_improved": int((diff > 0).sum())})
    pd.DataFrame(paired).to_csv(out_dir / "anomaly_v2_floor_test_paired.csv", index=False)
    pd.DataFrame([{"floor_kw": floor, "window": window, "threshold": threshold,
                   "guard_mismatch_hours": guard_mismatch, **clean}]).to_csv(
        out_dir / "anomaly_v2_floor_test_clean.csv", index=False)

    reproduced = summary.set_index(["scenario", "method"]).loc[("main", BASE_METHOD), "event_f1_mean"]
    ok = abs(reproduced - OFFICIAL_MAIN_F1) <= TOLERANCE
    print(f"Yeniden üretim: A3-s2s-norm ana olay F1 = {reproduced:.4f} (resmî {OFFICIAL_MAIN_F1}); "
          f"{'TAMAM' if ok else 'YENİDEN ÜRETİLEMEDİ'}")
    print(summary.round(3).to_string(index=False))
    print(pd.DataFrame(paired).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
