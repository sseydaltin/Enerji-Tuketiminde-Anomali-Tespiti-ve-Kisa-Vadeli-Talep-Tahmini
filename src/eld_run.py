"""ELD (şebeke düzeyi) deneyleri: household yöntemleri aynen, yalnız bu veride yeniden eğitilir.

--stage val : F5-opt / F10-opt eğitimi (3 seed), OOF modelleri, val tahminleri (üç seviye),
              anomali penceresi ve olay-F1 eşiğinin val'de seçimi.
--stage test: yalnız kullanıcı onayıyla, bir kez. Val'de sabitlenen her şey aynen uygulanır.
Protokol: docs/eld_protokol.md. Çıktılar results/eld/, models/eld/.
"""

import argparse
from datetime import date

import pandas as pd
import torch

from src.anomaly_protocol import (
    RULE,
    apply_fixed,
    score_copies,
    select_on_dev,
)
from src.data.eld import eld_config
from src.data.inject_anomalies import train_sigma_table
from src.data.load import PROJECT_ROOT
from src.data.split import chronological_split
from src.evaluate_anomaly import fit_combination, fit_detectors, paired_differences, summarize
from src.evaluate_forecast import all_predictions, save_error_figures, summarize_levels
from src.train_forecast import train_model
from src.train_oof import train_oof_models
from src.utils.console import setup_console
from src.utils.results_io import record_forecast_results, resolve_path, upsert_csv

PREFIX = "ELD-"
MODEL_CONFIGS = {   # ana exp_id → (household Optuna config'i, model adı)
    "ELD-F10-opt": ("configs/optuna_seq2seq_best.yaml", "seq2seq"),
    "ELD-F5-opt": ("configs/optuna_lstm_best.yaml", "lstm"),
}
SEED_SUFFIX = {42: "", 7: "-s7", 2026: "-s2026"}
ROSTER = {   # exp_id → (aile, etiket, öznitelik seti, lookback); household kadrosunun alt kümesi
    "ELD-F3-cal": ("ridge", "Ridge", "B", 168),
    **{f"ELD-F5-opt{s}": ("torch", "LSTM", "B", None) for s in SEED_SUFFIX.values()},
    **{f"ELD-F10-opt{s}": ("torch", "Seq2Seq-LSTM", "B", None) for s in SEED_SUFFIX.values()},
    "ELD-F12-L336": ("xgboost", "XGBoost", "B", 336),
}
ENSEMBLE = {f"ELD-F13-ens{s}": (f"ELD-F10-opt{s}", f"ELD-F5-opt{s}") for s in SEED_SUFFIX.values()}
GROUPS = {
    "F1 Seasonal Naive 24": ["ELD-F1"], "F2 Seasonal Naive 168": ["ELD-F2"],
    "F3-cal Ridge": ["ELD-F3-cal"], "F12-L336 XGBoost": ["ELD-F12-L336"],
    "F5-opt LSTM": [f"ELD-F5-opt{s}" for s in SEED_SUFFIX.values()],
    "F10-opt Seq2Seq": [f"ELD-F10-opt{s}" for s in SEED_SUFFIX.values()],
    "F13-ens (final)": [f"ELD-F13-ens{s}" for s in SEED_SUFFIX.values()],
}


def train_deep_models(df, device):
    """F10-opt ve F5-opt'u 3 seed ile ELD train'de eğitir (household hiperparametreleri aynen).

    Checkpoint zaten varsa yeniden eğitilmez (eğitim deterministiktir; tekrar
    koşu aynı ağırlıkları verir).
    """
    for base_id, (config_path, model_name) in MODEL_CONFIGS.items():
        for seed, suffix in SEED_SUFFIX.items():
            config = eld_config(config_path)
            config["seed"] = seed
            exp_id = base_id + suffix
            path = resolve_path(PROJECT_ROOT, config["paths"]["models_dir"]) / f"{exp_id}_best.pt"
            if path.exists():
                print(f"{exp_id}: checkpoint var, eğitim atlandı")
                continue
            train_model(df, config, model_name, device_preference=device, exp_id=exp_id)


def train_oof(df, device):
    """A3'ün OOF std'si için kat modelleri (household düzeni: train'in son 12 ayı, 3 kat)."""
    config = eld_config()
    models_dir = resolve_path(PROJECT_ROOT, config["paths"]["models_dir"])
    if all((models_dir / f"{b}-oof{k}_best.pt").exists() for b in MODEL_CONFIGS for k in (1, 2, 3)):
        print("OOF checkpoint'leri var, eğitim atlandı")
        return
    models = {base_id: (eld_config(path), name) for base_id, (path, name) in MODEL_CONFIGS.items()}
    rows = train_oof_models(df, models, inner_val_months=1, device=device)
    upsert_csv(resolve_path(PROJECT_ROOT, config["paths"]["experiments_file"]), rows, ["exp_id"])


def forecast_stage(df, config, split, device):
    """Kadronun bir kümedeki tahminleri: kayıt, üç seviye tablo (ve testte figürler)."""
    results = all_predictions(df, config, split, device, ROSTER, ENSEMBLE, prefix=PREFIX)
    for exp_id, (y_true, y_pred, times) in results.items():
        if exp_id in ("ELD-F0",):
            continue                                       # Persistence protokolde yok
        label = next((g for g, ids in GROUPS.items() if exp_id in ids), exp_id)
        record_forecast_results(PROJECT_ROOT, config, split, exp_id, label, y_true, y_pred, times,
                                train_time_s=0.0, device=str(device), feature_set="B",
                                lookback="", note=f"ELD {split}; household ayarları aynen")
    tables_dir = resolve_path(PROJECT_ROOT, config["paths"]["tables_dir"])
    per_run, summary = summarize_levels(results, config["evaluation"]["mape_min_y"], GROUPS)
    per_run.to_csv(tables_dir / f"forecast_levels_{split}_per_seed.csv", index=False)
    summary.to_csv(tables_dir / f"forecast_levels_{split}.csv", index=False)
    if split == "test":
        save_error_figures(results, resolve_path(PROJECT_ROOT, config["paths"]["figures_dir"]),
                           "test", prefix=PREFIX, dataset="ELD")
    return summary


def anomaly_stage(df, config, split, device):
    """Anomali: val'de pencere/eşik seçimi; testte val seçimlerinin aynen uygulanması."""
    tables_dir = resolve_path(PROJECT_ROOT, config["paths"]["tables_dir"])
    splits = chronological_split(df, config)
    fitted = fit_detectors(splits, config, device, std_source="oof")
    sigma_table = train_sigma_table(splits["train"], config["preprocess"]["target"])
    combination = fit_combination(splits, config, fitted, device,
                                  config["protocol"]["combination_components"], reference_split="val")
    outputs = score_copies(splits, config, split, "main", fitted, device, combination, sigma_table)
    if split == "val":
        all_windows, windows, thresholds = select_on_dev(outputs, config, split="val")
        all_windows.to_csv(tables_dir / "anomaly_val_windows_per_seed.csv", index=False)
        pd.DataFrame({"method": list(windows), "smooth": list(windows.values()),
                      "threshold": [thresholds[m] for m in windows]}).to_csv(
            tables_dir / "anomaly_selection.csv", index=False)
        per_seed = all_windows[all_windows["smooth"] == all_windows["method"].map(windows)]
    else:
        chosen = pd.read_csv(tables_dir / "anomaly_selection.csv")
        windows = dict(zip(chosen["method"], chosen["smooth"].astype(int)))
        thresholds = dict(zip(chosen["method"], chosen["threshold"]))
        per_seed = apply_fixed(outputs, windows, thresholds, split, "main")
    per_seed.to_csv(tables_dir / f"anomaly_{split}_per_seed.csv", index=False)
    summary = summarize(per_seed, ("split", "scenario", "method", "rule", "smooth"))
    summary.to_csv(tables_dir / f"anomaly_{split}.csv", index=False)
    paired = paired_differences(per_seed, "A0", metrics=("event_f1", "f1", "pr_auc", "event_recall"),
                                rule=RULE)
    paired.to_csv(tables_dir / f"anomaly_{split}_paired_vs_A0.csv", index=False)
    return summary, paired


def main():
    """ELD val veya (onayla) test aşamasını çalıştırır."""
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("val", "test"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = eld_config()
    for key in ("tables_dir", "figures_dir", "predictions_dir", "models_dir"):
        resolve_path(PROJECT_ROOT, config["paths"][key]).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    df = pd.read_parquet(resolve_path(PROJECT_ROOT, config["paths"]["processed_file"]))

    if args.stage == "val":
        train_deep_models(df, args.device)
        train_oof(df, args.device)
    forecast = forecast_stage(df, config, args.stage, device)
    anomaly, paired = anomaly_stage(df, config, args.stage, device)

    pd.set_option("display.width", 250)
    print(forecast.round(4).to_string(index=False))
    columns = ["method", "smooth", "threshold", "n_events_mean", "event_f1_mean", "event_f1_std",
               "event_recall_mean", "event_precision_mean", "f1_mean", "pr_auc_mean", "roc_auc_mean"]
    print(anomaly[columns].round(3).to_string(index=False))
    print(paired.round(3).to_string(index=False))
    print(f"ELD {args.stage} tamamlandı ({date.today().isoformat()}).")


if __name__ == "__main__":
    main()
