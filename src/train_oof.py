"""Anomali std kalibrasyonu için OOF (kat dışı) tahmin modelleri.

Aşama 4'ün 3 katlı zamansal CV düzeni (train'in son 12 ayı) kullanılır: her
katta model YALNIZ o kattan önceki veriyle eğitilir. Early stopping için
gerçek val ya da katın kendisi değil, kat-öncesi verinin son ayı (iç val)
kullanılır; böylece katın saatleri modele hiçbir yoldan (ağırlık, epoch
seçimi, lr takvimi) girmez. Kat saatlerinin tahmini anomali dalında yapılır
(residual_detector); bu betik yalnız checkpoint'leri yazar.
Final değerlendirmede kullanılan ağırlıklar DEĞİŞMEZ (tüm train ile eğitilmiş
F10-opt / F5-opt); OOF modelleri yalnız artık std'si içindir.
"""

import argparse
from copy import deepcopy
from datetime import date

import pandas as pd
import torch

from src.data.load import PROJECT_ROOT, load_config
from src.train_forecast import checkpoint_payload, fit_forecaster
from src.tune_optuna import cv_folds
from src.utils.console import setup_console
from src.utils.results_io import resolve_path, upsert_csv

# Ana model → (Optuna config'i, model adı).
MODELS = {
    "F10-opt": ("configs/optuna_seq2seq_best.yaml", "seq2seq"),
    "F5-opt": ("configs/optuna_lstm_best.yaml", "lstm"),
}


def inner_fold_config(config, fold, inner_val_months):
    """Kat için eğitim config'i: train = kat öncesi (son ay hariç), val = kat öncesinin son ayı.

    Girdi: config, cv_folds çıktısındaki bir kat, iç val uzunluğu (ay).
    Çıktı: split tarihleri değiştirilmiş config kopyası.
    Neden: Early stopping'in kullandığı veri de kattan önce biter; katın
    saatleri yalnız OOF tahmini için kullanılır.
    """
    fold_train_end = pd.Timestamp(fold["train_end"])
    inner_val_start = fold_train_end + pd.Timedelta(days=1) - pd.DateOffset(months=inner_val_months)
    config = deepcopy(config)
    config["split"]["train"] = {
        "start": fold["train_start"],
        "end": str((inner_val_start - pd.Timedelta(days=1)).date()),
    }
    config["split"]["val"] = {"start": str(inner_val_start.date()), "end": fold["train_end"]}
    return config


def main():
    """Her ana model ve kat için checkpoint'i models/{ana}-oof{k}_best.pt olarak yazar."""
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--inner_val_months", type=int, default=1)
    args = parser.parse_args()

    rows = []
    for base_id, (config_path, model_name) in MODELS.items():
        config = load_config(config_path)
        df = pd.read_parquet(resolve_path(PROJECT_ROOT, config["paths"]["processed_file"]))
        settings = config["optuna"]
        split = config["split"]["train"]
        folds = cv_folds(split["start"], split["end"], settings["cv_months"], settings["cv_folds"])
        models_dir = resolve_path(PROJECT_ROOT, config["paths"]["models_dir"])
        for k, fold in enumerate(folds, start=1):
            exp_id = f"{base_id}-oof{k}"
            fold_cfg = inner_fold_config(config, fold, args.inner_val_months)
            fitted = fit_forecaster(df, fold_cfg, model_name, args.device, label=exp_id)
            payload = checkpoint_payload(fitted["model"], model_name, fitted["prepared"],
                                         fitted["config"]["forecast"])
            payload["oof_fold"] = {"start": fold["val_start"], "end": fold["val_end"]}
            torch.save(payload, models_dir / f"{exp_id}_best.pt")
            forecast = fitted["config"]["forecast"]
            rows.append({
                "exp_id": exp_id, "date": date.today().isoformat(),
                "model": f"{model_name} (OOF kat {k})", "feature_set": "B",
                "lookback": forecast["lookback"], "hidden": forecast["hidden"],
                "layers": forecast["layers"], "dropout": forecast["dropout"],
                "lr": forecast["lr"], "batch": forecast["batch_size"],
                "epochs_run": len(fitted["history"]), "best_epoch": fitted["best_epoch"],
                "seed": config["seed"], "val_mae": "", "val_rmse": "", "test_mae": "",
                "test_rmse": "", "train_time_s": fitted["elapsed"],
                "note": (f"OOF std kalibrasyonu; train {fold_cfg['split']['train']['start']}–"
                         f"{fold_cfg['split']['train']['end']}, iç val (early stopping) "
                         f"{fold_cfg['split']['val']['start']}–{fold_cfg['split']['val']['end']} "
                         f"MAE={fitted['best_mae']:.4f}; OOF kat {fold['val_start']}–{fold['val_end']} "
                         f"modele girmedi; gerçek val/test kullanılmadı"),
            })
            print(f"{exp_id}: en iyi epoch {fitted['best_epoch']}, iç val MAE {fitted['best_mae']:.4f}")
    upsert_csv(resolve_path(PROJECT_ROOT, "results/experiments.csv"), rows, ["exp_id"])
    pd.DataFrame(rows)[["exp_id", "epochs_run", "best_epoch", "train_time_s", "note"]].to_csv(
        resolve_path(PROJECT_ROOT, "results/tables/oof_models.csv"), index=False)


if __name__ == "__main__":
    main()
