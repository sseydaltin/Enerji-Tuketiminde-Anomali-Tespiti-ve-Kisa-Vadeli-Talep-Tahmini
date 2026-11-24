"""Yaz-dev (2009-06 – 2009-09) için kat 3'ü HİÇ görmemiş modeller (docs/gelistirme_protokolu.md §2).

Neden ayrı modeller: final protokolün F10-opt / F5-opt ayarları Optuna ile seçildi ve
Optuna'nın CV'si kat 3'ü (= yaz-dev) kullandı. Bu yüzden yaz-dev'de Optuna'dan ÖNCE
sabitlenmiş yapılandırmalar kullanılır: F10-L48 (seq2seq) ve F5-L48 (LSTM) — config
varsayılanları + lookback 48 (lookback F8'de val ile, kat 3 görülmeden seçildi).

Eğitilenler (hepsi yalnız 2009-05-31'e kadarki veriyle; gerçek val/test kullanılmaz):
- S-F10-L48, S-F5-L48: yaz-dev'i puanlayan modeller. Train 2006-12-16 – 2009-04-30,
  early stopping 2009-05 (iç val).
- S-F10-L48-oof{1,2}, S-F5-L48-oof{1,2}: A3 std tabloları için kat dışı modeller
  (katlar 2008-10 – 2009-01 ve 2009-02 – 2009-05; kat 3 yok).
- S-A2: LSTM-AE, train 2006-12-16 – 2009-04-30, early stopping 2009-05.
Komut: python -m src.train_summer_dev [--max_epochs 1]  (1 epoch = smoke test)
"""

import argparse
from copy import deepcopy
from datetime import date

import pandas as pd
import torch

from src.data.load import PROJECT_ROOT, load_config
from src.train_autoencoder import train_autoencoder
from src.train_forecast import checkpoint_payload, fit_forecaster
from src.train_oof import inner_fold_config
from src.tune_optuna import cv_folds
from src.utils.console import setup_console
from src.utils.results_io import resolve_path, upsert_csv

# Yaz-dev tahmin modelleri: ad → (model adı, lookback). Diğer ayarlar config.yaml varsayılanı.
SUMMER_MODELS = {"S-F10-L48": ("seq2seq", 48), "S-F5-L48": ("lstm", 48)}
TRAIN_START = "2006-12-16"


def summer_folds(config):
    """Std kalibrasyonu katları: final protokolün CV katlarından yaz-dev'den ÖNCE bitenler (1 ve 2)."""
    split = config["split"]["train"]
    settings = config["optuna"]
    folds = cv_folds(split["start"], split["end"], settings["cv_months"], settings["cv_folds"])
    summer_start = pd.Timestamp(config["development"]["sets"]["summer_dev"]["start"])
    kept = [fold for fold in folds if pd.Timestamp(fold["val_end"]) < summer_start]
    if len(kept) != 2:
        raise ValueError(f"Beklenen 2 kat, bulunan {len(kept)}.")
    return kept


def model_config(config, lookback, max_epochs):
    """Optuna öncesi yapılandırma: config varsayılanları + lookback (ve smoke test için epoch)."""
    config = deepcopy(config)
    config["forecast"]["lookback"] = lookback
    if max_epochs is not None:
        config["forecast"]["max_epochs"] = max_epochs
    return config


def train_one(df, config, model_name, exp_id, fold, oof_fold, device):
    """Bir modeli kat öncesi veriyle eğitir, checkpoint'i yazar, experiments satırını döndürür.

    Girdi: fold = {"train_start", "train_end"} (eğitim + iç val bu aralıkta),
    oof_fold = {"start", "end"} veya None (puanlama modeli).
    """
    fold_cfg = inner_fold_config(config, fold, inner_val_months=1)   # son ay early stopping
    fitted = fit_forecaster(df, fold_cfg, model_name, device, label=exp_id)
    payload = checkpoint_payload(fitted["model"], model_name, fitted["prepared"],
                                 fitted["config"]["forecast"])
    if oof_fold is not None:
        payload["oof_fold"] = oof_fold
    models_dir = resolve_path(PROJECT_ROOT, config["paths"]["models_dir"])
    torch.save(payload, models_dir / f"{exp_id}_best.pt")
    forecast = fitted["config"]["forecast"]
    train, val = fold_cfg["split"]["train"], fold_cfg["split"]["val"]
    role = f"OOF kat {oof_fold['start']}–{oof_fold['end']} std" if oof_fold else "yaz-dev puanlama"
    print(f"{exp_id}: en iyi epoch {fitted['best_epoch']}, iç val MAE {fitted['best_mae']:.4f}")
    return {
        "exp_id": exp_id, "date": date.today().isoformat(), "model": f"{model_name} ({role})",
        "feature_set": "B", "lookback": forecast["lookback"], "hidden": forecast["hidden"],
        "layers": forecast["layers"], "dropout": forecast["dropout"], "lr": forecast["lr"],
        "batch": forecast["batch_size"], "epochs_run": len(fitted["history"]),
        "best_epoch": fitted["best_epoch"], "seed": config["seed"], "val_mae": "", "val_rmse": "",
        "test_mae": "", "test_rmse": "", "train_time_s": fitted["elapsed"],
        "note": (f"Yaz-dev altyapısı (gelistirme_protokolu §2); Optuna öncesi ayar; train "
                 f"{train['start']}–{train['end']}, iç val {val['start']}–{val['end']} "
                 f"MAE={fitted['best_mae']:.4f}; kat 3 (2009-06–09), gerçek val ve test kullanılmadı"),
    }


def main():
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max_epochs", type=int, default=None, help="Smoke test için (ör. 1).")
    parser.add_argument("--skip_ae", action="store_true")
    args = parser.parse_args()

    config = load_config()
    df = pd.read_parquet(resolve_path(PROJECT_ROOT, config["paths"]["processed_file"]))
    summer_end = config["development"]["summer_train_end"]
    prefix = "SMOKE-" if args.max_epochs is not None else ""
    rows = []
    for name, (model_name, lookback) in SUMMER_MODELS.items():
        cfg = model_config(config, lookback, args.max_epochs)
        # Puanlama modeli: 2009-05-31'e kadarki veri (son ay early stopping).
        rows.append(train_one(df, cfg, model_name, prefix + name,
                              {"train_start": TRAIN_START, "train_end": summer_end}, None, args.device))
        # Std katları 1 ve 2.
        for k, fold in enumerate(summer_folds(config), start=1):
            rows.append(train_one(df, cfg, model_name, f"{prefix}{name}-oof{k}",
                                  {"train_start": fold["train_start"], "train_end": fold["train_end"]},
                                  {"start": fold["val_start"], "end": fold["val_end"]}, args.device))

    if not args.skip_ae:
        ae_cfg = deepcopy(config)
        ae_cfg["split"]["train"] = {"start": TRAIN_START, "end": "2009-04-30"}
        ae_cfg["split"]["val"] = {"start": "2009-05-01", "end": summer_end}   # yalnız early stopping
        if args.max_epochs is not None:
            ae_cfg["anomaly"]["ae"]["max_epochs"] = args.max_epochs
        device = "cpu" if args.device == "cpu" else "cuda"
        train_autoencoder(df, ae_cfg, prefix + "S-A2", device)

    if not prefix:
        upsert_csv(resolve_path(PROJECT_ROOT, config["paths"]["experiments_file"]), rows, ["exp_id"])
        pd.DataFrame(rows)[["exp_id", "epochs_run", "best_epoch", "train_time_s", "note"]].to_csv(
            resolve_path(PROJECT_ROOT, "results/tables/summer_dev_models.csv"), index=False)


if __name__ == "__main__":
    main()
