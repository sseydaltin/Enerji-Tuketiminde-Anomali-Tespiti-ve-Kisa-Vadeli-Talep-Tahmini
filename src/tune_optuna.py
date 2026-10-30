"""Aşama 4: Optuna ile train içi zamansal çapraz doğrulamayla hiperparametre araması."""

import argparse
from copy import deepcopy
from datetime import date
import time

import numpy as np
import optuna
import pandas as pd
import yaml

from src.data.load import PROJECT_ROOT, load_config
from src.train_forecast import MODEL_LABELS, fit_forecaster, train_model
from src.utils.console import setup_console
from src.utils.results_io import resolve_path, upsert_csv


def cv_folds(train_start, train_end, cv_months, n_folds):
    """Train'in son `cv_months` ayını `n_folds` ardışık doğrulama katına böler.

    Girdi: train başlangıç/bitiş tarihleri (dahil), ay sayısı, kat sayısı.
    Çıktı: [{train_start, train_end, val_start, val_end}] (tarih dizgileri).
    Neden: Genişleyen pencere: her katta model yalnız katın doğrulama
    döneminden ÖNCEKİ veriyle eğitilir (zamanda geriye bakma yok). Gerçek
    val kümesi aramaya hiç girmez; yalnız seçilen ayarı doğrulamak için kalır.
    """
    if cv_months % n_folds:
        raise ValueError("cv_months kat sayısına tam bölünmeli.")
    months_per_fold = cv_months // n_folds
    end = pd.Timestamp(train_end)
    first_val_start = (end + pd.Timedelta(days=1)) - pd.DateOffset(months=cv_months)
    folds = []
    for k in range(n_folds):
        val_start = first_val_start + pd.DateOffset(months=k * months_per_fold)
        val_end = val_start + pd.DateOffset(months=months_per_fold) - pd.Timedelta(days=1)
        folds.append({
            "train_start": str(pd.Timestamp(train_start).date()),
            "train_end": str((val_start - pd.Timedelta(days=1)).date()),
            "val_start": str(val_start.date()),
            "val_end": str(val_end.date()),
        })
    return folds


def fold_config(config, fold):
    """Bir CV katı için split tarihleri değiştirilmiş config kopyası döndürür.

    Test tarihleri olduğu gibi kalır (pencere hazırlığı test'i hiç okumaz).
    """
    config = deepcopy(config)
    config["split"]["train"] = {"start": fold["train_start"], "end": fold["train_end"]}
    config["split"]["val"] = {"start": fold["val_start"], "end": fold["val_end"]}
    return config


def suggest_params(trial, space):
    """Arama uzayından bir aday hiperparametre seti örnekler.

    Girdi: optuna.Trial ve config'teki optuna.space sözlüğü.
    Çıktı: config["forecast"]'a yazılacak sözlük.
    """
    return {
        "hidden": trial.suggest_categorical("hidden", space["hidden"]),
        "layers": trial.suggest_categorical("layers", space["layers"]),
        "lr": trial.suggest_float("lr", *space["lr"], log=True),
        "batch_size": trial.suggest_categorical("batch_size", space["batch_size"]),
        "dropout": trial.suggest_float("dropout", *space["dropout"]),
        "weight_decay": trial.suggest_float("weight_decay", *space["weight_decay"], log=True),
        "loss": trial.suggest_categorical("loss", space["loss"]),
        "shuffle_train": trial.suggest_categorical("shuffle_train", space["shuffle_train"]),
    }


def apply_params(config, params):
    """Aday hiperparametreleri config["forecast"] kopyasına yazar."""
    config = deepcopy(config)
    config["forecast"].update(params)
    return config


def run_study(df, config, model_name, exp_prefix, device):
    """Bir model için Optuna çalışmasını yürütür ve tüm denemeleri kaydeder.

    Girdi: saatlik DataFrame, config (lookback sabit), model adı, exp_id öneki.
    Çıktı: (study, en iyi parametreler, toplam eğitim süresi s).
    Neden: Amaç = 3 katın ortalama MAE'si (kW). Her kattan sonra ara değer
    bildirilir; MedianPruner, aynı adımdaki tamamlanmış denemelerin
    medyanından kötü olanları erken keser.
    """
    settings = config["optuna"]
    split = config["split"]["train"]
    folds = cv_folds(split["start"], split["end"], settings["cv_months"], settings["cv_folds"])
    base = deepcopy(config)
    base["forecast"]["max_epochs"] = settings["cv_max_epochs"]
    rows = []
    total_time = [0.0]

    def objective(trial):
        params = suggest_params(trial, settings["space"])
        trial_config = apply_params(base, params)
        fold_maes, fold_epochs = [], []
        for k, fold in enumerate(folds):
            fitted = fit_forecaster(
                df, fold_config(trial_config, fold), model_name, device,
                label=f"{exp_prefix}-t{trial.number:02d}-fold{k + 1}",
            )
            total_time[0] += fitted["elapsed"]
            fold_maes.append(fitted["best_mae"])
            fold_epochs.append(fitted["best_epoch"])
            trial.set_user_attr("fold_maes", fold_maes)
            trial.set_user_attr("fold_best_epochs", fold_epochs)
            trial.report(float(np.mean(fold_maes)), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(fold_maes))

    sampler = optuna.samplers.TPESampler(seed=settings["sampler_seed"])
    pruner = optuna.pruners.MedianPruner(n_startup_trials=settings["pruner_startup_trials"])
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(
        objective,
        n_trials=settings["n_trials"],
        timeout=settings["timeout_minutes"] * 60,
        catch=(RuntimeError, ValueError),
    )

    for trial in study.trials:
        fold_maes = trial.user_attrs.get("fold_maes", [])
        rows.append({
            "model": model_name,
            "trial": trial.number,
            "state": trial.state.name,
            "cv_mae": trial.value,
            "folds_completed": len(fold_maes),
            "fold_maes": fold_maes,
            "fold_best_epochs": trial.user_attrs.get("fold_best_epochs", []),
            "duration_s": trial.duration.total_seconds() if trial.duration else None,
            **trial.params,
        })
    tables_dir = resolve_path(PROJECT_ROOT, config["paths"]["tables_dir"])
    trials = pd.DataFrame(rows)
    trials.to_csv(tables_dir / f"optuna_{model_name}_trials.csv", index=False)
    pd.DataFrame([{
        "model": model_name,
        "best_trial": study.best_trial.number,
        "cv_mae": study.best_value,
        "n_trials": len(study.trials),
        "n_complete": int((trials["state"] == "COMPLETE").sum()),
        "n_pruned": int((trials["state"] == "PRUNED").sum()),
        "total_train_time_s": total_time[0],
        "folds": folds,
        **study.best_params,
    }]).to_csv(tables_dir / f"optuna_{model_name}_best.csv", index=False)

    # Her deneme ayrı bir konfigürasyon olarak experiments.csv'ye yazılır
    # (val sütunları boş: bu denemeler gerçek val'i görmez).
    label = MODEL_LABELS.get(model_name, model_name.upper())
    experiment_rows = []
    for row in rows:
        experiment_rows.append({
            "exp_id": f"{exp_prefix}-t{row['trial']:02d}",
            "date": date.today().isoformat(),
            "model": label,
            "feature_set": config["features"]["default_set"],
            "lookback": config["forecast"]["lookback"],
            "hidden": row.get("hidden", ""),
            "layers": row.get("layers", ""),
            "dropout": row.get("dropout", ""),
            "lr": row.get("lr", ""),
            "batch": row.get("batch_size", ""),
            "seed": config["seed"],
            "val_mae": "", "val_rmse": "", "test_mae": "", "test_rmse": "",
            "train_time_s": row["duration_s"],
            "note": (
                f"Optuna train-ici CV ({settings['cv_folds']} kat, son "
                f"{settings['cv_months']} ay); state={row['state']}; "
                f"cv_mae={row['cv_mae']}; fold_maes={row['fold_maes']}; "
                f"loss={row.get('loss')}; weight_decay={row.get('weight_decay')}; "
                f"shuffle={row.get('shuffle_train')}; gercek val kullanilmadi"
            ),
        })
    upsert_csv(
        resolve_path(PROJECT_ROOT, config["paths"]["experiments_file"]),
        experiment_rows,
        ["exp_id"],
    )
    return study, study.best_params, total_time[0]


def main():
    """Optuna aramasını yapar, en iyi config'i kaydeder ve val'de doğrular."""
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("seq2seq", "lstm"), required=True)
    parser.add_argument("--exp_prefix", required=True, help="ör. F10-opt")
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.lookback is not None:
        config["forecast"]["lookback"] = args.lookback
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    df = pd.read_parquet(resolve_path(PROJECT_ROOT, config["paths"]["processed_file"]))

    start = time.perf_counter()
    study, best_params, train_time = run_study(
        df, config, args.model, args.exp_prefix, args.device
    )
    print(f"\n=== {args.exp_prefix}: en iyi CV MAE {study.best_value:.4f} kW ===")
    print(best_params)
    print(f"Arama süresi: {time.perf_counter() - start:.0f} s (eğitim {train_time:.0f} s)")

    # En iyi ayar tam config olarak saklanır; seed tekrarları bu dosyayı kullanır.
    best_config = apply_params(config, best_params)
    config_path = PROJECT_ROOT / "configs" / f"optuna_{args.model}_best.yaml"
    with open(config_path, "w", encoding="utf-8") as handle:
        handle.write(
            f"# Otomatik üretildi: src/tune_optuna.py --model {args.model} "
            f"(en iyi CV denemesi {study.best_trial.number}). Elle düzenlemeyin.\n"
        )
        yaml.safe_dump(best_config, handle, allow_unicode=True, sort_keys=False)

    # Doğrulama: tam train'de eğit, gerçek val ile erken durdur ve değerlendir.
    scores, _ = train_model(df, best_config, args.model, args.device, exp_id=args.exp_prefix)
    print(f"Val doğrulaması ({args.exp_prefix}): MAE {scores['MAE']:.4f} kW")


if __name__ == "__main__":
    main()
