"""Kısa vadeli enerji tahmini için basit referans modeller."""

import argparse
from datetime import date

import numpy as np
import pandas as pd

from src.data.load import PROJECT_ROOT, load_config
from src.data.split import chronological_split
from src.utils.metrics import calculate_metrics
from src.utils.console import setup_console
from src.utils.results_io import (
    record_forecast_results,
    resolve_path,
    upsert_csv,
)


def persistence(history, horizon):
    return np.repeat(history[-1], horizon)


def seasonal_naive(history, horizon, period):
    if len(history) < period:
        raise ValueError("Yetersiz geçmiş veri.")
    return np.resize(history[-period:], horizon)


def evaluate(y_true, y_pred):
    scores = calculate_metrics(y_true, y_pred)
    return {"MAE": scores["MAE"], "RMSE": scores["RMSE"]}


NAIVE_EXPERIMENT_IDS = {
    "Persistence": "F0",
    "SeasonalNaive24": "F1",
    "SeasonalNaive168": "F2",
}


def naive_forecasts(df, part, target, lookback, horizon, exclude_imputed):
    """Bir kümedeki tüm pencereler için üç naive tahmini üretir.

    Girdi: tüm saatlik tablo df (geçmiş için), değerlendirilen küme part.
    Çıktı: y_true [N, horizon] kW, {model: y_pred [N, horizon]} ve
    target_times [N] (her pencerenin ilk hedef zamanı).
    Neden: Hedef penceresi kümenin içinde kalır; geçmiş önceki kümeden
    gelebilir (gelecek bilgisi değil). Doldurulmuş hedefli pencereler
    LSTM/GRU/Ridge ile aynı kuralla atlanır.
    """
    start = df.index.get_loc(part.index.min())
    end = df.index.get_loc(part.index.max())
    actual, times = [], []
    predictions = {name: [] for name in NAIVE_EXPERIMENT_IDS}

    for i in range(start, end - horizon + 2):
        if i < lookback:
            continue
        if exclude_imputed and df["is_imputed"].iloc[i:i + horizon].any():
            continue

        history = df[target].iloc[i - lookback:i].to_numpy()
        actual.append(df[target].iloc[i:i + horizon].to_numpy())
        times.append(df.index[i])
        predictions["Persistence"].append(persistence(history, horizon))
        predictions["SeasonalNaive24"].append(
            seasonal_naive(history, horizon, 24)
        )
        predictions["SeasonalNaive168"].append(
            seasonal_naive(history, horizon, 168)
        )

    if not actual:
        raise ValueError("Geçerli pencere bulunamadı.")
    return (
        np.stack(actual),
        {name: np.stack(values) for name, values in predictions.items()},
        pd.DatetimeIndex(times),
    )


def main():
    """Seçilen kümede naive referansları değerlendirir ve sonuçları yazar.

    Çıktı: results/tables/forecast_{split}_results.csv ve _horizon.csv
    satırları (F0, F1, F2); val için experiments.csv satırları.
    """
    setup_console()  # Windows konsolunda Türkçe karakterler için UTF-8
    parser = argparse.ArgumentParser(
        description="Persistence ve Seasonal Naive referanslarını değerlendir."
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="Değerlendirme kümesi (varsayılan: val).",
    )
    args = parser.parse_args()

    config = load_config()
    df = pd.read_parquet(PROJECT_ROOT / config["paths"]["processed_file"])
    part = chronological_split(df, config)[args.split]
    target = config["preprocess"]["target"]
    lookback = config["forecast"]["lookback"]

    y_true, predictions, times = naive_forecasts(
        df,
        part,
        target,
        lookback,
        config["forecast"]["horizon"],
        config["evaluation"]["exclude_imputed_targets"],
    )

    rows = []
    for model, y_pred in predictions.items():
        exp_id = NAIVE_EXPERIMENT_IDS[model]
        scores = record_forecast_results(
            PROJECT_ROOT, config, args.split, exp_id, model,
            y_true, y_pred, times,
            train_time_s=0.0, device="cpu", feature_set="A",
            lookback=lookback, note="parametresiz referans",
        )
        rows.append({"exp_id": exp_id, "model": model,
                     "windows": len(y_true), **scores})
        if args.split == "val":
            upsert_csv(
                resolve_path(PROJECT_ROOT, config["paths"]["experiments_file"]),
                [{
                    "exp_id": exp_id, "date": date.today().isoformat(),
                    "model": model, "feature_set": "A", "lookback": lookback,
                    "seed": config["seed"],
                    "val_mae": scores["MAE"], "val_rmse": scores["RMSE"],
                    "test_mae": "", "test_rmse": "", "train_time_s": 0.0,
                    "note": "parametresiz referans; device=cpu",
                }],
                ["exp_id"],
            )

    print(f"\n=== NAIVE REFERANSLAR ({args.split}) ===")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
