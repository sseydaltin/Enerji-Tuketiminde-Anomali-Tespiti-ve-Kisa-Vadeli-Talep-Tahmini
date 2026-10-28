"""Kısa vadeli enerji tahmini için basit referans modeller."""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data.load import PROJECT_ROOT, load_config
from src.data.split import chronological_split
from src.utils.console import setup_console


def persistence(history, horizon):
    return np.repeat(history[-1], horizon)


def seasonal_naive(history, horizon, period):
    if len(history) < period:
        raise ValueError("Yetersiz geçmiş veri.")
    return np.resize(history[-period:], horizon)


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def main():
    """Seçilen kümeyi değerlendirir ve baseline sonuçlarını CSV'ye yazar."""
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
    df = pd.read_parquet(
        PROJECT_ROOT / config["paths"]["processed_file"]
    )

    splits = chronological_split(df, config)
    target = config["preprocess"]["target"]
    lookback = config["forecast"]["lookback"]
    horizon = config["forecast"]["horizon"]

    results = []

    for name in (args.split,):
        part = splits[name]
        start = df.index.get_loc(part.index.min())
        end = df.index.get_loc(part.index.max())

        actual, predictions = [], {
            "Persistence": [],
            "SeasonalNaive24": [],
            "SeasonalNaive168": [],
        }

        for i in range(start, end - horizon + 2):
            if i < lookback:
                continue

            if config["evaluation"]["exclude_imputed_targets"]:
                if df["is_imputed"].iloc[i:i + horizon].any():
                    continue

            history = df[target].iloc[i - lookback:i].to_numpy()
            truth = df[target].iloc[i:i + horizon].to_numpy()

            actual.append(truth)
            predictions["Persistence"].append(
                persistence(history, horizon)
            )
            predictions["SeasonalNaive24"].append(
                seasonal_naive(history, horizon, 24)
            )
            predictions["SeasonalNaive168"].append(
                seasonal_naive(history, horizon, 168)
            )

        if not actual:
            raise ValueError(f"{name} için geçerli pencere bulunamadı.")

        y_true = np.concatenate(actual)

        for model, preds in predictions.items():
            scores = evaluate(y_true, np.concatenate(preds))
            results.append({
                "split": name,
                "model": model,
                "windows": len(actual),
                **scores,
            })

    result = pd.DataFrame(results)
    print("\n=== BASELINE SONUÇLARI ===")
    print(result.to_string(index=False))

    output = PROJECT_ROOT / "results/tables/baseline_results.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        previous = pd.read_csv(output)
        previous = previous.loc[previous["split"] != args.split]
        result = pd.concat([previous, result], ignore_index=True)

    result.to_csv(output, index=False)
    print("\nKaydedildi:", output)


if __name__ == "__main__":
    main()
