"""Kısa vadeli enerji tahmini için basit referans modeller."""

import argparse

import numpy as np
import pandas as pd

from src.data.load import PROJECT_ROOT, load_config
from src.data.split import chronological_split
from src.utils.metrics import calculate_metrics, metrics_by_horizon
from src.utils.console import setup_console


def persistence(history, horizon):
    return np.repeat(history[-1], horizon)


def seasonal_naive(history, horizon, period):
    if len(history) < period:
        raise ValueError("Yetersiz geçmiş veri.")
    return np.resize(history[-period:], horizon)


def evaluate(y_true, y_pred):
    scores = calculate_metrics(y_true, y_pred)
    return {"MAE": scores["MAE"], "RMSE": scores["RMSE"]}


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
    horizon_results = []

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

        y_true = np.stack(actual)

        for model, preds in predictions.items():
            y_pred = np.stack(preds)
            scores = calculate_metrics(
                y_true,
                y_pred,
                mape_min_y=config.get("evaluation", {}).get(
                    "mape_min_y",
                    0.1,
                ),
            )
            results.append({
                "split": name,
                "model": model,
                "windows": len(actual),
                **scores,
            })
            horizon_table = metrics_by_horizon(
                y_true,
                y_pred,
                mape_min_y=config.get("evaluation", {}).get(
                    "mape_min_y",
                    0.1,
                ),
            )
            horizon_table.insert(0, "model", model)
            horizon_table.insert(0, "split", name)
            horizon_table["windows"] = len(actual)
            horizon_results.extend(horizon_table.to_dict("records"))

    result = pd.DataFrame(results)
    print("\n=== BASELINE SONUÇLARI ===")
    print(result.to_string(index=False))

    output = PROJECT_ROOT / "results/tables/baseline_results.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        previous = pd.read_csv(output)
        result_keys = set(
            result[["split", "model"]].itertuples(index=False, name=None)
        )
        previous_keys = previous[["split", "model"]].itertuples(
            index=False,
            name=None,
        )
        previous = previous.loc[
            [key not in result_keys for key in previous_keys]
        ]
        result = pd.concat([previous, result], ignore_index=True)

    result.to_csv(output, index=False)
    horizon_output = PROJECT_ROOT / "results/tables/forecast_horizon_metrics.csv"
    horizon_output.parent.mkdir(parents=True, exist_ok=True)
    horizon_result = pd.DataFrame(horizon_results)
    if horizon_output.exists():
        previous_horizon = pd.read_csv(horizon_output)
        result_keys = set(
            horizon_result[["split", "model", "horizon"]].itertuples(
                index=False,
                name=None,
            )
        )
        previous_keys = previous_horizon[
            ["split", "model", "horizon"]
        ].itertuples(index=False, name=None)
        previous_horizon = previous_horizon.loc[
            [key not in result_keys for key in previous_keys]
        ]
        horizon_result = pd.concat(
            [previous_horizon, horizon_result],
            ignore_index=True,
        )
    horizon_result.to_csv(horizon_output, index=False)
    print("\nKaydedildi:", output)


if __name__ == "__main__":
    main()
