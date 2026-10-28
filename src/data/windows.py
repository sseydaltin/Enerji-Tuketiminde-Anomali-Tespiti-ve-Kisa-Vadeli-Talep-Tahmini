"""168 saat geçmişten 24 saatlik doğrudan tahmin pencereleri."""

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.load import PROJECT_ROOT, load_config
from src.data.split import chronological_split
from src.utils.console import setup_console


def get_feature_columns(config, feature_set=None):
    choice = feature_set or config["features"]["default_set"]
    target = config["preprocess"]["target"]

    if choice == "A":
        return [target]
    if choice == "B":
        return [target] + config["features"]["calendar"]
    if choice == "C":
        return (
            [target]
            + config["features"]["calendar"]
            + config["features"]["extra"]
        )
    raise ValueError(f"Bilinmeyen özellik seti: {choice}")


def make_windows(
    df, target, features, lookback, horizon,
    stride=1, target_start=None, target_end=None,
    exclude_imputed=False,
):
    """Hedef aralığına göre geçmiş ve gelecek pencerelerini oluşturur."""
    if lookback < 1 or horizon < 1 or stride < 1:
        raise ValueError("Pencere parametreleri pozitif olmalı.")

    values = df[features].to_numpy(dtype=np.float32)
    targets = df[target].to_numpy(dtype=np.float32)
    flags = df["is_imputed"].to_numpy(dtype=bool)

    X, y, times = [], [], []

    for end in range(lookback, len(df) - horizon + 1, stride):
        first_target = df.index[end]
        last_target = df.index[end + horizon - 1]

        if target_start is not None and first_target < pd.Timestamp(target_start):
            continue
        if target_end is not None and last_target > pd.Timestamp(target_end):
            continue

        if exclude_imputed and flags[end:end + horizon].any():
            continue

        X.append(values[end - lookback:end])
        y.append(targets[end:end + horizon])
        times.append(first_target)

    if not X:
        return (
            np.empty((0, lookback, len(features)), dtype=np.float32),
            np.empty((0, horizon), dtype=np.float32),
            pd.DatetimeIndex([]),
        )

    return (
        np.stack(X),
        np.stack(y),
        pd.DatetimeIndex(times),
    )


def build_datasets(df, config):
    splits = chronological_split(df, config)
    features = get_feature_columns(config)
    target = config["preprocess"]["target"]

    scaler = StandardScaler()
    scaler.fit(splits["train"][features])

    scaled = df.copy()
    scaled[features] = scaler.transform(df[features])

    forecast = config["forecast"]
    datasets = {}

    for name, part in splits.items():
        X, _, times = make_windows(
            scaled,
            target,
            features,
            forecast["lookback"],
            forecast["horizon"],
            forecast["stride"],
            target_start=part.index.min(),
            target_end=part.index.max(),
            exclude_imputed=config["evaluation"]["exclude_imputed_targets"],
        )

        # Hedefler orijinal kW ölçeğinde tutulur.
        _, y, _ = make_windows(
            df,
            target,
            features,
            forecast["lookback"],
            forecast["horizon"],
            forecast["stride"],
            target_start=part.index.min(),
            target_end=part.index.max(),
            exclude_imputed=config["evaluation"]["exclude_imputed_targets"],
        )

        datasets[name] = (X, y, times)

    return datasets, scaler, features


def main():
    setup_console()  # Windows konsolunda Türkçe karakterler için UTF-8
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    df = pd.read_parquet(
        PROJECT_ROOT / config["paths"]["processed_file"]
    )

    datasets, scaler, features = build_datasets(df, config)

    print("\n=== 168 -> 24 PENCERELEME ===")
    print("Özellikler:", features)
    print("Scaler fit: yalnızca train")

    for name, (X, y, times) in datasets.items():
        print(
            f"{name}: X={X.shape}, y={y.shape}, "
            f"ilk hedef={times.min() if len(times) else 'YOK'}"
        )


if __name__ == "__main__":
    main()
