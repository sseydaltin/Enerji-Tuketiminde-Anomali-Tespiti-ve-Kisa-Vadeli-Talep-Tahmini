"""Saatlik verinin kronolojik olarak bölünmesi."""

import argparse

import pandas as pd

from src.data.load import PROJECT_ROOT, load_config


def chronological_split(df, config):
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DatetimeIndex gerekli.")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Zaman indeksi sıralı olmalı.")

    if df.index.has_duplicates:
        raise ValueError("Tekrarlanan zaman damgaları var.")

    result = {}

    for name in ("train", "val", "test"):
        settings = config["split"][name]

        start = pd.Timestamp(settings["start"])
        end = pd.Timestamp(settings["end"]) + pd.Timedelta(days=1)

        result[name] = df.loc[
            (df.index >= start) & (df.index < end)
        ].copy()

        if result[name].empty:
            raise ValueError(f"{name} kümesi boş.")

    train, val, test = (
        result["train"],
        result["val"],
        result["test"],
    )

    if not (train.index.max() < val.index.min()):
        raise ValueError("Train ve validation çakışıyor.")

    if not (val.index.max() < test.index.min()):
        raise ValueError("Validation ve test çakışıyor.")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    path = PROJECT_ROOT / config["paths"]["processed_file"]
    df = pd.read_parquet(path)

    splits = chronological_split(df, config)

    print("\n=== KRONOLOJİK VERİ BÖLME ===")

    for name, part in splits.items():
        print(
            f"{name:5s} | "
            f"saat={len(part):6d} | "
            f"başlangıç={part.index.min()} | "
            f"bitiş={part.index.max()} | "
            f"doldurulan={int(part['is_imputed'].sum())}"
        )

    print("\nKümeler arasında zaman çakışması yok.")


if __name__ == "__main__":
    main()
