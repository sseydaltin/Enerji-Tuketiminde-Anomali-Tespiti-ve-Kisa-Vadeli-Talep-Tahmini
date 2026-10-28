"""UCI enerji tüketimi veri setini yükleme ve doğrulama modülü."""

from pathlib import Path
import argparse

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

NUMERIC_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

REQUIRED_COLUMNS = ["Date", "Time", *NUMERIC_COLUMNS]


def load_config(config_path=None):
    """YAML yapılandırmasını yükler."""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs/config.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_raw_data(file_path=None, config_path=None):
    """Ham UCI verisini datetime indeksli DataFrame olarak yükler."""

    config = load_config(config_path)

    if file_path is None:
        file_path = config["paths"]["raw_file"]

    file_path = Path(file_path)

    if not file_path.is_absolute():
        file_path = PROJECT_ROOT / file_path

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Ham veri dosyası bulunamadı: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        sep=";",
        na_values=["?"],
        low_memory=False,
    )

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Eksik sütunlar: {sorted(missing_columns)}"
        )

    datetime_values = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="raise",
    )

    df = df.drop(columns=["Date", "Time"])

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    df.index = pd.DatetimeIndex(
        datetime_values,
        name="datetime",
    )

    df = df.sort_index()

    if df.index.has_duplicates:
        raise ValueError("Tekrarlanan zaman damgaları bulundu.")

    return df


def summarize_raw_data(df):
    """Ham verinin temel kalite bilgilerini döndürür."""

    return {
        "row_count": len(df),
        "start_date": str(df.index.min()),
        "end_date": str(df.index.max()),
        "missing_values": df.isna().sum().to_dict(),
        "missing_ratio_percent": (
            df.isna().mean() * 100
        ).round(4).to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="UCI enerji tüketimi verisini yükle."
    )

    parser.add_argument(
        "--raw-file",
        default=None,
        help="Ham veri dosyasının yolu.",
    )

    parser.add_argument(
        "--config",
        default=None,
        help="YAML yapılandırma dosyasının yolu.",
    )

    args = parser.parse_args()

    df = load_raw_data(
        file_path=args.raw_file,
        config_path=args.config,
    )

    summary = summarize_raw_data(df)

    print("\n=== HAM VERİ ÖZETİ ===")
    print("Satır sayısı:", summary["row_count"])
    print("Başlangıç:", summary["start_date"])
    print("Bitiş:", summary["end_date"])

    print("\nEksik değer sayıları:")
    for column, count in summary["missing_values"].items():
        print(f"  {column}: {count}")

    print("\nEksik değer oranları (%):")
    for column, ratio in summary["missing_ratio_percent"].items():
        print(f"  {column}: {ratio:.4f}")

    print("\nİlk 5 kayıt:")
    print(df.head())


if __name__ == "__main__":
    main()
