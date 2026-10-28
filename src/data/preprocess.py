"""Dakikalık UCI verisini saatlik özellik tablosuna dönüştürür."""

import argparse
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from src.data.load import PROJECT_ROOT, load_config, load_raw_data


MEAN_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
]
SUM_COLUMNS = [
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
    "other_energy",
]


def fill_missing_hours(df, max_gap, lag):
    """Kısa boşlukları interpolasyonla, uzunları geçmişten doldurur."""
    result = df.copy()
    original_missing = result["Global_active_power"].isna()

    # Gelecek gözlemleri kullanan interpolasyon yalnızca eğitimde
    # güvenlidir; burada nedensel (geçmişe dayalı) doldurma uygulanır.
    for col in MEAN_COLUMNS + SUM_COLUMNS:
        series = result[col].copy()
        missing = series.isna()

        # Kısa boşluklar: iki gerçek gözlem arasında doğrusal interpolasyon.
        groups = missing.ne(missing.shift()).cumsum()
        gap_sizes = missing.groupby(groups).transform("sum")
        short = missing & (gap_sizes <= max_gap)

        interpolated = series.interpolate(
            method="time",
            limit_area="inside",
        )
        series.loc[short] = interpolated.loc[short]

        # Uzun boşluklar: yalnızca geçmiş haftanın aynı saati.
        for i in np.flatnonzero(series.isna().to_numpy()):
            if i >= lag and pd.notna(series.iloc[i - lag]):
                series.iloc[i] = series.iloc[i - lag]

        # İlk haftadaki boşluklar veya doldurulamayan durumlar.
        series = series.ffill()
        result[col] = series

    result["is_imputed"] = original_missing.astype("int8")
    return result


def preprocess(df, config):
    settings = config["preprocess"]
    target = settings["target"]

    # Eksik dakika zaman damgalarını da görünür hale getir.
    df = df.asfreq("1min")

    # Alt sayaçlar Wh/dk; diğer enerji de Wh/dk.
    df["other_energy"] = (
        df[target] * 1000 / 60
        - df["Sub_metering_1"]
        - df["Sub_metering_2"]
        - df["Sub_metering_3"]
    )

    counts = df[target].resample("1h").count()
    hourly_mean = df[MEAN_COLUMNS].resample("1h").mean()
    hourly_sum = df[SUM_COLUMNS].resample("1h").sum(min_count=1)

    hourly = pd.concat([hourly_mean, hourly_sum], axis=1)

    # >%50 eksik dakika: saatin bütün ölçümleri geçersiz.
    invalid = counts < (
        60 * (1 - settings["max_missing_ratio_per_hour"])
    )
    hourly.loc[invalid, MEAN_COLUMNS + SUM_COLUMNS] = np.nan

    hourly = fill_missing_hours(
        hourly,
        max_gap=settings["interp_max_gap_hours"],
        lag=settings["long_gap_fill_lag_hours"],
    )

    # Saatlik toplam enerji Wh cinsindedir.
    # Doldurulan değerlerden sonra enerji tutarlılığını yeniden kur.
    hourly["other_energy"] = (
        hourly[target] * 1000
        - hourly["Sub_metering_1"]
        - hourly["Sub_metering_2"]
        - hourly["Sub_metering_3"]
    )

    idx = hourly.index
    hour = idx.hour
    dow = idx.day_of_week
    month = idx.month

    hourly["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    hourly["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    hourly["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    hourly["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    hourly["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    hourly["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    hourly["is_weekend"] = (dow >= 5).astype("int8")

    fr_holidays = holidays.country_holidays(
        settings["holiday_country"],
        years=range(idx.min().year, idx.max().year + 1),
    )
    hourly["is_holiday"] = [
        int(timestamp.date() in fr_holidays)
        for timestamp in idx
    ]

    return hourly


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    raw = load_raw_data(config_path=args.config)

    hourly = preprocess(raw, config)

    output = Path(config["paths"]["processed_file"])
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    output.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(output)

    print("\n=== SAATLİK VERİ ÖZETİ ===")
    print("Satır:", len(hourly))
    print("Sütun:", len(hourly.columns))
    print("Başlangıç:", hourly.index.min())
    print("Bitiş:", hourly.index.max())
    print("Doldurulan saat:", int(hourly["is_imputed"].sum()))
    print("Kalan eksik değer:", int(hourly.isna().sum().sum()))
    print("Kaydedildi:", output)


if __name__ == "__main__":
    main()
