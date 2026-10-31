"""Dakikalık UCI verisini saatlik özellik tablosuna dönüştürür."""

import argparse
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from src.data.load import PROJECT_ROOT, load_config, load_raw_data
from src.utils.console import setup_console


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


def fill_missing_hours(df, max_gap, lag, split_boundaries=None):
    """Saatlik boşlukları doldurur; girdisi ve çıktısı aynı indeksli tablodur.

    En fazla max_gap saatlik boşluklar, aynı küme içindeki iki gözlem
    arasında doğrusal interpolasyonla doldurulur. Küme sınırının max_gap
    saatlik çevresindeki boşluklar ve uzun boşluklar lag saat önceki
    değerle doldurulur; kalan baştaki boşluklar geçmiş değerle tamamlanır.
    """
    result = df.copy()
    original_missing = result["Global_active_power"].isna()
    boundaries = sorted(
        pd.Timestamp(boundary) for boundary in (split_boundaries or [])
    )
    protected = np.zeros(len(result), dtype=bool)

    for boundary in boundaries:
        distance = np.abs((result.index - boundary).asi8)
        protected |= distance <= pd.Timedelta(hours=max_gap).value

    segment_starts = [0]
    segment_starts.extend(
        int(result.index.searchsorted(boundary))
        for boundary in boundaries
    )
    segment_starts.append(len(result))

    for col in MEAN_COLUMNS + SUM_COLUMNS:
        series = result[col].copy()

        # Küme sınırının çevresinde interpolasyon yapma; haftalık geçmişi kullan.
        boundary_missing = series.isna().to_numpy() & protected
        for i in np.flatnonzero(boundary_missing):
            if i >= lag and pd.notna(series.iloc[i - lag]):
                series.iloc[i] = series.iloc[i - lag]

        # Her küme içinde ayrı interpolasyon, karşı kümeden gözlem kullanmaz.
        for segment_start, segment_end in zip(
            segment_starts[:-1],
            segment_starts[1:],
        ):
            if segment_start == segment_end:
                continue

            segment = series.iloc[segment_start:segment_end]
            missing = segment.isna()
            groups = missing.ne(missing.shift()).cumsum()
            gap_sizes = missing.groupby(groups).transform("sum")
            short = missing & (gap_sizes <= max_gap)
            short.iloc[protected[segment_start:segment_end]] = False

            interpolated = segment.interpolate(
                method="time",
                limit_area="inside",
            )
            series.iloc[segment_start:segment_end] = segment.where(
                ~short,
                interpolated,
            )

        # Uzun boşluklar: bir hafta önceki aynı saat.
        for i in np.flatnonzero(series.isna().to_numpy()):
            if i >= lag and pd.notna(series.iloc[i - lag]):
                series.iloc[i] = series.iloc[i - lag]

        # İlk haftadaki boşluklar veya doldurulamayan durumlar.
        series = series.ffill()
        result[col] = series

    result["is_imputed"] = original_missing.astype("int8")
    return result


def preprocess(df, config):
    """Dakikalık [N, F] veriyi saatlik [T, F] özellik tablosuna dönüştürür."""
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
        split_boundaries=[
            config["split"][name]["start"]
            for name in ("val", "test")
        ],
    )

    # Saatlik toplam enerji Wh cinsindedir.
    # Doldurulan değerlerden sonra enerji tutarlılığını yeniden kur.
    hourly["other_energy"] = (
        hourly[target] * 1000
        - hourly["Sub_metering_1"]
        - hourly["Sub_metering_2"]
        - hourly["Sub_metering_3"]
    )

    return add_calendar_features(hourly, settings["holiday_country"])


def add_calendar_features(hourly, holiday_country):
    """Saatlik tabloya takvim özniteliklerini ekler (yerinde) ve tabloyu döndürür.

    Girdi: saatlik DatetimeIndex'li tablo, tatil ülkesi kodu (FR, PT, ...).
    Çıktı: hour/dow/month sin-cos, is_weekend, is_holiday sütunları eklenmiş tablo.
    Neden: Takvim önceden bilinir (gelecek bilgisi değildir); household ve ELD
    aynı fonksiyonu kullanır.
    """
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

    country_holidays = holidays.country_holidays(
        holiday_country,
        years=range(idx.min().year, idx.max().year + 1),
    )
    hourly["is_holiday"] = [
        int(timestamp.date() in country_holidays)
        for timestamp in idx
    ]

    return hourly


def main():
    setup_console()  # Windows konsolunda Türkçe karakterler için UTF-8
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
