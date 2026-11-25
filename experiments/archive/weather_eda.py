"""Aşama 3b EDA notu: günlük ortalama sıcaklık – günlük tüketim ilişkisi (YALNIZ TRAIN)."""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.load import PROJECT_ROOT, load_config
from src.data.split import chronological_split
from src.data.weather import load_weather_frame
from src.utils.console import setup_console
from src.utils.results_io import resolve_path


def daily_table(train, target):
    """Saatlik train tablosundan günlük tüketim (kWh) ve günlük hava özetleri.

    Girdi: saatlik train tablosu (hava sütunları eklenmiş), hedef sütunu (kW).
    Çıktı: [gün, (kwh, temperature, hdh, cdh, is_weekend)] DataFrame.
    Neden: Saatlik ortalama kW × 1 saat = kWh; günlük toplam günlük enerji olur.
    Doldurulmuş saat içeren ve 24 saatten eksik günler dışarıda bırakılır.
    """
    grouped = train.groupby(train.index.date)
    daily = pd.DataFrame({
        "kwh": grouped[target].sum(),
        "temperature": grouped["temperature_2m"].mean(),
        "hdh": grouped["heating_degree_hours"].sum(),
        "cdh": grouped["cooling_degree_hours"].sum(),
        "is_weekend": grouped["is_weekend"].first(),
        "hours": grouped[target].count(),
        "imputed": grouped["is_imputed"].sum(),
    })
    return daily[(daily["hours"] == 24) & (daily["imputed"] == 0)]


def main():
    """Saçılım grafiğini ve korelasyon tablosunu yazar."""
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    target = config["preprocess"]["target"]
    df = pd.read_parquet(resolve_path(PROJECT_ROOT, config["paths"]["processed_file"]))
    df = load_weather_frame(df, config)
    train = chronological_split(df, config)["train"]   # val/test bakılmaz
    daily = daily_table(train, target)

    summary = pd.DataFrame([{
        "split": "train", "days": len(daily),
        "pearson_kwh_temperature": daily["kwh"].corr(daily["temperature"]),
        "spearman_kwh_temperature": daily["kwh"].corr(daily["temperature"], method="spearman"),
        "pearson_kwh_hdh": daily["kwh"].corr(daily["hdh"]),
        "slope_kwh_per_c": np.polyfit(daily["temperature"], daily["kwh"], 1)[0],
        "days_with_cdh": int((daily["cdh"] > 0).sum()),
        "temperature_min": daily["temperature"].min(),
        "temperature_max": daily["temperature"].max(),
    }])
    tables_dir = resolve_path(PROJECT_ROOT, config["paths"]["tables_dir"])
    figures_dir = resolve_path(PROJECT_ROOT, config["paths"]["figures_dir"])
    summary.to_csv(tables_dir / "weather_eda_train.csv", index=False)

    figure, axis = plt.subplots(figsize=(8, 5.5))
    for weekend, color, name in ((0, "tab:blue", "Hafta içi"), (1, "tab:orange", "Hafta sonu")):
        part = daily[daily["is_weekend"] == weekend]
        axis.scatter(part["temperature"], part["kwh"], s=8, alpha=0.5, color=color, label=name)
    # 2 °C'lik kutularda medyan: doğrusal olmayan ilişkiyi gösterir.
    bins = np.arange(np.floor(daily["temperature"].min()), daily["temperature"].max() + 2, 2)
    centers = (bins[:-1] + bins[1:]) / 2
    medians = daily.groupby(pd.cut(daily["temperature"], bins), observed=False)["kwh"].median()
    axis.plot(centers, medians.to_numpy(), color="black", linewidth=2, label="2 °C kutu medyanı")
    axis.axvline(config["weather"]["heating_base_c"], color="gray", linestyle=":",
                 label="Isıtma tabanı 18 °C")
    axis.set_xlabel("Günlük ortalama sıcaklık (°C, ERA5)")
    axis.set_ylabel("Günlük tüketim (kWh)")
    axis.set_title(f"Sıcaklık – günlük tüketim, TRAIN ({len(daily)} gün), "
                   f"Pearson r = {summary.loc[0, 'pearson_kwh_temperature']:.2f}")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figures_dir / "EDA_temperature_vs_daily_consumption_train.png", dpi=200)
    plt.close(figure)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
