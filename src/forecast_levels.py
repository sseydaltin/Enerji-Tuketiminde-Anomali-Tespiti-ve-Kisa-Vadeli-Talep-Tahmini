"""Final tahmin modellerinin VAL'de üç seviyede raporu (yeni eğitim yok; kayıtlı tahminler).

(a) h=1: her pencerenin yalnız ilk saati. (b) 24 saat ortalaması: mevcut ana
metrik (tüm ufuklar). (c) Günlük toplam: ilk hedef saati 00:00 olan pencerelerin
24 saatlik tahmin toplamı (kWh) ile gerçek günlük toplamın karşılaştırması.
Saatlik ortalama kW × 1 saat = kWh olduğundan toplam doğrudan kWh'tir.
"""

import argparse

import numpy as np
import pandas as pd

from src.data.load import PROJECT_ROOT, load_config
from src.utils.console import setup_console
from src.utils.metrics import calculate_metrics
from src.utils.results_io import resolve_path

# Rapor edilen modeller: final topluluk, üyeleri ve iki naif referans.
MODELS = {
    "F13-ens": ["F13-ens", "F13-ens-s7", "F13-ens-s2026"],
    "F10-opt": ["F10-opt", "F10-opt-s7", "F10-opt-s2026"],
    "F5-opt": ["F5-opt", "F5-opt-s7", "F5-opt-s2026"],
    "F2 (Seasonal Naive 168)": ["F2"],
    "F1 (Seasonal Naive 24)": ["F1"],
}


def level_metrics(y_true, y_pred, times, mape_min_y):
    """Bir tahmin dosyası için üç seviyenin metrikleri.

    Girdi: y_true, y_pred [N, 24] (kW), ilk hedef saatleri [N], MAPE alt sınırı (kW).
    Çıktı: {seviye: metrik sözlüğü} — (c)'de birim kWh, gün sayısı n.
    Neden: Günlük toplamda yalnız gece yarısı başlayan pencereler alınır; her
    takvim günü bir kez sayılır ve tahmin bir önceki günün 23:00'üne kadarki
    veriyle yapılmıştır. Doldurulmuş hedefli pencereler kayıtlı dosyada zaten yoktur.
    """
    midnight = pd.DatetimeIndex(times).hour == 0
    daily_true, daily_pred = y_true[midnight].sum(axis=1), y_pred[midnight].sum(axis=1)
    return {
        "a_h1": {**calculate_metrics(y_true[:, 0], y_pred[:, 0], mape_min_y), "n": len(y_true)},
        "b_24h": {**calculate_metrics(y_true, y_pred, mape_min_y), "n": len(y_true)},
        "c_daily": {**calculate_metrics(daily_true, daily_pred, mape_min_y), "n": int(midnight.sum())},
    }


def main():
    """Tabloyu results/tables/forecast_levels_val.csv/.md olarak yazar."""
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    predictions_dir = resolve_path(PROJECT_ROOT, config["paths"]["predictions_dir"])
    mape_min_y = config["evaluation"]["mape_min_y"]

    rows = []
    for label, exp_ids in MODELS.items():
        for exp_id in exp_ids:
            data = np.load(predictions_dir / f"{exp_id}_val.npz")
            for level, metrics in level_metrics(data["y_true"], data["y_pred"],
                                                data["target_times"], mape_min_y).items():
                rows.append({"model": label, "exp_id": exp_id, "level": level, **metrics})
    per_run = pd.DataFrame(rows)
    per_run.to_csv(resolve_path(PROJECT_ROOT, "results/tables/forecast_levels_val_per_seed.csv"),
                   index=False)
    metrics = ["MAE", "RMSE", "MAPE*", "sMAPE", "R2"]
    summary = per_run.groupby(["model", "level"], sort=False).agg(
        n_seeds=("exp_id", "count"), n=("n", "first"),
        **{f"{m}_mean": (m, "mean") for m in metrics},
        **{f"{m}_std": (m, "std") for m in metrics},
    ).reset_index()
    tables_dir = resolve_path(PROJECT_ROOT, config["paths"]["tables_dir"])
    summary.to_csv(tables_dir / "forecast_levels_val.csv", index=False)
    # Markdown tablosu elle yazılır (tabulate bağımlılığı eklenmez).
    table = summary.round(4).astype(str)
    lines = ["| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
    lines += ["| " + " | ".join(row) + " |" for row in table.to_numpy()]
    (tables_dir / "forecast_levels_val.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
