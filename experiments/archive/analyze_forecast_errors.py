"""Kayıtlı val tahminleriyle hata analizi: günün saati, tüketim düzeyi, ufuk."""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.load import PROJECT_ROOT, load_config
from src.utils.console import setup_console
from src.utils.metrics import calculate_metrics
from src.utils.results_io import load_predictions, resolve_path

# Tüketim aralıkları (kW): [alt, üst) ; son aralığın üstü sınırsız.
LEVEL_BINS = [0.0, 0.5, 1.0, 2.0, np.inf]
LEVEL_LABELS = ["<0.5", "0.5-1", "1-2", ">2"]


def target_hours(target_times, horizon):
    """Her [pencere, ufuk] hücresinin günün hangi saatine düştüğünü verir.

    Girdi: target_times [N] (ilk hedef zamanı), horizon (24).
    Çıktı: [N, horizon] tamsayı dizi (0–23).
    Neden: Ufuk adımı h olan hedefin saati = ilk hedef saati + h (mod 24).
    """
    start_hours = np.asarray(pd.DatetimeIndex(target_times).hour)
    return (start_hours[:, None] + np.arange(horizon)[None, :]) % 24


def grouped_metrics(y_true, y_pred, groups, labels, mape_min_y):
    """Hücreleri gruplara ayırıp her grup için MAE, sMAPE, örnek sayısı hesaplar.

    Girdi: y_true/y_pred [N, ufuk] kW, groups [N, ufuk] grup etiketi.
    Çıktı: grup başına satırlı DataFrame.
    """
    rows = []
    for label in labels:
        mask = groups == label
        if not mask.any():
            continue
        scores = calculate_metrics(y_true[mask], y_pred[mask], mape_min_y)
        rows.append({
            "group": label,
            "n_points": int(mask.sum()),
            "MAE": scores["MAE"],
            "sMAPE": scores["sMAPE"],
            "mean_actual_kw": float(y_true[mask].mean()),
        })
    return pd.DataFrame(rows)


def main():
    """Seçilen exp_id'ler için tablo ve figürleri üretir (yalnız val)."""
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", nargs="+", required=True,
                        help="Saat/düzey analizi için exp_id'ler")
    parser.add_argument("--horizon_exp", nargs="+", default=None,
                        help="Ufuk grafiği için exp_id'ler (varsayılan: --exp)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    mape_min_y = config["evaluation"]["mape_min_y"]
    tables_dir = resolve_path(PROJECT_ROOT, config["paths"]["tables_dir"])
    figures_dir = resolve_path(PROJECT_ROOT, config["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(tables_dir / "forecast_val_results.csv").set_index("exp_id")
    name_tag = "_".join(args.exp)

    hourly_tables, level_tables = [], []
    for exp_id in args.exp:
        y_true, y_pred, times = load_predictions(PROJECT_ROOT, config, exp_id)
        label = f"{results.loc[exp_id, 'model']} ({exp_id})"

        hours = target_hours(times, y_true.shape[1])
        hourly = grouped_metrics(y_true, y_pred, hours, range(24), mape_min_y)
        hourly.insert(0, "model", label)
        hourly_tables.append(hourly)

        # Düzey, gerçek tüketime (y_true) göre belirlenir.
        levels = np.array(LEVEL_LABELS, dtype=object)[
            np.digitize(y_true, LEVEL_BINS[1:-1])
        ]
        by_level = grouped_metrics(y_true, y_pred, levels, LEVEL_LABELS, mape_min_y)
        by_level.insert(0, "model", label)
        level_tables.append(by_level)

    hourly = pd.concat(hourly_tables, ignore_index=True).rename(columns={"group": "hour"})
    by_level = pd.concat(level_tables, ignore_index=True).rename(columns={"group": "level_kw"})
    hourly.to_csv(tables_dir / f"{name_tag}_val_hourly_errors.csv", index=False)
    by_level.to_csv(tables_dir / f"{name_tag}_val_error_by_level.csv", index=False)

    # --- Günün saatine göre MAE ve sMAPE ---
    figure, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for label, table in hourly.groupby("model", sort=False):
        axes[0].plot(table["hour"], table["MAE"], marker="o", label=label)
        axes[1].plot(table["hour"], table["sMAPE"], marker="o", label=label)
    first = hourly[hourly["model"] == hourly["model"].iloc[0]]
    axes[2].bar(first["hour"], first["mean_actual_kw"], color="gray")
    axes[0].set_ylabel("MAE (kW)")
    axes[1].set_ylabel("sMAPE (%)")
    axes[2].set_ylabel("Ortalama gerçek tüketim (kW)")
    axes[2].set_xlabel("Günün saati (hedef saat, 0–23)")
    axes[2].set_xticks(range(24))
    axes[0].set_title("Saat bazında hatalar (val kümesi)")
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(figures_dir / f"{name_tag}_val_hourly_errors.png", dpi=200)
    plt.close(figure)

    # --- Ufka göre MAE (h = 1..24) ---
    horizon_ids = args.horizon_exp or args.exp
    horizon_table = pd.read_csv(tables_dir / "forecast_val_horizon.csv")
    figure, axis = plt.subplots(figsize=(9, 5))
    for exp_id in horizon_ids:
        rows = horizon_table[horizon_table["exp_id"] == exp_id]
        axis.plot(rows["horizon"], rows["MAE"], marker="o",
                  label=f"{results.loc[exp_id, 'model']} ({exp_id})")
    axis.set_xlabel("Tahmin ufku h (saat)")
    axis.set_ylabel("MAE (kW)")
    axis.set_xticks(range(1, 25))
    axis.set_title("Ufka göre MAE (val kümesi)")
    axis.legend()
    figure.tight_layout()
    horizon_tag = "_".join(horizon_ids)
    figure.savefig(figures_dir / f"{horizon_tag}_val_horizon_mae.png", dpi=200)
    plt.close(figure)

    print("=== Tüketim düzeyine göre (val) ===")
    print(by_level.round(4).to_string(index=False))
    print("\n=== Saat bazında MAE (val) ===")
    print(hourly.pivot(index="hour", columns="model", values="MAE").round(3).to_string())


if __name__ == "__main__":
    main()
