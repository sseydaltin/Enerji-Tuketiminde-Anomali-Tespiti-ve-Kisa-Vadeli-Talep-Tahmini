"""Final rapor tabloları: val (dev, hold) ve test sonuçlarını yan yana koyar (yeni hesap yok).

Girdi yalnız results/tables altındaki kayıtlı dosyalardır; çıktı
results/tables/final_*_val_vs_test.csv. docs/final_sonuclar.md bu tablolardan yazıldı.
"""

import numpy as np
import pandas as pd

from src.data.load import PROJECT_ROOT, load_config
from src.evaluate_forecast import GROUPS
from src.utils.metrics import calculate_metrics
from src.utils.results_io import load_predictions, resolve_path


def forecast_table(config):
    """Her grup için 24 s MAE / RMSE / R²: val ve test (3 seed'lilerde ort ± std)."""
    rows = []
    for group, exp_ids in GROUPS.items():
        row = {"group": group, "n_seeds": len(exp_ids)}
        for split in ("val", "test"):
            scores = pd.DataFrame([
                calculate_metrics(*load_predictions(PROJECT_ROOT, config, exp_id, split)[:2],
                                  config["evaluation"]["mape_min_y"])
                for exp_id in exp_ids
            ])
            for metric in ("MAE", "RMSE", "R2"):
                row[f"{split}_{metric}_mean"] = scores[metric].mean()
                row[f"{split}_{metric}_std"] = scores[metric].std(ddof=1) if len(scores) > 1 else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    for split in ("val", "test"):
        reference = table.loc[table["group"] == "F2 Seasonal Naive 168", f"{split}_MAE_mean"].iloc[0]
        table[f"{split}_vs_SN168_pct"] = 100 * (1 - table[f"{split}_MAE_mean"] / reference)
    return table


def anomaly_table(tables_dir):
    """Yöntem × (val_dev, val_hold, test) × (ana, zor): olay F1 ve nokta metrikleri."""
    val = pd.read_csv(tables_dir / "anomaly_protocol_val.csv")
    test = pd.read_csv(tables_dir / "anomaly_protocol_test.csv")
    columns = ["split", "scenario", "method", "smooth", "threshold", "n_events_mean",
               "event_f1_mean", "event_f1_std", "event_recall_mean", "event_precision_mean",
               "f1_mean", "pr_auc_mean", "roc_auc_mean"]
    return pd.concat([val[columns], test[columns]], ignore_index=True)


def main():
    """İki tabloyu yazar ve özetini yazdırır."""
    config = load_config()
    tables_dir = resolve_path(PROJECT_ROOT, config["paths"]["tables_dir"])
    forecast = forecast_table(config)
    forecast.to_csv(tables_dir / "final_forecast_val_vs_test.csv", index=False)
    anomaly = anomaly_table(tables_dir)
    anomaly.to_csv(tables_dir / "final_anomaly_val_vs_test.csv", index=False)
    pd.set_option("display.width", 250)
    print(forecast.round(4).to_string(index=False))
    print(anomaly.pivot_table(index="method", columns=["scenario", "split"],
                              values="event_f1_mean").round(3))


if __name__ == "__main__":
    main()
