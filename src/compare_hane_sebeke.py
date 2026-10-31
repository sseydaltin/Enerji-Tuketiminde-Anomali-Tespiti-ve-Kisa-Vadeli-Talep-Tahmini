"""Hane (household) ve şebeke (ELD) TEST sonuçlarını yan yana koyar (yeni hesap yok).

Ölçek farkı (~1 kW ve ~106 MW) yüzünden MAE karşılaştırılmaz; MAPE*, sMAPE,
R² ve aynı dönemdeki SN168'e göre MAE kazancı kullanılır. Anomalide olay F1
ve PR-AUC (ana senaryo, eşik/pencere her veri setinin kendi val'inde seçildi).
Çıktı: results/eld/tables/hane_vs_sebeke_{forecast,anomaly}.csv
"""

import pandas as pd

from src.data.load import PROJECT_ROOT
from src.utils.results_io import resolve_path

GROUPS = ["F1 Seasonal Naive 24", "F2 Seasonal Naive 168", "F3-cal Ridge", "F12-L336 XGBoost",
          "F5-opt LSTM", "F10-opt Seq2Seq", "F13-ens (final)"]
METHODS = ["A0", "A3-s2s-norm", "A5-mean-z"]


def forecast_table(household, eld):
    """Grup × seviye: iki veri setinin MAPE*, sMAPE, R² ve SN168 kazancı (MAE oranından)."""
    rows = []
    for level in ("a_h1", "b_24h", "c_daily"):
        for group in GROUPS:
            row = {"level": level, "group": group}
            for name, table in (("hane", household), ("sebeke", eld)):
                part = table[table["level"] == level].set_index("group")
                reference = part.loc["F2 Seasonal Naive 168", "MAE_mean"]
                row[f"{name}_MAPE"] = part.loc[group, "MAPE*_mean"]
                row[f"{name}_sMAPE"] = part.loc[group, "sMAPE_mean"]
                row[f"{name}_R2"] = part.loc[group, "R2_mean"]
                row[f"{name}_vs_SN168_pct"] = 100 * (1 - part.loc[group, "MAE_mean"] / reference)
                row[f"{name}_n_seeds"] = part.loc[group, "n_seeds"]
            rows.append(row)
    return pd.DataFrame(rows)


def anomaly_table(household, eld):
    """Yöntem: iki veri setinin olay F1 ve PR-AUC'si (ort ± std), ana senaryo, test."""
    household = household[household["scenario"] == "main"].set_index("method")
    eld = eld.set_index("method")
    rows = []
    for method in METHODS:
        row = {"method": method}
        for name, table in (("hane", household), ("sebeke", eld)):
            for metric in ("event_f1", "pr_auc", "event_recall", "event_precision"):
                row[f"{name}_{metric}_mean"] = table.loc[method, f"{metric}_mean"]
                row[f"{name}_{metric}_std"] = table.loc[method, f"{metric}_std"]
            row[f"{name}_smooth"] = int(table.loc[method, "smooth"])
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    """İki tabloyu yazar ve yazdırır."""
    household_dir = resolve_path(PROJECT_ROOT, "results/tables")
    eld_dir = resolve_path(PROJECT_ROOT, "results/eld/tables")
    forecast = forecast_table(pd.read_csv(household_dir / "forecast_levels_test.csv"),
                              pd.read_csv(eld_dir / "forecast_levels_test.csv"))
    anomaly = anomaly_table(pd.read_csv(household_dir / "anomaly_protocol_test.csv"),
                            pd.read_csv(eld_dir / "anomaly_test.csv"))
    forecast.to_csv(eld_dir / "hane_vs_sebeke_forecast.csv", index=False)
    anomaly.to_csv(eld_dir / "hane_vs_sebeke_anomaly.csv", index=False)
    pd.set_option("display.width", 250)
    print(forecast.round(3).to_string(index=False))
    print(anomaly.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
