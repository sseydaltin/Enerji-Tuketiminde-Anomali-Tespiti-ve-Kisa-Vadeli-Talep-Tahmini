"""Tek tip enjeksiyonda yöntemlerin yakalama oranı tablosu (demo denetimi; final sonuç değildir).

Çalıştırma: python -m app.capture_table
Çıktı: results/tables/demo_capture_by_type.csv ve .md

Final tablolar dört tipi karışık enjekte eder; burada her koşu tek tip içerir ve test
seed'leri (config) kullanılır. Model, eşik veya yöntem seçimine bağlanmaz.
"""

import numpy as np
import pandas as pd

from app.anomaly_services import (
    CAUGHT,
    METHODS,
    analyze_test_range,
    anomaly_demo_paths,
    checkpoint_signature,
    complete_test_days,
    event_outcomes,
    load_anomaly_calibration,
)
from app.services import load_f13_models, load_household_data
from app.ui import ANOMALY_LABELS, METHOD_LABELS
from src.utils.console import setup_console

OUTPUT_NAME = "demo_capture_by_type"


def build_table():
    """Her tip × yöntem için 5 seed'in yakalama oranını hesaplar (varsayılan şiddet ×1).

    Çıktı: (seed başına oranlar, ortalama ± std ve ortalama yanlış alarm içeren
    DataFrame; tablo klasörü).
    Neden: Sayfadaki tek koşuluk "yakalanan/toplam" değerinin seed'e ne kadar bağlı
    olduğunu gösterir.
    """
    config, data_path, models_dir, tables_dir, features = anomaly_demo_paths()
    frame = load_household_data(str(data_path))
    calibration = load_anomaly_calibration(str(tables_dir))
    models = load_f13_models(str(models_dir), features)
    days = complete_test_days(frame, config)
    start = pd.Timestamp(days[0])
    end = pd.Timestamp(days[-1]) + pd.Timedelta(hours=23)
    seeds = config["protocol"]["scenarios"]["main"]["seeds"]["test"]
    signature = checkpoint_signature(models_dir)

    rows = []
    for kind, kind_label in ANOMALY_LABELS.items():
        per_seed = {method: [] for method in METHODS}
        for seed in seeds:
            result = analyze_test_range(
                frame, config, start, end, True, kind, "main", seed, calibration, signature, models,
            )
            for method in METHODS:
                table, false_alarms = event_outcomes(result, method)
                per_seed[method].append(((table["Durum"] == CAUGHT).sum(), len(table), false_alarms))
        for method in METHODS:
            caught, total, false_alarms = np.array(per_seed[method], dtype=float).T
            rates = caught / total
            rows.append({
                "Tip": kind_label,
                "Yöntem": METHOD_LABELS[method],
                "Seed başına yakalanan / toplam": ", ".join(
                    f"{int(c)}/{int(t)}" for c, t in zip(caught, total)
                ),
                "Yakalama oranı (ort)": rates.mean(),
                "Yakalama oranı (std)": rates.std(ddof=1),
                "Yanlış alarm (ort)": false_alarms.mean(),
            })
    return pd.DataFrame(rows), tables_dir


def _markdown(table):
    """Tabloyu ek kütüphane gerektirmeden Markdown metnine çevirir."""
    rounded = table.round(3).astype(str)
    lines = ["| " + " | ".join(rounded.columns) + " |", "|" + "---|" * len(rounded.columns)]
    lines += ["| " + " | ".join(row) + " |" for row in rounded.to_numpy()]
    return "\n".join(lines) + "\n"


def main():
    """Tabloyu üretir, results/tables altına yazar ve ekrana basar."""
    setup_console()
    table, tables_dir = build_table()
    table.to_csv(tables_dir / f"{OUTPUT_NAME}.csv", index=False)
    (tables_dir / f"{OUTPUT_NAME}.md").write_text(_markdown(table), encoding="utf-8")
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
