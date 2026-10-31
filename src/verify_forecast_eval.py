"""Tüm tahmin modellerinin aynı val kurallarıyla değerlendirildiğini doğrular."""

import argparse
from copy import deepcopy

import numpy as np
import pandas as pd

from src.data.load import PROJECT_ROOT, load_config
from src.data.split import chronological_split
from src.data.windows import make_windows
from src.models.baselines import naive_forecasts
from src.models.forecast_baselines import build_train_val_windows
from src.train_forecast import prepare_forecast_data
from src.utils.console import setup_console
from src.utils.results_io import resolve_path


def check_hourly_index(df):
    """Verinin saatlik, sıralı ve boşluksuz olduğunu kontrol eder.

    Girdi: data/processed/hourly.parquet tablosu.
    Çıktı: satır sayısı, aralık, adım kontrolleri sözlüğü.
    Neden: Modeller "1 satır = 1 saat" varsayar; lookback=168 bir haftadır.
    """
    steps = df.index.to_series().diff().dropna()
    return {
        "rows": len(df),
        "start": df.index.min(),
        "end": df.index.max(),
        "inferred_freq": pd.infer_freq(df.index),
        "all_steps_1h": bool((steps == pd.Timedelta(hours=1)).all()),
        "monotonic": df.index.is_monotonic_increasing,
        "duplicates": int(df.index.duplicated().sum()),
    }


def pipeline_windows(df, config):
    """Her değerlendirme hattının ürettiği val pencerelerini toplar.

    Çıktı: {hat adı: (target_times [N], y_true [N, ufuk] kW)}.
    Neden: Naive, Ridge ve MLP/LSTM/GRU pencereleri farklı kodla üretilir;
    aynı hedef zamanlarını ve aynı kW değerlerini verdikleri gösterilmeli.
    """
    splits = chronological_split(df, config)
    target = config["preprocess"]["target"]
    forecast = config["forecast"]
    exclude = config["evaluation"]["exclude_imputed_targets"]

    y_naive, _, times_naive = naive_forecasts(
        df, splits["val"], target, forecast["lookback"], forecast["horizon"], exclude
    )
    _, (_, y_ridge, times_ridge) = build_train_val_windows(
        splits, target, forecast["lookback"], forecast["horizon"], exclude
    )
    windows = {
        "naive (F0-F2)": (times_naive, y_naive),
        "ridge (F3)": (times_ridge, y_ridge),
    }
    for feature_set in ("A", "B", "C"):
        set_config = deepcopy(config)
        set_config["features"]["default_set"] = feature_set
        prepared = prepare_forecast_data(df, set_config)
        windows[f"mlp/lstm/gru set {feature_set}"] = (
            prepared["val_times"], prepared["y_val"]
        )
    return windows


def compare_to_reference(name, times, y_true, ref_times, ref_y):
    """Bir pencere kümesini referansla karşılaştırır (zaman ve kW değerleri)."""
    same_times = pd.DatetimeIndex(times).equals(ref_times)
    max_diff = (
        float(np.abs(np.asarray(y_true, dtype=np.float64) - ref_y).max())
        if same_times else np.nan
    )
    return {
        "source": name,
        "windows": len(times),
        "first_target": pd.DatetimeIndex(times).min(),
        "last_target": pd.DatetimeIndex(times).max(),
        "same_target_times": same_times,
        "max_abs_diff_y_true_kw": max_diff,
        "y_true_min_kw": float(np.min(y_true)),
    }


def main():
    """Kontrolleri çalıştırır, tabloyu yazar; tutarsızlıkta hata verir."""
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    df = pd.read_parquet(resolve_path(PROJECT_ROOT, config["paths"]["processed_file"]))
    target = config["preprocess"]["target"]
    forecast = config["forecast"]

    print("=== 1) Saatlik veri kontrolü ===")
    for key, value in check_hourly_index(df).items():
        print(f"{key}: {value}")

    print("\n=== 2) is_imputed kuralı (val) ===")
    val = chronological_split(df, config)["val"]
    context = pd.concat((chronological_split(df, config)["train"].tail(forecast["lookback"]), val))
    window_args = dict(
        target=target, features=[target], lookback=forecast["lookback"],
        horizon=forecast["horizon"], target_start=val.index.min(),
        target_end=val.index.max(),
    )
    _, _, all_times = make_windows(context, exclude_imputed=False, **window_args)
    _, _, kept_times = make_windows(context, exclude_imputed=True, **window_args)
    flags = df["is_imputed"].astype(bool)
    kept_has_imputed = any(
        flags.loc[start:start + pd.Timedelta(hours=forecast["horizon"] - 1)].any()
        for start in kept_times
    )
    print(f"val'de olası pencere: {len(all_times)}, değerlendirilen: {len(kept_times)}, "
          f"çıkarılan (doldurulmuş hedefli): {len(all_times) - len(kept_times)}")
    print(f"val'deki doldurulmuş saat sayısı: {int(flags.loc[val.index].sum())}")
    print(f"değerlendirilen pencerelerde doldurulmuş hedef var mı: {kept_has_imputed}")

    print("\n=== 3) Hatlar arası pencere ve y_true karşılaştırması ===")
    windows = pipeline_windows(df, config)
    ref_times, ref_y = windows["ridge (F3)"]
    ref_y = np.asarray(ref_y, dtype=np.float64)
    rows = [
        compare_to_reference(name, times, y, ref_times, ref_y)
        for name, (times, y) in windows.items()
    ]

    print("\n=== 4) Kayıtlı tahmin dosyaları (results/predictions/*_val.npz) ===")
    predictions_dir = resolve_path(PROJECT_ROOT, config["paths"]["predictions_dir"])
    for path in sorted(predictions_dir.glob("*_val.npz")):
        data = np.load(path)
        rows.append(compare_to_reference(
            f"kayıt: {path.stem}",
            pd.DatetimeIndex(data["target_times"]),
            data["y_true"],
            ref_times,
            ref_y,
        ))

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    output = resolve_path(PROJECT_ROOT, config["paths"]["tables_dir"]) / "eval_consistency_val.csv"
    table.to_csv(output, index=False)
    print(f"\nKaydedildi: {output}")

    # float32 saklama farkı dışında y_true birebir aynı olmalı.
    consistent = table["same_target_times"].all() and (
        table["max_abs_diff_y_true_kw"] < 1e-5
    ).all()
    if not consistent or kept_has_imputed:
        raise SystemExit("TUTARSIZLIK BULUNDU")
    print("SONUÇ: tüm hatlar ve kayıtlar aynı val pencerelerini ve aynı kW hedeflerini kullanıyor.")


if __name__ == "__main__":
    main()
