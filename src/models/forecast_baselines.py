"""Ridge tahmin baseline'ını (F3) yalnızca doğrulamada çalıştırır."""

import argparse
from datetime import date
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.data.load import PROJECT_ROOT, load_config
from src.data.split import chronological_split
from src.data.windows import make_windows
from src.utils.console import setup_console
from src.utils.results_io import (
    record_forecast_results,
    resolve_path,
    upsert_csv,
)
from src.utils.seed import set_seed


def build_train_val_windows(
    splits, target, lookback, horizon, exclude_imputed, features=None, eval_split="val",
):
    """Train ve val için sınırları aşmayan tahmin pencereleri hazırlar.

    Girdi: kronolojik split tabloları; saatlik [N, F] DataFrame'ler;
    features verilmezse yalnız hedef (ilk sütun her zaman hedef olmalı).
    Çıktı: train ve val için (X: [örnek, lookback, len(features)],
    y: [örnek, horizon], hedef zamanları) üçlüleri.
    Neden: Eğitim hedefleri train'de kalır; val geçmişi train sonundan
    devam edebilir ancak val hedefleri train/val sınırını aşmaz.
    """
    features = features or [target]
    if features[0] != target:
        raise ValueError("İlk öznitelik hedef olmalı (gecikmeler ondan alınır).")
    train_frame = splits["train"]
    # eval_split="test" yalnız final değerlendirmede: geçmiş train + val'den gelir.
    history = train_frame if eval_split == "val" else pd.concat((train_frame, splits["val"]))
    val_frame = pd.concat((history, splits[eval_split]))

    train_windows = make_windows(
        train_frame,
        target,
        features,
        lookback,
        horizon,
        exclude_imputed=False,
    )
    val_windows = make_windows(
        val_frame,
        target,
        features,
        lookback,
        horizon,
        target_start=splits[eval_split].index.min(),
        target_end=splits[eval_split].index.max(),
        exclude_imputed=exclude_imputed,
    )

    if len(train_windows[0]) == 0:
        raise ValueError("Train kümesinde geçerli pencere bulunamadı.")
    if len(val_windows[0]) == 0:
        raise ValueError("Doldurulmuş hedefler çıkarıldıktan sonra val boş.")

    return train_windows, val_windows


def select_ridge_lags(windows, lags):
    """168 saatlik geçmişten config'te seçilen gecikmeleri çıkarır.

    Girdi: windows [örnek, lookback, 1], lags: saat cinsinden gecikmeler.
    Çıktı: [örnek, len(lags)] lag özellik matrisi.
    Neden: En yeni değer lag 1, en eski değer lag lookback olacak biçimde
    README'deki F3 gecikme listesini uygular.
    """
    lookback = windows.shape[1]
    if not lags or any(lag < 1 or lag > lookback for lag in lags):
        raise ValueError("Ridge gecikmeleri 1 ile lookback arasında olmalı.")

    lag_indices = [lookback - lag for lag in lags]
    return windows[:, lag_indices, 0]


def fit_ridge(X_train, y_train, X_val, train_target, alpha):
    """Ridge modelini train'de fit eder, val tahminini özgün kW ölçeğine çevirir.

    Girdi: X dizileri lag değerleri, y dizileri [örnek, ufuk], train_target
    yalnız train tüketim serisi, alpha config'ten gelir.
    Çıktı: val tahminleri [örnek, ufuk], scaler'lar, fit süresi.
    Neden: Özellik ve hedef ölçekleri yalnız train tüketiminden öğrenilir;
    val hiçbir aşamada fit işlemine katılmaz.
    """
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(
        np.asarray(train_target, dtype=np.float64).reshape(-1, 1)
    )

    X_train_scaled = x_scaler.transform(X_train)
    X_val_scaled = x_scaler.transform(X_val)
    y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1)).reshape(
        y_train.shape
    )

    model = Ridge(alpha=alpha)
    start_time = time.perf_counter()
    model.fit(X_train_scaled, y_train_scaled)
    elapsed = time.perf_counter() - start_time

    predictions_scaled = model.predict(X_val_scaled)
    predictions = y_scaler.inverse_transform(
        predictions_scaled.reshape(-1, 1)
    ).reshape(predictions_scaled.shape)
    return predictions, x_scaler, y_scaler, elapsed


def ridge_design_matrix(windows, lags, use_calendar):
    """Ridge girdisini oluşturur: hedef gecikmeleri (+ son girdi saatinin takvimi).

    Girdi: windows [örnek, lookback, 1 + takvim]; ilk sütun hedef.
    Çıktı: [örnek, len(lags)] veya [örnek, len(lags) + takvim].
    Neden: Takvim, son gözlenen saatten (t−1) alınır; LSTM/GRU'nun B setinde
    gördüğü bilgiyle aynıdır (hedef saatlerin takvimi bundan türetilebilir).
    """
    lag_values = select_ridge_lags(windows[:, :, :1], lags)
    if not use_calendar:
        return lag_values
    return np.concatenate((lag_values, windows[:, -1, 1:]), axis=1)


def run_ridge(df, config, alpha=None, exp_id="F3", use_calendar=False):
    """F3 Ridge'i train'de fit eder, yalnız val'de değerlendirip kaydeder.

    Girdi: saatlik [zaman, özellik] DataFrame, config, isteğe bağlı alpha
    (verilmezse config'teki değer), deney kimliği ve takvim bayrağı
    (True: B setindeki takvim öznitelikleri eklenir, F3-cal).
    Çıktı: val metrik sözlüğü. Sonuç, ufuk ve tahmin dosyaları ile
    experiments.csv satırı record_forecast_results/upsert_csv ile yazılır.
    Neden: Alpha ızgarası her değeri ayrı exp_id ile kaydeder; seçim val MAE
    ile yapılır. Test kümesi pencere üretiminde de kullanılmaz.
    """
    set_seed(config["seed"])
    splits = chronological_split(df, config)
    forecast = config["forecast"]
    target = config["preprocess"]["target"]
    lookback = forecast["lookback"]
    alpha = forecast["ridge"]["alpha"] if alpha is None else alpha
    lags = forecast["ridge"]["lags"]
    calendar = config["features"]["calendar"] if use_calendar else []

    (X_train, y_train, _), (X_val, y_val, val_times) = build_train_val_windows(
        splits,
        target,
        lookback,
        forecast["horizon"],
        config["evaluation"]["exclude_imputed_targets"],
        features=[target] + calendar,
    )
    predictions, _, _, train_time = fit_ridge(
        ridge_design_matrix(X_train, lags, use_calendar),
        y_train,
        ridge_design_matrix(X_val, lags, use_calendar),
        splits["train"][target].to_numpy(dtype=np.float64),
        alpha,
    )

    feature_set = "B" if use_calendar else "A"
    note = (
        f"Direct {forecast['horizon']}-hour forecast; lags={lags}; "
        f"calendar={calendar}; alpha={alpha}"
    )
    scores = record_forecast_results(
        PROJECT_ROOT, config, "val", exp_id, "Ridge",
        y_val, predictions, val_times,
        train_time_s=train_time, device="cpu", feature_set=feature_set,
        lookback=lookback, note=note,
    )
    upsert_csv(
        resolve_path(PROJECT_ROOT, config["paths"]["experiments_file"]),
        [{
            "exp_id": exp_id, "date": date.today().isoformat(),
            "model": "Ridge", "feature_set": feature_set, "lookback": lookback,
            "seed": config["seed"],
            "val_mae": scores["MAE"], "val_rmse": scores["RMSE"],
            "test_mae": "", "test_rmse": "", "train_time_s": train_time,
            "note": note + "; device=cpu",
        }],
        ["exp_id"],
    )
    return scores


def main():
    """Ridge'i tek alpha ile çalıştırır (ızgara için --alpha/--exp verilir).

    Girdi: --config, --alpha, --exp. Çıktı: val metrikleri ve sonuç dosyaları.
    Neden: CLI'da split seçeneği yoktur; bu deneyde test kümesi kullanılamaz.
    """
    setup_console()  # Windows konsolunda Türkçe karakterler için UTF-8
    parser = argparse.ArgumentParser(
        description="Ridge baseline'ını yalnızca val'de değerlendir."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--exp", default="F3")
    args = parser.parse_args()

    config = load_config(args.config)
    hourly = pd.read_parquet(
        resolve_path(PROJECT_ROOT, config["paths"]["processed_file"])
    )
    scores = run_ridge(hourly, config, alpha=args.alpha, exp_id=args.exp)
    print(f"\n=== {args.exp} RIDGE VAL ===")
    print(pd.Series(scores).to_string())


if __name__ == "__main__":
    main()
