"""Ridge ve MLP tahmin baseline'larını yalnızca doğrulamada çalıştırır."""

import argparse
from datetime import date
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

from src.data.load import PROJECT_ROOT, load_config
from src.data.split import chronological_split
from src.data.windows import make_windows
from src.utils.metrics import calculate_metrics, metrics_by_horizon
from src.utils.seed import set_seed


def build_train_val_windows(splits, target, lookback, horizon, exclude_imputed):
    """Train ve val için sınırları aşmayan tahmin pencereleri hazırlar.

    Girdi: kronolojik split tabloları; saatlik [N, F] DataFrame'ler.
    Çıktı: train ve val için (X: [örnek, lookback, 1],
    y: [örnek, horizon], hedef zamanları) üçlüleri.
    Neden: Eğitim hedefleri train'de kalır; val geçmişi train sonundan
    devam edebilir ancak val hedefleri train/val sınırını aşmaz.
    """
    train_frame = splits["train"]
    val_frame = pd.concat((train_frame, splits["val"]))

    train_windows = make_windows(
        train_frame,
        target,
        [target],
        lookback,
        horizon,
        exclude_imputed=False,
    )
    val_windows = make_windows(
        val_frame,
        target,
        [target],
        lookback,
        horizon,
        target_start=splits["val"].index.min(),
        target_end=splits["val"].index.max(),
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


def fit_mlp(
    X_train,
    y_train,
    X_val,
    y_val,
    train_target,
    forecast_config,
    seed,
):
    """MLPRegressor'ı train'de fit eder, val tahminleri ve eğitim kaybını döndürür.

    Girdi: X_train/X_val [örnek, lookback, 1], y_train/y_val [örnek, ufuk].
    Çıktı: val tahminleri [örnek, ufuk], scaler'lar, epoch sayısı, süre ve
    convergence warning bayrağı.
    Neden: sklearn MLPRegressor çok çıktılı regresyonu doğrudan destekler.
    Ölçekleyiciler train'e fit edilir; dahili rastgele validation split kapalıdır.
    """
    set_seed(seed)
    lookback = X_train.shape[1]
    x_scaler = StandardScaler().fit(X_train.reshape(-1, 1))
    y_scaler = StandardScaler().fit(
        np.asarray(train_target, dtype=np.float64).reshape(-1, 1)
    )
    X_train_scaled = x_scaler.transform(
        X_train.reshape(-1, 1)
    ).reshape(X_train.shape[0], lookback)
    X_val_scaled = x_scaler.transform(
        X_val.reshape(-1, 1)
    ).reshape(X_val.shape[0], lookback)
    y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1)).reshape(
        y_train.shape
    )
    max_epochs = forecast_config["max_epochs"]
    model = MLPRegressor(
        hidden_layer_sizes=tuple(forecast_config["mlp"]["hidden_sizes"]),
        activation="relu",
        solver="adam",
        alpha=forecast_config["weight_decay"],
        batch_size=forecast_config["batch_size"],
        learning_rate_init=forecast_config["lr"],
        max_iter=max_epochs,
        random_state=seed,
        shuffle=False,
        early_stopping=False,
        tol=0.0,
        n_iter_no_change=max_epochs + 1,
    )
    start_time = time.perf_counter()
    # max_iter, config'teki epoch üst sınırıdır; iç doğrulama bölmesi kapalıdır.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X_train_scaled, y_train_scaled)
    convergence_warning = False
    for warning in caught:
        if issubclass(warning.category, ConvergenceWarning):
            convergence_warning = True
            print(f"MLPRegressor uyarısı: {warning.message}")
        else:
            warnings.warn(warning.message, warning.category)

    predictions_scaled = model.predict(X_val_scaled)
    val_predictions = y_scaler.inverse_transform(
        predictions_scaled.reshape(-1, 1)
    ).reshape(y_val.shape)
    elapsed = time.perf_counter() - start_time
    train_loss = model.loss_curve_[-1]
    return (
        val_predictions,
        x_scaler,
        y_scaler,
        model,
        model.n_iter_,
        train_loss,
        elapsed,
        convergence_warning,
    )


def _append_rows(path, new_rows, key_columns):
    """CSV'ye yeni sonuçları ekler; aynı anahtardaki eski koşuyu yeniler.

    Girdi: CSV yolu, sözlük satırları, tekil koşuyu belirleyen sütunlar.
    Çıktı: Güncellenmiş CSV dosyası.
    Neden: Aynı deney tekrarlandığında eski F3/F4 satırları çoğalmaz ve
    diğer modellerin mevcut sonuçları korunur.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    updated = pd.DataFrame(new_rows)

    if output_path.exists():
        previous = pd.read_csv(output_path)
        for column in updated.columns:
            if column not in previous.columns:
                previous[column] = np.nan
        previous = previous.loc[
            ~previous[key_columns].apply(tuple, axis=1).isin(
                updated[key_columns].apply(tuple, axis=1)
            )
        ]
        updated = pd.concat((previous, updated), ignore_index=True)

    updated.to_csv(output_path, index=False)


def run_forecast_baselines(df, config, model_names=("Ridge", "MLP")):
    """F3 Ridge ve F4 MLP'yi train'de eğitir, yalnız val'de değerlendirir.

    Girdi: saatlik [zaman, özellik] DataFrame, config ve seçilen F3/F4 modelleri.
    Çıktı: val özetleri; ayrıca baseline, ufuk ve experiments CSV dosyalarını
    günceller. Test verisi model kurulumuna veya değerlendirmeye girmez.
    Neden: F3/F4 sonuçları mevcut naive sonuçlarla aynı doğrulama tablosunda
    karşılaştırılabilir ve deney şemasına uygun kaydedilir.
    """
    set_seed(config["seed"])
    splits = chronological_split(df, config)
    forecast = config["forecast"]
    target = config["preprocess"]["target"]
    lookback = forecast["lookback"]
    horizon = forecast["horizon"]
    exclude_imputed = config["evaluation"]["exclude_imputed_targets"]

    train_windows, val_windows = build_train_val_windows(
        splits,
        target,
        lookback,
        horizon,
        exclude_imputed,
    )
    X_train, y_train, _ = train_windows
    X_val, y_val, _ = val_windows
    train_target = splits["train"][target].to_numpy(dtype=np.float64)

    allowed_models = {"Ridge", "MLP"}
    if not model_names or not set(model_names).issubset(allowed_models):
        raise ValueError(f"model_names yalnızca {sorted(allowed_models)} olabilir.")
    model_results = []
    ridge_config = forecast["ridge"]
    lags = ridge_config["lags"]
    if "Ridge" in model_names:
        ridge_train_X = select_ridge_lags(X_train, lags)
        ridge_val_X = select_ridge_lags(X_val, lags)
        ridge_predictions, _, _, ridge_time = fit_ridge(
            ridge_train_X,
            y_train,
            ridge_val_X,
            train_target,
            ridge_config["alpha"],
        )
        model_results.append(("Ridge", ridge_predictions, ridge_time, None, None))

    if "MLP" in model_names:
        (
            mlp_predictions,
            _,
            _,
            _,
            mlp_epochs,
            mlp_train_loss,
            mlp_time,
            mlp_convergence_warning,
        ) = fit_mlp(
            X_train,
            y_train,
            X_val,
            y_val,
            train_target,
            forecast,
            config["seed"],
        )
        model_results.append(
            (
                "MLP",
                mlp_predictions,
                mlp_time,
                mlp_train_loss,
                mlp_convergence_warning,
            )
        )

    summary_rows = []
    horizon_rows = []
    experiment_rows = []
    for (
        model_name,
        predictions,
        train_time,
        train_loss,
        convergence_warning,
    ) in model_results:
        scores = calculate_metrics(
            y_val,
            predictions,
            mape_min_y=config["evaluation"]["mape_min_y"],
        )
        summary_rows.append({
            "split": "val",
            "model": model_name,
            "windows": len(y_val),
            "MAE": scores["MAE"],
            "RMSE": scores["RMSE"],
            "MAPE*": scores["MAPE*"],
            "sMAPE": scores["sMAPE"],
            "R2": scores["R2"],
        })
        horizon_table = metrics_by_horizon(
            y_val,
            predictions,
            mape_min_y=config["evaluation"]["mape_min_y"],
        )
        horizon_table.insert(0, "model", model_name)
        horizon_table.insert(0, "split", "val")
        horizon_table["windows"] = len(y_val)
        horizon_rows.extend(horizon_table.to_dict("records"))

        is_mlp = model_name == "MLP"
        experiment_rows.append({
            "exp_id": "F4" if is_mlp else "F3",
            "date": date.today().isoformat(),
            "model": model_name,
            "feature_set": "A",
            "lookback": lookback,
            "hidden": "x".join(
                str(size) for size in forecast["mlp"]["hidden_sizes"]
            ) if is_mlp else "",
            "layers": len(forecast["mlp"]["hidden_sizes"]) if is_mlp else "",
            "dropout": "",
            "lr": forecast["lr"] if is_mlp else "",
            "batch": forecast["batch_size"] if is_mlp else "",
            "epochs_run": mlp_epochs if is_mlp else "",
            "best_epoch": "",
            "seed": config["seed"],
            "val_mae": scores["MAE"],
            "val_rmse": scores["RMSE"],
            "test_mae": "",
            "test_rmse": "",
            "train_time_s": train_time,
            "note": (
                f"Direct {horizon}-hour forecast; "
                f"lags={lags}, alpha={ridge_config['alpha']}"
                if not is_mlp
                else f"Direct {horizon}-hour forecast; "
                f"hidden_sizes={forecast['mlp']['hidden_sizes']}; "
                f"train_loss={train_loss:.6f}; "
                f"iterations={mlp_epochs}; "
                f"convergence_warning={convergence_warning}"
            ),
        })

    tables_dir = Path(config["paths"]["tables_dir"])
    if not tables_dir.is_absolute():
        tables_dir = PROJECT_ROOT / tables_dir
    experiments_file = Path(config["paths"]["experiments_file"])
    if not experiments_file.is_absolute():
        experiments_file = PROJECT_ROOT / experiments_file

    _append_rows(
        tables_dir / "baseline_results.csv",
        summary_rows,
        ["split", "model"],
    )
    _append_rows(
        tables_dir / "forecast_horizon_metrics.csv",
        horizon_rows,
        ["split", "model", "horizon"],
    )
    _append_rows(
        experiments_file,
        experiment_rows,
        ["exp_id"],
    )

    return pd.DataFrame(summary_rows)


def main():
    """Train saatlik veriyi okur; Ridge ve MLP için yalnız val sonuçlarını yazar.

    Girdi: isteğe bağlı --config YAML yolu.
    Çıktı: val MAE/RMSE tablosu ve proje sonuç CSV dosyaları.
    Neden: CLI'da split seçeneği sunulmaz; bu deneyde test kümesi kullanılamaz.
    """
    parser = argparse.ArgumentParser(
        description="Ridge ve MLP baseline'larını yalnızca val'de değerlendir."
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    data_path = Path(config["paths"]["processed_file"])
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    hourly = pd.read_parquet(data_path)

    results = run_forecast_baselines(hourly, config)
    print("\n=== RIDGE + MLP VAL SONUÇLARI ===")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
