"""XGBoost ile 24 saatlik doğrudan tahmin (F12), yalnız doğrulamada."""

from datetime import date
import time

from xgboost import XGBRegressor

from src.data.load import PROJECT_ROOT
from src.data.split import chronological_split
from src.models.forecast_baselines import build_train_val_windows, ridge_design_matrix
from src.utils.results_io import record_forecast_results, resolve_path, upsert_csv
from src.utils.seed import set_seed


def xgboost_lags(config):
    """Lookback'e sığan gecikmeleri döndürür (ör. 48'de 168 ve 336 düşer)."""
    lookback = config["forecast"]["lookback"]
    return [lag for lag in config["forecast"]["xgboost"]["lags"] if lag <= lookback]


def make_xgboost(settings, seed):
    """Config ayarlarıyla (CPU, hist, early stopping val MAE) XGBRegressor kurar."""
    # Bu makinede XGBoost GPU'yu göremiyor (PyTorch görüyor); sessizce CPU'ya
    # düşüp yanlış cihaz kaydetmemek için XGBoost açıkça CPU'da çalıştırılır.
    return XGBRegressor(
        n_estimators=settings["n_estimators"],
        learning_rate=settings["learning_rate"],
        max_depth=settings["max_depth"],
        subsample=settings["subsample"],
        colsample_bytree=settings["colsample_bytree"],
        early_stopping_rounds=settings["early_stopping_rounds"],
        eval_metric="mae",
        tree_method="hist",
        device="cpu",
        random_state=seed,
    )


def run_xgboost(df, config, exp_id="F12", device_preference="cpu"):
    """XGBoost'u train'de eğitir; early stopping ve değerlendirme val'de.

    Girdi: saatlik DataFrame, config (lookback ve forecast.xgboost ayarları),
    deney kimliği; device_preference yok sayılır (her zaman CPU, aşağıya bkz.).
    Çıktı: val metrik sözlüğü; sonuç/ufuk/tahmin dosyaları ve experiments satırı.
    Neden: Ridge F3-cal ile aynı tablo kullanılır (hedef gecikmeleri + son
    gözlenen saatin B takvimi); böylece doğrusal ve ağaç tabanlı model aynı
    bilgiyi görür. Ağaçlar ölçeğe duyarsız olduğundan hedef kW'da kalır.
    Early stopping derin modellerde olduğu gibi val MAE ile yapılır.
    """
    set_seed(config["seed"])
    forecast = config["forecast"]
    settings = forecast["xgboost"]
    target = config["preprocess"]["target"]
    lookback = forecast["lookback"]
    lags = xgboost_lags(config)
    calendar = config["features"]["calendar"]

    splits = chronological_split(df, config)
    (X_train, y_train, _), (X_val, y_val, val_times) = build_train_val_windows(
        splits, target, lookback, forecast["horizon"],
        config["evaluation"]["exclude_imputed_targets"],
        features=[target] + calendar,
    )
    design_train = ridge_design_matrix(X_train, lags, use_calendar=True)
    design_val = ridge_design_matrix(X_val, lags, use_calendar=True)

    device = "cpu"
    model = make_xgboost(settings, config["seed"])
    start_time = time.perf_counter()
    # 24 çıktılı tek model; her boosting turu her ufuk için bir ağaç ekler.
    model.fit(design_train, y_train, eval_set=[(design_val, y_val)], verbose=False)
    train_time = time.perf_counter() - start_time
    predictions = model.predict(design_val)

    note = (
        f"Direct {forecast['horizon']}-hour forecast; lags={lags}; "
        f"calendar={calendar}; best_iteration={model.best_iteration}; "
        f"settings={settings}; device={device}"
    )
    scores = record_forecast_results(
        PROJECT_ROOT, config, "val", exp_id, "XGBoost",
        y_val, predictions, val_times,
        train_time_s=train_time, device=device, feature_set="B",
        lookback=lookback, note=note,
    )
    upsert_csv(
        resolve_path(PROJECT_ROOT, config["paths"]["experiments_file"]),
        [{
            "exp_id": exp_id, "date": date.today().isoformat(),
            "model": "XGBoost", "feature_set": "B", "lookback": lookback,
            "hidden": "", "layers": "", "dropout": "",
            "lr": settings["learning_rate"], "batch": "",
            # "epoch" yerine boosting turu: koşulan ve en iyi tur sayısı.
            "epochs_run": len(model.evals_result()["validation_0"]["mae"]),
            "best_epoch": int(model.best_iteration) + 1,
            "seed": config["seed"],
            "val_mae": scores["MAE"], "val_rmse": scores["RMSE"],
            "test_mae": "", "test_rmse": "", "train_time_s": train_time,
            "note": note,
        }],
        ["exp_id"],
    )
    return scores
