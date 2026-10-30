"""LSTM ve GRU tahmin modellerinin eğitim/doğrulama akışı."""

import argparse
from copy import deepcopy
from datetime import date
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.data.load import PROJECT_ROOT, load_config
from src.data.split import chronological_split
from src.data.windows import get_feature_columns, make_windows
from src.models.forecast_baselines import run_forecast_baselines
from src.models.gru import GRUForecaster
from src.models.lstm import LSTMForecaster
from src.utils.metrics import calculate_metrics, metrics_by_horizon
from src.utils.seed import set_seed


MODEL_CLASSES = {
    "lstm": LSTMForecaster,
    "gru": GRUForecaster,
}
EXPERIMENT_IDS = {"lstm": "F5", "gru": "F6"}


def _forecast_device(preference="auto"):
    """Tahmin eğitimi için istenen veya uygun cihazı seçer.

    Girdi: auto, cuda, mps veya cpu tercihi.
    Çıktı: PyTorch device.
    Neden: Otomatik seçim donanıma uyum sağlar; bellek kısıtında CPU açıkça
    seçilebilir ve mevcut olmayan aygıt tercihi hata olarak bildirilir.
    """
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA aygıtı bu ortamda kullanılamıyor.")
        return torch.device("cuda")
    if preference == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS aygıtı bu ortamda kullanılamıyor.")
        return torch.device("mps")
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError("device tercihi auto, cuda, mps veya cpu olmalı.")


def _resolve_path(root, configured_path):
    """Yapılandırılmış yolu mutlak proje yolu olarak döndürür.

    Girdi: proje kökü ve config'teki mutlak veya göreli yol.
    Çıktı: Path nesnesi.
    Neden: Komut farklı çalışma dizinlerinden çağrılsa da aynı sonuç konumunu
    kullanır.
    """
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def prepare_forecast_data(df, config):
    """Train/val pencerelerini train-fit ölçekleyicilerle hazırlar.

    Girdi: saatlik [zaman, özellik] DataFrame ve proje config'i.
    Çıktı: train/val X [örnek, lookback, özellik], y [örnek, ufuk],
    ölçekleyiciler ve özellik adları.
    Neden: Val girdisi train sonundaki geçmişi kullanabilir; val hedefleri val'de
    kalır. Test kümesine pencere veya metrik üretilmez.
    """
    split_frames = chronological_split(df, config)
    train_frame = split_frames["train"]
    val_frame = split_frames["val"]
    features = get_feature_columns(config)
    target = config["preprocess"]["target"]
    forecast = config["forecast"]

    feature_scaler = StandardScaler().fit(train_frame[features])
    target_scaler = StandardScaler().fit(
        train_frame[[target]].to_numpy(dtype=np.float64)
    )

    train_scaled = train_frame.copy()
    train_feature_values = feature_scaler.transform(train_frame[features])
    for column_index, column in enumerate(features):
        train_scaled[column] = train_feature_values[:, column_index]
    val_context = pd.concat(
        (train_frame.tail(forecast["lookback"]), val_frame)
    )
    val_scaled = val_context.copy()
    val_feature_values = feature_scaler.transform(val_context[features])
    for column_index, column in enumerate(features):
        val_scaled[column] = val_feature_values[:, column_index]

    train_X, train_y, train_times = make_windows(
        train_scaled,
        target,
        features,
        forecast["lookback"],
        forecast["horizon"],
        forecast["stride"],
        exclude_imputed=False,
    )
    val_X, val_y, val_times = make_windows(
        val_scaled,
        target,
        features,
        forecast["lookback"],
        forecast["horizon"],
        forecast["stride"],
        target_start=val_frame.index.min(),
        target_end=val_frame.index.max(),
        exclude_imputed=config["evaluation"]["exclude_imputed_targets"],
    )
    if not len(train_X):
        raise ValueError("Train kümesinde tahmin penceresi bulunamadı.")
    if not len(val_X):
        raise ValueError("Geçerli validation penceresi bulunamadı.")

    train_y_scaled = target_scaler.transform(
        train_y.reshape(-1, 1)
    ).reshape(train_y.shape)
    val_y_scaled = target_scaler.transform(
        val_y.reshape(-1, 1)
    ).reshape(val_y.shape)
    return {
        "X_train": train_X,
        "y_train": train_y,
        "y_train_scaled": train_y_scaled,
        "train_times": train_times,
        "X_val": val_X,
        "y_val": val_y,
        "y_val_scaled": val_y_scaled,
        "val_times": val_times,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "features": features,
        "train_frame": train_frame,
    }


def _upsert_csv(path, rows, key_columns):
    """CSV dosyasındaki aynı deney anahtarlarını yenileyip diğerlerini korur.

    Girdi: dosya yolu, yeni satırlar ve tekil kayıt anahtarları.
    Çıktı: güncellenmiş CSV dosyası.
    Neden: Tekrarlanan koşular eski değerleri çoğaltmadan günceller.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    new_frame = pd.DataFrame(rows)
    if path.exists():
        old_frame = pd.read_csv(path)
        keys = set(new_frame[key_columns].itertuples(index=False, name=None))
        old_keys = old_frame[key_columns].itertuples(index=False, name=None)
        old_frame = old_frame.loc[
            [key not in keys for key in old_keys]
        ]
        new_frame = pd.concat((old_frame, new_frame), ignore_index=True)
    new_frame.to_csv(path, index=False)


def _save_learning_curve(history, model_name, exp_id, figures_dir):
    """Train ve val kayıplarını etiketli PNG öğrenme eğrisi olarak kaydeder.

    Girdi: epoch/loss sözlükleri, model kimliği ve çıktı klasörü.
    Çıktı: dpi=200 PNG dosyası.
    Neden: Her iki kümenin kayıp eğilimi aynı grafikte karşılaştırılır.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, train_loss, label="Train")
    axis.plot(epochs, val_loss, label="Val")
    axis.set_title(f"{exp_id} {model_name.upper()} öğrenme eğrisi (train/val)")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE (standartlaştırılmış hedef, boyutsuz)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures_dir / f"{exp_id}_learning_curve.png", dpi=200)
    plt.close(figure)


def train_model(df, config, model_name, device_preference="auto"):
    """LSTM veya GRU'yu train'de eğitir, val MAE ile seçip checkpoint'i kaydeder.

    Girdi: saatlik DataFrame, config ve `lstm`/`gru` adı.
    Çıktı: val özet metrikleri, ufuk tablosu ve epoch geçmişi.
    Neden: Early stopping, scheduler ve checkpoint seçimi yalnız val üzerinden
    yapılır; test kümesi eğitim veya değerlendirme hattına alınmaz.
    """
    if model_name not in MODEL_CLASSES:
        raise ValueError("Model adı 'lstm' veya 'gru' olmalı.")
    set_seed(config["seed"])
    torch.set_num_threads(min(torch.get_num_threads(), 4))

    prepared = prepare_forecast_data(df, config)
    device = _forecast_device(device_preference)
    X_train = torch.from_numpy(
        prepared["X_train"].astype(np.float32)
    ).to(device)
    y_train = torch.from_numpy(
        prepared["y_train_scaled"].astype(np.float32)
    ).to(device)
    X_val = torch.from_numpy(
        prepared["X_val"].astype(np.float32)
    )
    y_val_scaled = torch.from_numpy(
        prepared["y_val_scaled"].astype(np.float32)
    )
    forecast = config["forecast"]
    model = MODEL_CLASSES[model_name](
        input_size=X_train.shape[2],
        hidden_size=forecast["hidden"],
        layers=forecast["layers"],
        dropout=forecast["dropout"],
        horizon=forecast["horizon"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=forecast["lr"],
        weight_decay=forecast["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=forecast["scheduler_factor"],
        patience=forecast["scheduler_patience"],
    )
    loss_function = nn.MSELoss()
    patience = forecast["early_stopping_patience"]
    best_mae = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    start_time = time.perf_counter()

    for epoch in range(1, forecast["max_epochs"] + 1):
        model.train()
        train_loss_total = torch.zeros((), device=device)
        train_sample_total = 0
        for batch_start in range(0, len(X_train), forecast["batch_size"]):
            batch_end = batch_start + forecast["batch_size"]
            batch_X = X_train[batch_start:batch_end]
            batch_y = y_train[batch_start:batch_end]
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = loss_function(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                forecast["grad_clip"],
            )
            optimizer.step()
            # MPS/CUDA'dan her batch'te .item() çağırmak cihazı senkronize eder.
            train_loss_total += loss.detach() * len(batch_X)
            train_sample_total += len(batch_X)

        model.eval()
        with torch.no_grad():
            val_prediction_batches = []
            val_loss_total = torch.zeros((), device=device)
            for batch_start in range(0, len(X_val), forecast["batch_size"]):
                batch_end = batch_start + forecast["batch_size"]
                batch_X = X_val[batch_start:batch_end].to(device)
                batch_y = y_val_scaled[batch_start:batch_end].to(device)
                batch_predictions = model(batch_X)
                val_prediction_batches.append(batch_predictions)
                val_loss_total += (
                    loss_function(batch_predictions, batch_y)
                    * len(batch_X)
                )
            val_predictions_scaled = torch.cat(val_prediction_batches)
            val_predictions = prepared["target_scaler"].inverse_transform(
                val_predictions_scaled.cpu().numpy().reshape(-1, 1)
            ).reshape(prepared["y_val"].shape)
        val_mae = calculate_metrics(
            prepared["y_val"],
            val_predictions,
            mape_min_y=config["evaluation"]["mape_min_y"],
        )["MAE"]
        train_loss = train_loss_total.item() / train_sample_total
        val_loss_value = val_loss_total.item() / len(X_val)
        scheduler.step(val_loss_value)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss_value,
            "val_mae": val_mae,
        })
        print(
            f"{model_name.upper()} epoch {epoch}: "
            f"train_loss={train_loss:.6f}, val_loss={val_loss_value:.6f}, "
            f"val_mae={val_mae:.6f} kW"
        )

        if val_mae < best_mae:
            best_mae = val_mae
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError(f"{model_name} için en iyi val checkpoint'i oluşmadı.")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        best_prediction_batches = []
        for batch_start in range(0, len(X_val), forecast["batch_size"]):
            batch_end = batch_start + forecast["batch_size"]
            batch_X = X_val[batch_start:batch_end].to(device)
            best_prediction_batches.append(model(batch_X))
        best_predictions_scaled = torch.cat(
            best_prediction_batches
        ).cpu().numpy()
    predictions = prepared["target_scaler"].inverse_transform(
        best_predictions_scaled.reshape(-1, 1)
    ).reshape(prepared["y_val"].shape)
    elapsed = time.perf_counter() - start_time

    exp_id = EXPERIMENT_IDS[model_name]
    models_dir = _resolve_path(
        PROJECT_ROOT,
        config["paths"]["models_dir"],
    )
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {
                name: tensor.cpu()
                for name, tensor in model.state_dict().items()
            },
            "features": prepared["features"],
            "feature_mean": prepared["feature_scaler"].mean_.tolist(),
            "feature_scale": prepared["feature_scaler"].scale_.tolist(),
            "target_mean": prepared["target_scaler"].mean_.tolist(),
            "target_scale": prepared["target_scaler"].scale_.tolist(),
            "lookback": forecast["lookback"],
            "horizon": forecast["horizon"],
            "hidden": forecast["hidden"],
            "layers": forecast["layers"],
            "dropout": forecast["dropout"],
        },
        models_dir / f"{exp_id}_best.pt",
    )

    tables_dir = _resolve_path(PROJECT_ROOT, config["paths"]["tables_dir"])
    figures_dir = _resolve_path(PROJECT_ROOT, config["paths"]["figures_dir"])
    results = calculate_metrics(
        prepared["y_val"],
        predictions,
        mape_min_y=config["evaluation"]["mape_min_y"],
    )
    summary = {
        "split": "val",
        "model": model_name.upper(),
        "windows": len(prepared["y_val"]),
        **results,
    }
    horizon_metrics = metrics_by_horizon(
        prepared["y_val"],
        predictions,
        mape_min_y=config["evaluation"]["mape_min_y"],
    )
    horizon_metrics.insert(0, "model", model_name.upper())
    horizon_metrics.insert(0, "split", "val")
    horizon_metrics["windows"] = len(prepared["y_val"])

    history_path = tables_dir / f"{exp_id}_training_history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    _save_learning_curve(history, model_name, exp_id, figures_dir)

    row = {
        "exp_id": exp_id,
        "date": date.today().isoformat(),
        "model": model_name.upper(),
        "feature_set": config["features"]["default_set"],
        "lookback": forecast["lookback"],
        "hidden": forecast["hidden"],
        "layers": forecast["layers"],
        "dropout": forecast["dropout"],
        "lr": forecast["lr"],
        "batch": forecast["batch_size"],
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "seed": config["seed"],
        "val_mae": results["MAE"],
        "val_rmse": results["RMSE"],
        "test_mae": "",
        "test_rmse": "",
        "train_time_s": elapsed,
        "note": (
            f"Direct {forecast['horizon']}-hour forecast; "
            f"features={prepared['features']}; "
            "train/val only; test not evaluated"
        ),
    }
    _upsert_csv(
        _resolve_path(PROJECT_ROOT, config["paths"]["experiments_file"]),
        [row],
        ["exp_id"],
    )
    _upsert_csv(
        tables_dir / "baseline_results.csv",
        [summary],
        ["split", "model"],
    )
    _upsert_csv(
        tables_dir / "forecast_horizon_metrics.csv",
        horizon_metrics.to_dict("records"),
        ["split", "model", "horizon"],
    )

    if device.type == "mps":
        del (
            model,
            best_state,
            X_train,
            y_train,
            X_val,
            y_val_scaled,
            train_loss_total,
            val_loss_total,
            val_prediction_batches,
            val_predictions_scaled,
            val_predictions,
            best_prediction_batches,
            best_predictions_scaled,
            predictions,
            batch_X,
            batch_y,
            batch_predictions,
        )
        torch.mps.empty_cache()

    return summary, horizon_metrics, pd.DataFrame(history)


def main():
    """CLI ile Ridge/MLP baseline'larını ve/veya LSTM/GRU'yu çalıştırır.

    Girdi: --model all|ridge|mlp|lstm|gru, --device auto|cuda|mps|cpu ve
    isteğe bağlı --config.
    Çıktı: validation sonuçları, deney CSV'leri, eğitim eğrileri ve checkpoint'ler.
    Neden: `all` tek akışta dört tahmin modelini çalıştırır; split seçeneği yoktur.
    """
    parser = argparse.ArgumentParser(
        description="Tahmin modellerini train'de eğitip yalnız val'de değerlendir."
    )
    parser.add_argument(
        "--model",
        choices=("all", "ridge", "mlp", "lstm", "gru"),
        default="all",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    data_path = _resolve_path(PROJECT_ROOT, config["paths"]["processed_file"])
    df = pd.read_parquet(data_path)
    summaries = []

    if args.model in ("all", "ridge", "mlp"):
        baseline_models = (
            ("Ridge", "MLP")
            if args.model == "all"
            else ("Ridge",)
            if args.model == "ridge"
            else ("MLP",)
        )
        baseline_results = run_forecast_baselines(
            df,
            config,
            model_names=baseline_models,
        )
        summaries.extend(baseline_results.to_dict("records"))
    deep_models = ("lstm", "gru") if args.model == "all" else (
        (args.model,) if args.model in ("lstm", "gru") else ()
    )
    for model_name in deep_models:
        result, _, _ = train_model(
            df,
            config,
            model_name,
            device_preference=args.device,
        )
        summaries.append(result)

    print("\n=== VALIDATION KARŞILAŞTIRMASI ===")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
