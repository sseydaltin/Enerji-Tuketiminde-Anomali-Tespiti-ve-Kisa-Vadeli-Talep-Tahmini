"""MLP, LSTM ve GRU tahmin modellerinin ortak eğitim/doğrulama akışı."""

import argparse
from copy import deepcopy
from datetime import date
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
from src.models.cnn_lstm import CNNLSTMForecaster
from src.models.forecast_baselines import run_ridge
from src.models.gru import GRUForecaster
from src.models.lstm import LSTMForecaster
from src.models.mlp import MLPForecaster
from src.models.seq2seq import Seq2SeqForecaster
from src.models.xgb_forecast import run_xgboost
from src.utils.console import setup_console
from src.utils.metrics import calculate_metrics
from src.utils.results_io import record_forecast_results, resolve_path, upsert_csv
from src.utils.seed import set_seed


# --exp verilmezse kullanılan varsayılan deney kimlikleri. F4 eski sklearn
# MLPRegressor koşusuna ait olduğundan PyTorch MLP ayrı kimlikle yazılır.
EXPERIMENT_IDS = {
    "mlp": "F4-torch", "lstm": "F5", "gru": "F6",
    "seq2seq": "F10", "cnn_lstm": "F11", "xgboost": "F12",
}
MODEL_LABELS = {"seq2seq": "Seq2Seq-LSTM", "cnn_lstm": "CNN-LSTM"}


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


def seasonal_positions(frame_index, target_times, horizon, season):
    """Her hedef saatin `season` saat önceki satır konumunu verir.

    Girdi: tablonun DatetimeIndex'i, target_times [N] (ilk hedef zamanları),
    ufuk ve mevsim gecikmesi (168 = bir hafta).
    Çıktı: [N, ufuk] tamsayı konumlar; hedef t+h için konum(t+h) − season.
    Neden: y[t+h−168] hem bilinen gelecek özniteliği hem de artık hedefin
    tabanıdır. season ≥ ufuk olduğundan t+h−season ≤ t−1: hepsi tahmin
    başlangıcından öncedir (gelecek bilgisi yok).
    """
    first = frame_index.get_indexer(target_times)
    if (first < 0).any():
        raise ValueError("Hedef zamanları tabloda bulunamadı.")
    return first[:, None] + np.arange(horizon)[None, :] - season


def build_future_features(frame, positions, season, target, calendar, level_scaler):
    """Bilinen gelecek özniteliklerini [N, ufuk, 1 + takvim] olarak üretir.

    Girdi: ham saatlik tablo, seasonal_positions çıktısı [N, ufuk], takvim
    sütunları ve train'e fit edilmiş tüketim ölçekleyicisi.
    Çıktı: her hedef saat için [ölçekli y(hedef − 168), hedef saatin takvimi].
    Neden: Takvim önceden bilinir; geçen haftanın aynı saati geçmiştir.
    Takvim sin/cos ve 0/1 değerleri zaten [-1, 1] aralığında olduğundan ölçeklenmez.
    """
    level = frame[target].to_numpy(dtype=np.float64)[positions]  # [N, ufuk]
    level = (level - level_scaler.mean_[0]) / level_scaler.scale_[0]
    calendar_values = frame[calendar].to_numpy(dtype=np.float64)[positions + season]
    return np.concatenate((level[..., None], calendar_values), axis=2).astype(np.float32)


def build_future_calendar(frame, target_positions, calendar):
    """Hedef saatlerin takvimini [N, ufuk, takvim] olarak verir (seq2seq decoder'ı).

    Girdi: ham saatlik tablo, hedef satır konumları [N, ufuk], takvim sütunları.
    Neden: Takvim önceden bilinir; hedef değerlerine bakılmaz.
    """
    return frame[calendar].to_numpy(dtype=np.float32)[target_positions]


def prepare_forecast_data(df, config):
    """Train/val pencerelerini train-fit ölçekleyicilerle hazırlar.

    Girdi: saatlik [zaman, özellik] DataFrame ve proje config'i.
    Çıktı: train/val X [örnek, lookback, özellik], y [örnek, ufuk] (kW),
    model hedefi (ölçekli y veya ölçekli artık), taban [örnek, ufuk] (artık
    hedefte y_SN168, değilse 0), isteğe bağlı gelecek öznitelikleri
    F [örnek, ufuk, k], ölçekleyiciler ve özellik adları.
    Neden: Val girdisi train sonundaki geçmişi kullanabilir; val hedefleri val'de
    kalır. Test kümesine pencere veya metrik üretilmez.
    """
    split_frames = chronological_split(df, config)
    train_frame = split_frames["train"]
    val_frame = split_frames["val"]
    features = get_feature_columns(config)
    target = config["preprocess"]["target"]
    forecast = config["forecast"]
    horizon = forecast["horizon"]
    season = forecast.get("season_lag", 168)
    use_future = forecast.get("future_features", False)
    use_residual = forecast.get("residual_target", False)
    decoder_calendar = forecast.get("decoder_calendar", False)
    if use_future and decoder_calendar:
        raise ValueError("future_features ve decoder_calendar birlikte kullanılmaz.")
    if (use_future or use_residual) and season < horizon:
        raise ValueError("season_lag ufuktan küçük olamaz (gelecek bilgisi sızar).")

    feature_scaler = StandardScaler().fit(train_frame[features])
    # Tüketim düzeyi ölçekleyicisi: yalnız train serisi.
    level_scaler = StandardScaler().fit(
        train_frame[[target]].to_numpy(dtype=np.float64)
    )

    train_scaled = train_frame.copy()
    train_feature_values = feature_scaler.transform(train_frame[features])
    for column_index, column in enumerate(features):
        train_scaled[column] = train_feature_values[:, column_index]
    # Val girdisi ve haftalık gecikme için train sonundan yeterli geçmiş alınır.
    val_context = pd.concat(
        (train_frame.tail(max(forecast["lookback"], season)), val_frame)
    )
    val_scaled = val_context.copy()
    val_feature_values = feature_scaler.transform(val_context[features])
    for column_index, column in enumerate(features):
        val_scaled[column] = val_feature_values[:, column_index]

    # Hedef sütunu öznitelik listesinde de olduğu için *_scaled tablolarında
    # ölçeklenmiş hâldedir. X ölçeklenmiş tablodan, y ise ham tablodan (kW)
    # alınır; böylece y yalnız target_scaler ile bir kez ölçeklenir ve
    # metrikler inverse transform sonrası gerçekten kW cinsinden hesaplanır.
    train_window_args = dict(
        target=target,
        features=features,
        lookback=forecast["lookback"],
        horizon=forecast["horizon"],
        stride=forecast["stride"],
        exclude_imputed=False,
    )
    val_window_args = dict(
        target=target,
        features=features,
        lookback=forecast["lookback"],
        horizon=forecast["horizon"],
        stride=forecast["stride"],
        target_start=val_frame.index.min(),
        target_end=val_frame.index.max(),
        exclude_imputed=config["evaluation"]["exclude_imputed_targets"],
    )
    train_X, _, train_times = make_windows(train_scaled, **train_window_args)
    _, train_y, _ = make_windows(train_frame, **train_window_args)
    val_X, _, val_times = make_windows(val_scaled, **val_window_args)
    _, val_y, val_y_times = make_windows(val_context, **val_window_args)
    # İki çağrı aynı pencereleri seçmeli (aynı indeks ve is_imputed bayrakları).
    if not val_times.equals(val_y_times):
        raise RuntimeError("Val X ve y pencereleri eşleşmiyor.")
    train_positions = seasonal_positions(train_frame.index, train_times, horizon, season)
    val_positions = seasonal_positions(val_context.index, val_times, horizon, season)
    if use_future or use_residual:
        # Train'in ilk haftasında y(hedef − 168) yoktur; bu pencereler atlanır.
        keep = train_positions.min(axis=1) >= 0
        train_X, train_y = train_X[keep], train_y[keep]
        train_times, train_positions = train_times[keep], train_positions[keep]
        if (val_positions < 0).any():
            raise RuntimeError("Val bağlamı haftalık gecikme için yetersiz.")
    if not len(train_X):
        raise ValueError("Train kümesinde tahmin penceresi bulunamadı.")
    if not len(val_X):
        raise ValueError("Geçerli validation penceresi bulunamadı.")

    if use_residual:
        # Taban = Seasonal Naive 168 tahmini (kW); model y − taban'ı öğrenir.
        train_base = train_frame[target].to_numpy(dtype=np.float64)[train_positions]
        val_base = val_context[target].to_numpy(dtype=np.float64)[val_positions]
        # Artık ölçekleyicisi yalnız train artıklarına fit edilir.
        target_scaler = StandardScaler().fit((train_y - train_base).reshape(-1, 1))
    else:
        train_base = np.zeros(train_y.shape)
        val_base = np.zeros(val_y.shape)
        target_scaler = level_scaler

    train_y_scaled = target_scaler.transform(
        (train_y - train_base).reshape(-1, 1)
    ).reshape(train_y.shape)
    val_y_scaled = target_scaler.transform(
        (val_y - val_base).reshape(-1, 1)
    ).reshape(val_y.shape)

    future_train = future_val = None
    if use_future:
        calendar = config["features"]["calendar"]
        future_train = build_future_features(
            train_frame, train_positions, season, target, calendar, level_scaler
        )
        future_val = build_future_features(
            val_context, val_positions, season, target, calendar, level_scaler
        )
    if decoder_calendar:
        calendar = config["features"]["calendar"]
        # val_positions + season = hedef saatlerin kendi satır konumları.
        future_train = build_future_calendar(train_frame, train_positions + season, calendar)
        future_val = build_future_calendar(val_context, val_positions + season, calendar)
    return {
        "X_train": train_X,
        "y_train": train_y,
        "y_train_scaled": train_y_scaled,
        "base_train": train_base,
        "future_train": future_train,
        "train_times": train_times,
        "X_val": val_X,
        "y_val": val_y,
        "y_val_scaled": val_y_scaled,
        "base_val": val_base,
        "future_val": future_val,
        "val_times": val_times,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "level_scaler": level_scaler,
        "features": features,
        "train_frame": train_frame,
    }


def build_model(model_name, input_size, forecast, future_size=0):
    """Config'teki ayarlarla MLP, LSTM veya GRU modelini kurar.

    Girdi: model adı, özellik sayısı F, config["forecast"] sözlüğü,
    hedef saat başına bilinen gelecek öznitelik sayısı (0 = yok).
    Çıktı: [batch, lookback, F] -> [batch, horizon] nn.Module.
    Neden: Üç model aynı eğitim döngüsünü kullanır; yalnız mimari değişir.
    """
    if model_name == "mlp":
        return MLPForecaster(
            input_size=input_size,
            lookback=forecast["lookback"],
            hidden_sizes=forecast["mlp"]["hidden_sizes"],
            dropout=forecast["dropout"],
            horizon=forecast["horizon"],
            future_size=future_size,
        )
    if model_name == "seq2seq":
        # future_size burada decoder girdisi = hedef saat başına takvim sayısı.
        return Seq2SeqForecaster(
            input_size=input_size,
            decoder_size=future_size,
            hidden_size=forecast["hidden"],
            layers=forecast["layers"],
            dropout=forecast["dropout"],
        )
    if model_name == "cnn_lstm":
        return CNNLSTMForecaster(
            input_size=input_size,
            channels=forecast["cnn"]["channels"],
            kernel_size=forecast["cnn"]["kernel_size"],
            pool=forecast["cnn"]["pool"],
            hidden_size=forecast["hidden"],
            layers=forecast["layers"],
            dropout=forecast["dropout"],
            horizon=forecast["horizon"],
        )
    recurrent_classes = {"lstm": LSTMForecaster, "gru": GRUForecaster}
    if model_name not in recurrent_classes:
        raise ValueError(f"Bilinmeyen model: {model_name}")
    return recurrent_classes[model_name](
        input_size=input_size,
        hidden_size=forecast["hidden"],
        layers=forecast["layers"],
        dropout=forecast["dropout"],
        horizon=forecast["horizon"],
        future_size=future_size,
    )


def _predict_in_batches(model, X, future, batch_size, device):
    """Modeli değerlendirme modunda parça parça çalıştırır.

    Girdi: X [N, lookback, F] ve future [N, ufuk, k] (veya None) CPU tensörleri.
    Çıktı: [N, horizon] cihaz tensörü.
    Neden: Tüm val kümesini tek seferde GPU'ya koymak 4 GB VRAM'i aşabilir.
    """
    outputs = []
    for batch_start in range(0, len(X), batch_size):
        batch = slice(batch_start, batch_start + batch_size)
        batch_future = None if future is None else future[batch].to(device)
        outputs.append(model(X[batch].to(device), batch_future))
    return torch.cat(outputs)


def _save_learning_curve(history, model_label, exp_id, figures_dir):
    """Train ve val kayıplarını etiketli PNG öğrenme eğrisi olarak kaydeder.

    Girdi: epoch/loss sözlükleri, model adı, exp_id ve çıktı klasörü.
    Çıktı: dpi=200 PNG dosyası.
    Neden: Her iki kümenin kayıp eğilimi aynı grafikte karşılaştırılır.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, [row["train_loss"] for row in history], label="Train")
    axis.plot(epochs, [row["val_loss"] for row in history], label="Val")
    axis.set_title(f"{exp_id} {model_label} öğrenme eğrisi (train/val)")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE (standartlaştırılmış hedef, boyutsuz)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures_dir / f"{exp_id}_learning_curve.png", dpi=200)
    plt.close(figure)


def train_model(df, config, model_name, device_preference="auto", exp_id=None):
    """MLP/LSTM/GRU'yu train'de eğitir, val MAE ile seçip sonuçları kaydeder.

    Girdi: saatlik DataFrame, config, `mlp`/`lstm`/`gru`, cihaz tercihi ve
    isteğe bağlı exp_id (verilmezse EXPERIMENT_IDS).
    Çıktı: val metrik sözlüğü ve epoch geçmişi (DataFrame).
    Neden: Early stopping, scheduler ve checkpoint seçimi yalnız val üzerinden
    yapılır; test kümesi eğitim veya değerlendirme hattına alınmaz. Üç model
    aynı optimizer, lr, scheduler, grad clip, early stopping ve pencereleri
    kullanır; böylece karşılaştırma adil olur.
    """
    exp_id = exp_id or EXPERIMENT_IDS[model_name]
    if model_name == "seq2seq":
        # Decoder'ın girdisi hedef saatlerin takvimidir (mimarinin parçası).
        config = deepcopy(config)
        config["forecast"]["decoder_calendar"] = True
    set_seed(config["seed"])
    torch.set_num_threads(min(torch.get_num_threads(), 4))

    prepared = prepare_forecast_data(df, config)
    device = _forecast_device(device_preference)
    forecast = config["forecast"]
    X_train = torch.from_numpy(prepared["X_train"].astype(np.float32)).to(device)
    y_train = torch.from_numpy(
        prepared["y_train_scaled"].astype(np.float32)
    ).to(device)
    X_val = torch.from_numpy(prepared["X_val"].astype(np.float32))
    y_val_scaled = torch.from_numpy(
        prepared["y_val_scaled"].astype(np.float32)
    ).to(device)
    target_scaler = prepared["target_scaler"]
    # Bilinen gelecek öznitelikleri: [N, ufuk, k] veya None.
    future_train = future_val = None
    future_size = 0
    if prepared["future_train"] is not None:
        future_train = torch.from_numpy(prepared["future_train"]).to(device)
        future_val = torch.from_numpy(prepared["future_val"])
        future_size = future_train.shape[2]
    shuffle = forecast.get("shuffle_train", False)

    model = build_model(model_name, X_train.shape[2], forecast, future_size).to(device)
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
    batch_size = forecast["batch_size"]
    best_mae = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    start_time = time.perf_counter()

    for epoch in range(1, forecast["max_epochs"] + 1):
        model.train()
        train_loss_total = torch.zeros((), device=device)
        # Karıştırma yalnız eğitim pencerelerinin sırasını değiştirir; pencereler
        # zaten oluşturulmuş olduğundan zaman sırası pencere içinde korunur.
        # Kapalıyken sıra 0, 1, 2, ... (eski davranışla birebir aynı).
        if shuffle:
            order = torch.randperm(len(X_train), device=device)
        else:
            order = torch.arange(len(X_train), device=device)
        for batch_start in range(0, len(X_train), batch_size):
            batch = order[batch_start:batch_start + batch_size]
            batch_X, batch_y = X_train[batch], y_train[batch]
            batch_future = None if future_train is None else future_train[batch]
            optimizer.zero_grad()
            loss = loss_function(model(batch_X, batch_future), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), forecast["grad_clip"])
            optimizer.step()
            # Her batch'te .item() çağırmak GPU'yu senkronize eder; toplam tutulur.
            train_loss_total += loss.detach() * len(batch_X)

        model.eval()
        with torch.no_grad():
            val_predictions_scaled = _predict_in_batches(
                model, X_val, future_val, batch_size, device
            )
            val_loss = loss_function(val_predictions_scaled, y_val_scaled).item()
        # Model ölçeklenmiş hedef (veya artık) üretir; inverse transform + taban
        # (artık hedefte y_SN168, değilse 0) ile kW'a dönülür, MAE kW'da hesaplanır.
        val_predictions = target_scaler.inverse_transform(
            val_predictions_scaled.cpu().numpy().reshape(-1, 1)
        ).reshape(prepared["y_val"].shape) + prepared["base_val"]
        val_mae = calculate_metrics(
            prepared["y_val"],
            val_predictions,
            mape_min_y=config["evaluation"]["mape_min_y"],
        )["MAE"]
        train_loss = train_loss_total.item() / len(X_train)
        scheduler.step(val_loss)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae,
        })
        print(
            f"{exp_id} epoch {epoch}: train_loss={train_loss:.6f}, "
            f"val_loss={val_loss:.6f}, val_mae={val_mae:.6f} kW"
        )

        # Early stopping ve model seçimi: val MAE (kW), patience config'ten.
        if val_mae < best_mae:
            best_mae = val_mae
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            best_predictions = val_predictions
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= forecast["early_stopping_patience"]:
            break

    if best_state is None:
        raise RuntimeError(f"{exp_id} için en iyi val checkpoint'i oluşmadı.")
    elapsed = time.perf_counter() - start_time
    model.load_state_dict(best_state)

    models_dir = resolve_path(PROJECT_ROOT, config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "state_dict": {
                name: tensor.cpu() for name, tensor in model.state_dict().items()
            },
            "features": prepared["features"],
            "feature_mean": prepared["feature_scaler"].mean_.tolist(),
            "feature_scale": prepared["feature_scaler"].scale_.tolist(),
            "target_mean": target_scaler.mean_.tolist(),
            "target_scale": target_scaler.scale_.tolist(),
            "forecast": forecast,
        },
        models_dir / f"{exp_id}_best.pt",
    )

    tables_dir = resolve_path(PROJECT_ROOT, config["paths"]["tables_dir"])
    figures_dir = resolve_path(PROJECT_ROOT, config["paths"]["figures_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(
        tables_dir / f"{exp_id}_training_history.csv", index=False
    )
    model_label = MODEL_LABELS.get(model_name, model_name.upper())
    _save_learning_curve(history, model_label, exp_id, figures_dir)

    feature_set = config["features"]["default_set"]
    if model_name == "mlp":
        hidden = "x".join(str(size) for size in forecast["mlp"]["hidden_sizes"])
        layers = len(forecast["mlp"]["hidden_sizes"])
    else:
        hidden, layers = forecast["hidden"], forecast["layers"]
    note = (
        f"Direct {forecast['horizon']}-hour forecast; "
        f"features={prepared['features']}; shuffle={shuffle}; "
        f"future_features={forecast.get('future_features', False)}; "
        f"decoder_calendar={forecast.get('decoder_calendar', False)}; "
        f"residual_target={forecast.get('residual_target', False)}; "
        f"device={device.type}; train/val only; test not evaluated"
    )
    scores = record_forecast_results(
        PROJECT_ROOT, config, "val", exp_id, model_label,
        prepared["y_val"], best_predictions, prepared["val_times"],
        train_time_s=elapsed, device=device.type, feature_set=feature_set,
        lookback=forecast["lookback"], note=note,
    )
    upsert_csv(
        resolve_path(PROJECT_ROOT, config["paths"]["experiments_file"]),
        [{
            "exp_id": exp_id,
            "date": date.today().isoformat(),
            "model": model_label,
            "feature_set": feature_set,
            "lookback": forecast["lookback"],
            "hidden": hidden,
            "layers": layers,
            "dropout": forecast["dropout"],
            "lr": forecast["lr"],
            "batch": batch_size,
            "epochs_run": len(history),
            "best_epoch": best_epoch,
            "seed": config["seed"],
            "val_mae": scores["MAE"],
            "val_rmse": scores["RMSE"],
            "test_mae": "",
            "test_rmse": "",
            "train_time_s": elapsed,
            "note": note,
        }],
        ["exp_id"],
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    return scores, pd.DataFrame(history)


def apply_cli_overrides(config, args):
    """CLI argümanlarını config'in kopyası üzerine yazar.

    Girdi: config sözlüğü ve argparse sonucu. Çıktı: yeni config.
    Neden: CLAUDE.md kuralı: CLI argümanları config'i ezer; orijinal config
    nesnesi değiştirilmez.
    """
    config = deepcopy(config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.feature_set is not None:
        config["features"]["default_set"] = args.feature_set
    if args.lookback is not None:
        config["forecast"]["lookback"] = args.lookback
    if args.hidden_sizes is not None:
        config["forecast"]["mlp"]["hidden_sizes"] = [
            int(size) for size in args.hidden_sizes.split(",")
        ]
    # Bayraklar yalnız açmak için kullanılır; verilmezse config değeri kalır.
    if args.shuffle:
        config["forecast"]["shuffle_train"] = True
    if args.future:
        config["forecast"]["future_features"] = True
    if args.residual:
        config["forecast"]["residual_target"] = True
    return config


def main():
    """CLI ile Ridge, MLP, LSTM veya GRU'yu çalıştırır (yalnız val).

    Girdi: --model, --exp, --feature_set, --lookback, --seed, --hidden_sizes
    (MLP, ör. 64,32), --alpha (Ridge), --device, --config.
    Çıktı: val sonuçları, experiments.csv, eğitim eğrileri ve checkpoint'ler.
    Neden: split seçeneği yoktur; test kümesi bu betikte kullanılamaz.
    """
    setup_console()  # Windows konsolunda Türkçe karakterler için UTF-8
    parser = argparse.ArgumentParser(
        description="Tahmin modellerini train'de eğitip yalnız val'de değerlendir."
    )
    parser.add_argument(
        "--model",
        choices=("all", "ridge", "mlp", "lstm", "gru", "seq2seq", "cnn_lstm", "xgboost"),
        default="all",
    )
    parser.add_argument("--exp", default=None, help="Deney kimliği (exp_id).")
    parser.add_argument("--feature_set", choices=("A", "B", "C"), default=None)
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--hidden_sizes", default=None, help="MLP, ör. 64,32")
    parser.add_argument("--alpha", type=float, default=None, help="Ridge alpha")
    parser.add_argument("--ridge_calendar", action="store_true",
                        help="Ridge: gecikmelere son girdi saatinin takvimini ekle")
    parser.add_argument("--shuffle", action="store_true",
                        help="Eğitim pencerelerini her epoch karıştır")
    parser.add_argument("--future", action="store_true",
                        help="Bilinen gelecek öznitelikleri (y[t+h-168] + takvim)")
    parser.add_argument("--residual", action="store_true",
                        help="Hedef: y - y_SN168 (tahmine SN168 geri eklenir)")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"), default="auto"
    )
    parser.add_argument(
        "--grid", action="store_true",
        help="Ridge: forecast.ridge.alpha_grid, MLP: forecast.mlp.hidden_grid",
    )
    args = parser.parse_args()
    if args.model == "all" and args.exp is not None:
        raise SystemExit("--exp tek bir modelle kullanılmalı.")

    config = apply_cli_overrides(load_config(args.config), args)
    df = pd.read_parquet(resolve_path(PROJECT_ROOT, config["paths"]["processed_file"]))
    summaries = []

    if args.grid:
        # Izgaranın her noktası ayrı exp_id ile kaydedilir; seçim val MAE ile.
        if args.model == "ridge":
            for alpha in config["forecast"]["ridge"]["alpha_grid"]:
                exp_id = f"F3-a{alpha:g}"
                scores = run_ridge(df, config, alpha=alpha, exp_id=exp_id)
                summaries.append({"exp_id": exp_id, "alpha": alpha, **scores})
        elif args.model == "mlp":
            for hidden_sizes in config["forecast"]["mlp"]["hidden_grid"]:
                grid_config = deepcopy(config)
                grid_config["forecast"]["mlp"]["hidden_sizes"] = hidden_sizes
                exp_id = "F4-torch-" + "x".join(str(size) for size in hidden_sizes)
                scores, _ = train_model(
                    df, grid_config, "mlp",
                    device_preference=args.device, exp_id=exp_id,
                )
                summaries.append({"exp_id": exp_id, **scores})
        else:
            raise SystemExit("--grid yalnız --model ridge veya mlp ile kullanılır.")
        print("\n=== IZGARA (VAL) ===")
        print(pd.DataFrame(summaries).to_string(index=False))
        return

    if args.model in ("all", "ridge"):
        scores = run_ridge(
            df, config, alpha=args.alpha, exp_id=args.exp or "F3",
            use_calendar=args.ridge_calendar,
        )
        summaries.append({"exp_id": args.exp or "F3", "model": "Ridge", **scores})
    if args.model == "xgboost":
        exp_id = args.exp or EXPERIMENT_IDS["xgboost"]
        scores = run_xgboost(df, config, exp_id=exp_id, device_preference=args.device)
        summaries.append({"exp_id": exp_id, "model": "XGBoost", **scores})
    deep_models = ("mlp", "lstm", "gru") if args.model == "all" else (
        (args.model,) if args.model not in ("ridge", "xgboost") else ()
    )
    for model_name in deep_models:
        scores, _ = train_model(
            df, config, model_name,
            device_preference=args.device, exp_id=args.exp,
        )
        summaries.append({
            "exp_id": args.exp or EXPERIMENT_IDS[model_name],
            "model": model_name.upper(),
            **scores,
        })

    print("\n=== VALIDATION KARŞILAŞTIRMASI ===")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
