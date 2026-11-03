"""Demo sayfaları için güvenli ve salt okunur veri/model servisleri."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.data.load import PROJECT_ROOT, load_config
from src.data.split import chronological_split
from src.data.windows import get_feature_columns
from src.train_forecast import build_model
from src.utils.metrics import calculate_metrics
from src.utils.results_io import resolve_path


@st.cache_data(show_spinner=False)
def load_household_data(parquet_path):
    """Saatlik household tablosunu okur.

    Girdi: parquet dosyasının mutlak yolu.
    Çıktı: DatetimeIndex ve işlenmiş enerji/takvim sütunları olan DataFrame.
    Neden: Demo her Streamlit yenilemesinde diski tekrar okumamalıdır.
    """
    path = Path(parquet_path)
    if not path.is_file():
        raise FileNotFoundError(f"Household verisi bulunamadı: {path}")
    frame = pd.read_parquet(path)
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Saatlik household tablosunun indeksi DatetimeIndex olmalı.")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("Saatlik household indeksi sıralı ve benzersiz olmalı.")
    required_columns = {"Global_active_power", "is_imputed"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Household tablosunda gerekli sütunlar yok: {missing_columns}")
    return frame


@st.cache_resource(show_spinner="Modeller yükleniyor...")
def load_f13_models(models_dir, expected_features):
    """F13'ün iki train checkpoint'ini güvenli biçimde yükler.

    Girdi: model klasörü ve config'ten beklenen sıralı B öznitelikleri.
    Çıktı: F10-opt seq2seq ve F5-opt LSTM için model/checkpoint eşleri.
    Neden: Ağırlıklar eğitim görmez; yalnız CPU'da güvenli şekilde yüklenip
    önbelleğe alınır. Özellik uyumsuzluğu tahmin öncesi açık hata üretir.
    """
    model_root = Path(models_dir)
    loaded = {}
    for exp_id, required_architecture in (
        ("F10-opt", "seq2seq"),
        ("F5-opt", "lstm"),
    ):
        checkpoint_path = model_root / f"{exp_id}_best.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Gerekli model checkpoint'i bulunamadı: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        required_metadata = {
            "model_name",
            "features",
            "forecast",
            "feature_mean",
            "feature_scale",
            "target_mean",
            "target_scale",
            "state_dict",
        }
        missing_metadata = sorted(required_metadata - set(checkpoint))
        if missing_metadata:
            raise ValueError(
                f"{checkpoint_path.name} checkpoint metadata'sı eksik: {missing_metadata}"
            )
        if checkpoint.get("model_name") != required_architecture:
            raise ValueError(
                f"{checkpoint_path.name}: {required_architecture} mimarisi bekleniyordu, "
                f"{checkpoint.get('model_name')!r} bulundu."
            )
        features = checkpoint.get("features")
        if features != list(expected_features):
            raise ValueError(
                f"{checkpoint_path.name} özellikleri config ile uyuşmuyor. "
                f"Checkpoint={features}, beklenen={list(expected_features)}"
            )
        forecast = checkpoint.get("forecast", {})
        if forecast.get("lookback") != 48 or forecast.get("horizon") != 24:
            raise ValueError(
                f"{checkpoint_path.name}: demo 48 saat giriş ve 24 saat çıktı bekliyor; "
                f"metadata lookback={forecast.get('lookback')}, horizon={forecast.get('horizon')}."
            )
        if len(checkpoint.get("feature_mean", [])) != len(features):
            raise ValueError(f"{checkpoint_path.name}: feature_mean boyutu özellik listesiyle uyuşmuyor.")
        if len(checkpoint.get("feature_scale", [])) != len(features):
            raise ValueError(f"{checkpoint_path.name}: feature_scale boyutu özellik listesiyle uyuşmuyor.")
        if len(checkpoint.get("target_mean", [])) != 1 or len(checkpoint.get("target_scale", [])) != 1:
            raise ValueError(f"{checkpoint_path.name}: hedef scaler metadata'sı geçersiz.")
        scales = np.asarray(
            list(checkpoint["feature_scale"]) + list(checkpoint["target_scale"]),
            dtype=np.float64,
        )
        if not np.isfinite(scales).all() or (scales == 0).any():
            raise ValueError(f"{checkpoint_path.name}: scaler scale değerleri geçersiz.")

        if required_architecture == "seq2seq":
            future_size = len(features) - 1
            if not forecast.get("decoder_calendar") or future_size < 1:
                raise ValueError(f"{checkpoint_path.name}: beklenen takvim decoder metadata'sı yok.")
        else:
            future_size = 0

        model = build_model(required_architecture, len(features), forecast, future_size)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        loaded[exp_id] = (model, checkpoint)
    return loaded


@st.cache_data(show_spinner=False)
def load_demo_config(config_path):
    """Demo yapılandırmasını yükler ve önbelleğe alır."""
    return load_config(config_path)


def valid_test_days(frame, config, lookback=48, horizon=24):
    """Gerçek test kümesinde geçmişi ve hedefi eksiksiz olan günleri bulur.

    Girdi: saatlik tablo, config ve pencere boyutları.
    Çıktı: YYYY-MM-DD metinleri; her birinde 48 saat gerçek geçmiş ve 24 saat
    doldurulmamış test hedefi bulunur.
    Neden: Örnek grafiklerde eksik veya imputasyonla oluşturulmuş ölçüm gerçek
    tüketimmiş gibi gösterilmemelidir.
    """
    test = chronological_split(frame, config)["test"]
    target = config["preprocess"]["target"]
    imputed = frame["is_imputed"].astype(bool)
    days = []
    for day in pd.date_range(test.index.min().normalize(), test.index.max().normalize(), freq="D"):
        target_times = pd.date_range(day, periods=horizon, freq="h")
        history_times = pd.date_range(day - pd.Timedelta(hours=lookback), periods=lookback, freq="h")
        if not target_times.isin(test.index).all() or not history_times.isin(frame.index).all():
            continue
        if imputed.reindex(history_times).any() or imputed.reindex(target_times).any():
            continue
        values = frame[target].reindex(history_times.append(target_times)).to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            continue
        days.append(day.strftime("%Y-%m-%d"))
    if not days:
        raise ValueError("48 saatlik gerçek geçmişi ve 24 saatlik gerçek hedefi olan test günü bulunamadı.")
    return days


def _day_inputs(frame, day, checkpoint):
    """Bir gün için checkpoint özelliklerinden normalize model girdilerini kurar."""
    forecast = checkpoint["forecast"]
    target_times = pd.date_range(day, periods=forecast["horizon"], freq="h")
    history_times = pd.date_range(
        day - pd.Timedelta(hours=forecast["lookback"]),
        periods=forecast["lookback"],
        freq="h",
    )
    features = checkpoint["features"]
    absent = [name for name in features if name not in frame.columns]
    if absent:
        raise ValueError(f"Checkpoint özellikleri household verisinde yok: {absent}")

    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float64)
    feature_scale = np.asarray(checkpoint["feature_scale"], dtype=np.float64)
    raw_history = frame.loc[history_times, features].to_numpy(dtype=np.float64)
    inputs = ((raw_history - feature_mean) / feature_scale).astype(np.float32)
    if not np.isfinite(inputs).all():
        raise ValueError("Model giriş penceresinde sonlu olmayan değer var.")

    future = None
    if checkpoint["model_name"] == "seq2seq":
        calendar_features = features[1:]
        future = frame.loc[target_times, calendar_features].to_numpy(dtype=np.float32)[None, ...]
    actual = frame.loc[target_times, checkpoint["features"][0]].to_numpy(dtype=np.float64)
    return history_times, target_times, inputs[None, ...], future, actual


def predict_test_day(frame, day, models, mape_min_y=0.1):
    """Seçili test günü için F13, SN168 ve gerçek tüketimi üretir.

    Girdi: household saatlik DataFrame, YYYY-MM-DD test günü ve yüklenmiş F13
    model eşleri.
    Çıktı: 48 saat geçmişi, 24 saat hedef/tahmin serileri ve örnek metrikler.
    Neden: Model ve scaler istatistikleri checkpoint metadata'sından gelir;
    tahminler yalnız geçmiş girişini kullanıp gerçek hedef zamanlarına hizalanır.
    """
    day = pd.Timestamp(day).normalize()
    seq_model, seq_checkpoint = models["F10-opt"]
    lstm_model, lstm_checkpoint = models["F5-opt"]
    if seq_checkpoint["features"] != lstm_checkpoint["features"]:
        raise ValueError("F13 üyelerinin checkpoint özellik listeleri eşleşmiyor.")
    if seq_checkpoint["forecast"]["lookback"] != lstm_checkpoint["forecast"]["lookback"]:
        raise ValueError("F13 üyelerinin lookback değerleri eşleşmiyor.")
    if seq_checkpoint["forecast"]["horizon"] != lstm_checkpoint["forecast"]["horizon"]:
        raise ValueError("F13 üyelerinin horizon değerleri eşleşmiyor.")

    history_times, target_times, inputs, future, actual = _day_inputs(
        frame, day, seq_checkpoint
    )
    _, lstm_target_times, lstm_inputs, _, _ = _day_inputs(frame, day, lstm_checkpoint)
    if not target_times.equals(lstm_target_times) or not np.array_equal(inputs, lstm_inputs):
        raise ValueError("F13 ensemble üyeleri için pencere girdileri eşleşmiyor.")

    def run_model(model, checkpoint, model_input, decoder_input=None):
        """Bir modelin [1, lookback, özellik] girdisinden 24 kW tahmini üretir."""
        with torch.inference_mode():
            input_tensor = torch.from_numpy(model_input)
            if decoder_input is None:
                scaled = model(input_tensor)
            else:
                scaled = model(input_tensor, torch.from_numpy(decoder_input))
        prediction = scaled.cpu().numpy().reshape(-1)
        if prediction.shape != (24,) or not np.isfinite(prediction).all():
            raise ValueError(f"{checkpoint['model_name']} modeli geçersiz 24 saat tahmini üretti.")
        return prediction * checkpoint["target_scale"][0] + checkpoint["target_mean"][0]

    seq_prediction = run_model(seq_model, seq_checkpoint, inputs, future)
    lstm_prediction = run_model(lstm_model, lstm_checkpoint, lstm_inputs)
    ensemble_prediction = 0.8 * seq_prediction + 0.2 * lstm_prediction

    target_name = seq_checkpoint["features"][0]
    seasonal_times = target_times - pd.Timedelta(hours=168)
    if not seasonal_times.isin(frame.index).all():
        raise ValueError("SN168 için önceki haftanın aynı saatleri veride bulunamadı.")
    seasonal_prediction = frame.loc[seasonal_times, target_name].to_numpy(dtype=np.float64)
    if not np.isfinite(actual).all() or not np.isfinite(ensemble_prediction).all():
        raise ValueError("Gerçek hedef veya F13 tahminleri sonlu değil.")

    return {
        "history_times": history_times,
        "history_kw": frame.loc[history_times, target_name].to_numpy(dtype=np.float64),
        "target_times": target_times,
        "actual_kw": actual,
        "f13_kw": ensemble_prediction,
        "sn168_kw": seasonal_prediction,
        "metrics": calculate_metrics(actual, ensemble_prediction, mape_min_y=mape_min_y),
    }


@st.cache_data(show_spinner="Görselleştirme örnekleri hesaplanıyor...")
def daily_test_mae(frame, days, _models, mape_min_y=0.1):
    """Kolay/zor görsel örneği için test günlerini F13 MAE'ye göre sıralar.

    Girdi: geçerli test günleri ve modeller.
    Çıktı: her gün için F13 örnek MAE'si.
    Neden: Yalnız görselleştirme örneği seçilir; bu değerler model veya eşik
    seçimi için kullanılmamalıdır.
    """
    return {
        day: predict_test_day(frame, day, _models, mape_min_y)["metrics"]["MAE"]
        for day in days
    }


def demo_paths():
    """Proje config'indeki demo veri ve model yollarını döndürür."""
    config = load_demo_config(str(PROJECT_ROOT / "configs/config.yaml"))
    data_path = resolve_path(PROJECT_ROOT, config["paths"]["processed_file"])
    models_dir = resolve_path(PROJECT_ROOT, config["paths"]["models_dir"])
    expected_features = get_feature_columns(config, feature_set="B")
    return config, data_path, models_dir, expected_features
