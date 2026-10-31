"""A3: tahmin artığı dedektörü — 1 adım ileri (h=1) tahminin mutlak hatası.

Tahmin modelleri seyda/model-fixes dalında YALNIZ TRAIN ile eğitildi (F10-opt
seq2seq, F5-opt LSTM; final topluluk F13-ens = 0,8 · F10-opt + 0,2 · F5-opt).
Burada yalnız kayıtlı ağırlıklar yüklenir; yeniden eğitim yoktur.
"""

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view

from src.models.lstm import LSTMForecaster
from src.models.seq2seq import Seq2SeqForecaster


def load_forecaster(checkpoint_path, device):
    """Kayıtlı tahmin checkpoint'inden modeli kurar.

    Çıktı: (eval modunda model, checkpoint sözlüğü). Checkpoint öznitelik
    listesini ve train'e fit edilmiş ölçek istatistiklerini içerir.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    forecast = checkpoint["forecast"]
    input_size = len(checkpoint["features"])
    if checkpoint["model_name"] == "seq2seq":
        # Decoder girdisi = hedef saat takvimi (B setinde hedef dışındaki 8 sütun).
        model = Seq2SeqForecaster(input_size, input_size - 1, forecast["hidden"],
                                  forecast["layers"], forecast["dropout"])
    elif checkpoint["model_name"] == "lstm":
        model = LSTMForecaster(input_size, forecast["hidden"], forecast["layers"],
                               forecast["dropout"], forecast["horizon"])
    else:
        raise ValueError(f"A3 için desteklenmeyen model: {checkpoint['model_name']}")
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def one_step_predictions(model, checkpoint, frame, device, batch_size=1024):
    """Her saat t için yalnız t'den önceki `lookback` saatle ŷ[t] (h=1, kW) üretir.

    Girdi: model + checkpoint, saatlik tablo (geçmiş bağlam dahil; enjekte
    kopyada anomaliler girdi penceresine de girer — gerçekçi durum).
    Çıktı: frame ile aynı indeksli Series; ilk `lookback` saat NaN.
    Neden: Pencere t−lookback … t−1 (gelecek yok). Seq2seq decoder'ı nedensel
    bir LSTM olduğundan ilk çıktı yalnız ilk hedef saatin takvimine bağlıdır;
    bu yüzden decoder'a yalnız t saatinin takvimi verilir ([N, 1, 8]).
    """
    features = checkpoint["features"]
    lookback = checkpoint["forecast"]["lookback"]
    mean = np.asarray(checkpoint["feature_mean"])
    scale = np.asarray(checkpoint["feature_scale"])
    values = ((frame[features].to_numpy(dtype=np.float64) - mean) / scale).astype(np.float32)
    # [T − lookback + 1, F, lookback] → [N, lookback, F]; son pencere hedefsiz kalır.
    windows = sliding_window_view(values, lookback, axis=0).transpose(0, 2, 1)[:-1]
    # [N, 1, takvim]: t saatinin takvimi. Eğitimdeki gibi HAM (ölçeklenmemiş) değerler
    # (model-fixes build_future_calendar; sin/cos ve 0/1 zaten [-1, 1] aralığında).
    calendar = frame[features[1:]].to_numpy(dtype=np.float32)[lookback:][:, None, :]
    outputs = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            X = torch.from_numpy(np.ascontiguousarray(windows[start:start + batch_size])).to(device)
            if checkpoint["model_name"] == "seq2seq":
                future = torch.from_numpy(np.ascontiguousarray(calendar[start:start + batch_size])).to(device)
                prediction = model(X, future)[:, 0]
            else:
                prediction = model(X)[:, 0]               # 24 çıktının ilki = h=1
            outputs.append(prediction.cpu().numpy())
    scaled = np.concatenate(outputs)
    kw = scaled * checkpoint["target_scale"][0] + checkpoint["target_mean"][0]
    result = pd.Series(np.nan, index=frame.index)
    result.iloc[lookback:] = kw
    return result


def multi_step_predictions(model, checkpoint, frame, device, batch_size=1024):
    """A-5a: her saat t ve ufuk h = 1…24 için ŷ_h[t] (kW) — t−h saatine kadarki geçmişle yapılan tahmin.

    Girdi: seq2seq model + checkpoint, saatlik tablo (geçmiş bağlam dahil).
    Çıktı: [T, 24] DataFrame (sütunlar 1…24 = ufuk h), frame ile aynı indeksli.
    Neden: Başlangıcı s olan tahmin (pencere s−lookback … s−1) s, s+1, …, s+23
    saatlerini verir. t saatinin h ufuklu tahmini, başlangıcı s = t−h+1 olan
    tahminin h. çıktısıdır; yani son görülen saat t−h. Uzun ufuklu tahmin,
    anomali başlamadan önceki geçmişe dayanır; seviye kayması girdi penceresine
    girse bile eski düzeyi "hatırlar". Yeterli geçmişi olmayan hücreler NaN.
    """
    if checkpoint["model_name"] != "seq2seq":
        raise ValueError("Çok ufuklu skor yalnız seq2seq ile hesaplanır.")
    features = checkpoint["features"]
    lookback = checkpoint["forecast"]["lookback"]
    horizon = checkpoint["forecast"]["horizon"]
    mean = np.asarray(checkpoint["feature_mean"])
    scale = np.asarray(checkpoint["feature_scale"])
    values = ((frame[features].to_numpy(dtype=np.float64) - mean) / scale).astype(np.float32)
    windows = sliding_window_view(values, lookback, axis=0).transpose(0, 2, 1)[:-1]   # [N, lookback, F]
    # Decoder takvimi (HAM): s … s+23. Sondaki eksik saatler 0 ile doldurulur; decoder
    # nedensel olduğundan k. çıktı yalnız ilk k+1 saatin takvimine bağlıdır ve
    # dolgu yalnız tablonun dışına düşen (kullanılmayan) çıktıları etkiler.
    calendar = frame[features[1:]].to_numpy(dtype=np.float32)
    padded = np.vstack([calendar, np.zeros((horizon - 1, calendar.shape[1]), dtype=np.float32)])
    future = sliding_window_view(padded, horizon, axis=0).transpose(0, 2, 1)[lookback:]   # [N, 24, takvim]
    outputs = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            X = torch.from_numpy(np.ascontiguousarray(windows[start:start + batch_size])).to(device)
            F = torch.from_numpy(np.ascontiguousarray(future[start:start + batch_size])).to(device)
            outputs.append(model(X, F).cpu().numpy())                         # [B, 24]
    by_start = np.full((len(frame), horizon), np.nan)
    by_start[lookback:] = (np.concatenate(outputs) * checkpoint["target_scale"][0]
                           + checkpoint["target_mean"][0])                    # satır = başlangıç s
    by_target = np.full((len(frame), horizon), np.nan)
    for h in range(1, horizon + 1):
        # ŷ_h[t] = başlangıcı t−h+1 olan tahminin h. çıktısı (sütun h−1).
        by_target[h - 1:, h - 1] = by_start[:len(frame) - h + 1, h - 1]
    return pd.DataFrame(by_target, index=frame.index, columns=range(1, horizon + 1))


def combine_predictions(predictions, weights):
    """Topluluk h=1 tahmini: Σ ağırlık · üye tahmini (F13-ens ağırlıkları)."""
    return sum(weights[name] * predictions[name] for name in weights)


def residual_std_table(train, prediction, target):
    """Train artıklarının saat × haftanın günü hücrelerindeki std'si (168 hücre).

    Girdi: train tablosu (enjeksiyonsuz), aynı indeksli h=1 tahmini, hedef.
    Çıktı: [hour, dow, std] DataFrame.
    Neden: Hata büyüklüğü saate ve güne göre değişir (akşam tepesi oynak, gece
    sakin). Normalize skor "bu saat için olağan hatanın kaç katı?" sorusunu
    yanıtlar. Yalnız train artıkları kullanılır; doldurulmuş saatler hariç.
    Not: Model train'de eğitildiği için bu artıklar val artıklarından biraz küçüktür.
    """
    residual = train[target] - prediction
    valid = residual.notna() & (train["is_imputed"] == 0)
    residual = residual[valid]
    table = residual.groupby([residual.index.hour, residual.index.dayofweek]).std(ddof=1)
    table.index.names = ["hour", "dow"]
    return table.rename("std").reset_index()


def normalized_scores(absolute_error, std_table):
    """|hata| / std(saat, gün). Girdi: mutlak hata Series, residual_std_table çıktısı."""
    lookup = std_table.set_index(["hour", "dow"])["std"]
    keys = pd.MultiIndex.from_arrays([absolute_error.index.hour, absolute_error.index.dayofweek])
    return absolute_error / lookup.reindex(keys).to_numpy()
