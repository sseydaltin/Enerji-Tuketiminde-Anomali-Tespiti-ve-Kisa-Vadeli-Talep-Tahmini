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
