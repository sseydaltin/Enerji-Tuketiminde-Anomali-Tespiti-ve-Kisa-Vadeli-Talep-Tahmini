"""A2: 24 saatlik pencereler için LSTM-Autoencoder ve nokta skoru."""

import numpy as np
import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    """Encoder 64→32 gizil vektöre sıkıştırır, simetrik decoder geri kurar."""

    def __init__(self, n_features, hidden, latent, dropout):
        """Katmanlar: enc LSTM(F→64) → LSTM(64→32) | dec LSTM(32→32) → LSTM(32→64) → Linear(64→F).

        Girdi: n_features (yalnız hedef → 1), hidden=64, latent=32, dropout.
        Neden: Normal günlük deseni öğrenen dar bir boğaz; anormal pencereler
        iyi geri kurulamaz ve yeniden yapılandırma hatası büyür.
        """
        super().__init__()
        self.encoder_1 = nn.LSTM(n_features, hidden, batch_first=True)
        self.encoder_2 = nn.LSTM(hidden, latent, batch_first=True)
        self.decoder_1 = nn.LSTM(latent, latent, batch_first=True)
        self.decoder_2 = nn.LSTM(latent, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden, n_features)

    def forward(self, windows):
        """Girdi: [batch, 24, F]. Çıktı: aynı şekilde yeniden yapılandırma."""
        encoded, _ = self.encoder_1(windows)                 # [batch, 24, 64]
        _, (latent, _) = self.encoder_2(self.dropout(encoded))
        latent = latent[-1]                                  # [batch, 32]
        # Gizil vektör her zaman adımına kopyalanır: [batch, 24, 32]
        repeated = latent.unsqueeze(1).repeat(1, windows.shape[1], 1)
        decoded, _ = self.decoder_1(repeated)
        decoded, _ = self.decoder_2(self.dropout(decoded))   # [batch, 24, 64]
        return self.output(decoded)


def make_windows(values, window, stride=1):
    """1B seriden [N, window] pencereler ve başlangıç konumlarını üretir."""
    starts = np.arange(0, len(values) - window + 1, stride)
    return np.stack([values[s:s + window] for s in starts]), starts


def point_scores(errors, starts, length, window):
    """Pencere içi konum hatalarını saatlere dağıtıp ortalamasını alır.

    Girdi: errors [N, window] (her pencerenin her konumdaki hatası),
    starts [N] pencere başlangıçları, serinin uzunluğu.
    Çıktı: [length] nokta skoru; hiçbir pencereye girmeyen saatte NaN.
    Neden: Bir saat 24 farklı pencerede yer alır; o saatteki hataların
    ortalaması tek pencereye göre daha kararlıdır (README §4.5).
    """
    total = np.zeros(length)
    count = np.zeros(length)
    for offset in range(window):
        np.add.at(total, starts + offset, errors[:, offset])
        np.add.at(count, starts + offset, 1)
    with np.errstate(invalid="ignore"):
        return np.where(count > 0, total / count, np.nan)


def reconstruct(model, windows, batch_size, device):
    """Pencereleri parça parça geri kurar. Girdi/çıktı: [N, 24] numpy (ölçekli)."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start:start + batch_size, :, None]).float()
            outputs.append(model(batch.to(device)).squeeze(-1).cpu().numpy())
    return np.concatenate(outputs)
