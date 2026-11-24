"""A2: 24 saatlik pencereler için LSTM-Autoencoder ve nokta skoru; A7: TCN-Autoencoder (keşif)."""

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


class TemporalBlock(nn.Module):
    """TCN bloğu: iki genişletilmiş (dilated) nedensel Conv1d + artık bağlantı.

    Girdi: [batch, in_channels, T]. Çıktı: [batch, out_channels, T].
    Nedensel: sola (kernel−1)·dilation sıfır eklenir, çıktı t yalnız ≤ t
    girdilerini görür. Artık bağlantı derin ağda gradyan akışını kolaylaştırır.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv_1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv_2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        # Kanal sayısı değişiyorsa artık yol 1x1 conv ile eşlenir.
        self.residual = (nn.Conv1d(in_channels, out_channels, 1)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        """[batch, C_in, T] → [batch, C_out, T]; uzunluk korunur."""
        out = self.conv_1(nn.functional.pad(x, (self.left_pad, 0)))   # yalnız sola dolgu
        out = self.dropout(torch.relu(out))
        out = self.conv_2(nn.functional.pad(out, (self.left_pad, 0)))
        out = self.dropout(torch.relu(out))
        return torch.relu(out + self.residual(x))


def receptive_field(kernel_size, dilations):
    """TCN yığınının alıcı alanı (saat): 1 + 2·(k−1)·Σd (blok başına iki conv)."""
    return 1 + 2 * (kernel_size - 1) * sum(dilations)


class TCNAutoencoder(nn.Module):
    """A7 (keşif): TCN encoder → zamansal havuzlama darboğazı → TCN decoder.

    Şekiller (varsayılan: window 24, channels 32, latent_channels 2, pool 4):
    [B, 24, 1] → TCN [B, 32, 24] → 1x1 conv [B, 2, 24] → AvgPool [B, 2, 6]
    (12 sayı < 24 girdi: gerçek darboğaz) → Upsample [B, 2, 24] → TCN [B, 32, 24]
    → 1x1 conv [B, 1, 24] → [B, 24, 1].
    Neden: LSTM-AE'ye alternatif; konvolüsyonlar paralel çalışır, alıcı alan
    dilation ile pencerenin tamamını kapsar. Havuzlama olmasaydı her saat kendi
    kanalında kopyalanıp kimlik fonksiyonu kolayca öğrenilebilirdi.
    """

    def __init__(self, n_features, channels, latent_channels, kernel_size, dilations, pool, dropout):
        super().__init__()
        encoder_blocks, decoder_blocks = [], []
        in_channels = n_features
        for dilation in dilations:
            encoder_blocks.append(TemporalBlock(in_channels, channels, kernel_size, dilation, dropout))
            in_channels = channels
        in_channels = latent_channels
        for dilation in dilations:
            decoder_blocks.append(TemporalBlock(in_channels, channels, kernel_size, dilation, dropout))
            in_channels = channels
        self.encoder = nn.Sequential(*encoder_blocks)
        self.to_latent = nn.Conv1d(channels, latent_channels, 1)
        self.pool = nn.AvgPool1d(pool)
        self.upsample = nn.Upsample(scale_factor=pool, mode="nearest")
        self.decoder = nn.Sequential(*decoder_blocks)
        self.output = nn.Conv1d(channels, n_features, 1)
        self.pool_size = pool

    def forward(self, windows):
        """Girdi: [batch, T, F] (LSTM-AE ile aynı). Çıktı: aynı şekilde yeniden yapılandırma."""
        if windows.shape[1] % self.pool_size != 0:
            raise ValueError("Pencere uzunluğu pool değerine tam bölünmeli.")
        x = windows.transpose(1, 2)                 # Conv1d [batch, F, T] bekler
        latent = self.pool(self.to_latent(self.encoder(x)))   # [batch, latent, T/pool]
        decoded = self.decoder(self.upsample(latent))         # [batch, channels, T]
        return self.output(decoded).transpose(1, 2)           # [batch, T, F]


def build_autoencoder(settings):
    """config ayarlarından AE kurar; `arch` yoksa LSTM (eski A2 checkpoint'leri)."""
    if settings.get("arch", "lstm") == "tcn":
        return TCNAutoencoder(1, settings["channels"], settings["latent_channels"],
                              settings["kernel_size"], settings["dilations"],
                              settings["pool"], settings["dropout"])
    return LSTMAutoencoder(1, settings["hidden"], settings["latent"], settings["dropout"])


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
