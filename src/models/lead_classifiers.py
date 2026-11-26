"""LEAD denetimli dizi sınıflandırıcıları: L2 GRU (ana sinir ağı) ve L3 1D-CNN + GRU (protokol §8).

Girdi: son 48 saat [batch, 48, C]. Kanal 0 = ölçeklenmiş log okuma (eksikse ileri doldurulmuş).
Model, pencerenin medyanına göre göreli okumayı kendisi ekler (yalnız pencere = yalnız geçmiş).
Çıktı: pencerenin son saati (t) için anomali logit'i [batch].
"""

import torch
from torch import nn


def add_relative_channel(x):
    """[B, L, C] → [B, L, C+1]: kanal 0'ın pencere medyanından farkı (bina ölçeğinden bağımsız şekil)."""
    median = x[:, :, :1].median(dim=1, keepdim=True).values      # [B, 1, 1]
    return torch.cat([x, x[:, :, :1] - median], dim=2)


class GRUClassifier(nn.Module):
    """L2: GRU → son gizli durum → doğrusal katman (logit)."""

    def __init__(self, channels, hidden, layers, dropout):
        """channels: girdi kanal sayısı C (göreli kanal hariç); hidden, layers, dropout Optuna'dan."""
        super().__init__()
        self.gru = nn.GRU(channels + 1, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        """x [B, 48, C] → logit [B]."""
        out, _ = self.gru(add_relative_channel(x))                # [B, 48, hidden]
        return self.head(out[:, -1]).squeeze(-1)


class CNNGRUClassifier(nn.Module):
    """L3: Conv1d (nedensel dolgu) → ReLU → GRU → doğrusal katman.

    Evrişim yalnız sol tarafa dolgu yapar; t'deki çıktı t'den sonraki saati görmez (zaten pencere t'de biter).
    """

    def __init__(self, channels, hidden, layers, dropout, filters, kernel):
        super().__init__()
        self.pad = nn.ConstantPad1d((kernel - 1, 0), 0.0)         # nedensel: yalnız geçmişe dolgu
        self.conv = nn.Conv1d(channels + 1, filters, kernel)
        self.gru = nn.GRU(filters, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        """x [B, 48, C] → logit [B]."""
        z = add_relative_channel(x).transpose(1, 2)               # [B, C+1, 48]
        z = torch.relu(self.conv(self.pad(z))).transpose(1, 2)    # [B, 48, filters]
        out, _ = self.gru(z)
        return self.head(out[:, -1]).squeeze(-1)
