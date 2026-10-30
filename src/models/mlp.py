"""24 saatlik doğrudan tahmin için çok katmanlı algılayıcı (MLP, F4)."""

from torch import nn
import torch


class MLPForecaster(nn.Module):
    """Geçmiş pencereyi düzleştirip tam bağlı katmanlarla ufuk tahmini üretir."""

    def __init__(self, input_size, lookback, hidden_sizes, dropout, horizon, future_size=0):
        """Gizli katmanları (Linear → ReLU → Dropout) ve çıktı katmanını kurar.

        Girdi: input_size özellik sayısı F, lookback pencere uzunluğu L,
        hidden_sizes ör. [128, 64], dropout oranı, horizon ufuk (24),
        future_size hedef saat başına bilinen gelecek öznitelik sayısı k.
        Çıktı: model; ileri geçişte [batch, L, F] -> [batch, horizon].
        Neden: LSTM/GRU ile aynı pencereyi alır ama zaman sırasını modellemez;
        "zaman yapısını bilmeyen sinir ağı" referansı olarak karşılaştırılır.
        """
        super().__init__()
        layers = []
        # düzleştirilmiş pencere L·F (+ bilinen gelecek ufuk·k)
        in_features = lookback * input_size + horizon * future_size
        for size in hidden_sizes:
            layers += [nn.Linear(in_features, size), nn.ReLU(), nn.Dropout(dropout)]
            in_features = size
        layers.append(nn.Linear(in_features, horizon))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs, future=None):
        """[batch, L, F] (+ [batch, ufuk, k]) girdiyi düzleştirip [batch, horizon] döndürür."""
        features = inputs.flatten(start_dim=1)
        if future is not None:
            features = torch.cat((features, future.flatten(start_dim=1)), dim=1)
        return self.network(features)
