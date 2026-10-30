"""24 saatlik doğrudan tahmin için çok katmanlı algılayıcı (MLP, F4)."""

from torch import nn


class MLPForecaster(nn.Module):
    """Geçmiş pencereyi düzleştirip tam bağlı katmanlarla ufuk tahmini üretir."""

    def __init__(self, input_size, lookback, hidden_sizes, dropout, horizon):
        """Gizli katmanları (Linear → ReLU → Dropout) ve çıktı katmanını kurar.

        Girdi: input_size özellik sayısı F, lookback pencere uzunluğu L,
        hidden_sizes ör. [128, 64], dropout oranı, horizon ufuk (24).
        Çıktı: model; ileri geçişte [batch, L, F] -> [batch, horizon].
        Neden: LSTM/GRU ile aynı pencereyi alır ama zaman sırasını modellemez;
        "zaman yapısını bilmeyen sinir ağı" referansı olarak karşılaştırılır.
        """
        super().__init__()
        layers = []
        in_features = lookback * input_size  # düzleştirilmiş pencere: L·F
        for size in hidden_sizes:
            layers += [nn.Linear(in_features, size), nn.ReLU(), nn.Dropout(dropout)]
            in_features = size
        layers.append(nn.Linear(in_features, horizon))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        """[batch, L, F] girdiyi [batch, L·F] yapıp [batch, horizon] döndürür."""
        return self.network(inputs.flatten(start_dim=1))
