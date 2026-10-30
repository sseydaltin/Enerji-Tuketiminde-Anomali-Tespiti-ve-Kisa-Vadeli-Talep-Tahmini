"""24 saatlik doğrudan tahmin için GRU modeli."""

from torch import nn
import torch


class GRUForecaster(nn.Module):
    """Geçmiş pencerenin GRU gizil durumundan ufuk tahminini üretir."""

    def __init__(self, input_size, hidden_size, layers, dropout, horizon, future_size=0):
        """GRU ve doğrusal tahmin katmanını kurar.

        Girdi: input_size özellik sayısı; future_size hedef saat başına bilinen
        gelecek öznitelik sayısı (0 = kullanılmaz); diğerleri config'ten gelir.
        Çıktı: model, ileri geçişte [batch, zaman, özellik] -> [batch, ufuk].
        Neden: LSTM ile aynı tahmin başlığı sayesinde iki kapılı ağı karşılaştırır.
        """
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_size + horizon * future_size, horizon)

    def forward(self, inputs, future=None):
        """GRU son katmanının son gizil durumundan çok adımlı tahmin üretir.

        Girdi: inputs [batch, lookback, özellik]; future [batch, ufuk, k] veya None.
        Çıktı: [batch, horizon].
        """
        _, hidden = self.gru(inputs)
        features = hidden[-1]  # [batch, hidden]
        if future is not None:
            # [batch, hidden + ufuk·k]
            features = torch.cat((features, future.flatten(start_dim=1)), dim=1)
        return self.output(features)
