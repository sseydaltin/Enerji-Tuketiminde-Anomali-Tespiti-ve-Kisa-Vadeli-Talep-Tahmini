"""24 saatlik doğrudan tahmin için LSTM modeli."""

from torch import nn
import torch


class LSTMForecaster(nn.Module):
    """Geçmiş pencerenin LSTM gizil durumundan ufuk tahminini üretir."""

    def __init__(self, input_size, hidden_size, layers, dropout, horizon, future_size=0):
        """LSTM ve doğrusal tahmin katmanını kurar.

        Girdi: input_size özellik sayısı; future_size hedef saat başına bilinen
        gelecek öznitelik sayısı (0 = kullanılmaz); diğerleri config'ten gelir.
        Çıktı: model, ileri geçişte [batch, zaman, özellik] -> [batch, ufuk].
        Neden: Son LSTM gizil durumu gelecek 24 saat için tek adımda çıktı verir.
        Bilinen gelecek öznitelikleri çıktı katmanına gizil durumla birlikte
        verilir; başlık yine tek doğrusal katmandır (yalnız bilgi eklenir).
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_size + horizon * future_size, horizon)

    def forward(self, inputs, future=None):
        """LSTM son katmanının son gizil durumundan çok adımlı tahmin üretir.

        Girdi: inputs [batch, lookback, özellik]; future [batch, ufuk, k] veya None.
        Çıktı: [batch, horizon].
        """
        _, (hidden, _) = self.lstm(inputs)
        features = hidden[-1]  # [batch, hidden]
        if future is not None:
            # [batch, hidden + ufuk·k]
            features = torch.cat((features, future.flatten(start_dim=1)), dim=1)
        return self.output(features)
