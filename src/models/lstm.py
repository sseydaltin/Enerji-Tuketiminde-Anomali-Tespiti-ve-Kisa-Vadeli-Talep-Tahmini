"""24 saatlik doğrudan tahmin için LSTM modeli."""

from torch import nn


class LSTMForecaster(nn.Module):
    """Geçmiş pencerenin LSTM gizil durumundan ufuk tahminini üretir."""

    def __init__(self, input_size, hidden_size, layers, dropout, horizon):
        """LSTM ve doğrusal tahmin katmanını kurar.

        Girdi: input_size özellik sayısı; diğer parametreler config'ten gelir.
        Çıktı: model, ileri geçişte [batch, zaman, özellik] -> [batch, ufuk].
        Neden: Son LSTM gizil durumu gelecek 24 saat için tek adımda çıktı verir.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_size, horizon)

    def forward(self, inputs):
        """LSTM son katmanının son gizil durumundan çok adımlı tahmin üretir.

        Girdi: [batch, lookback, özellik], çıktı: [batch, horizon].
        Neden: Geçmiş sırayı modele verir; doğrusal katman 24 hedefi beraber üretir.
        """
        _, (hidden, _) = self.lstm(inputs)
        return self.output(hidden[-1])
