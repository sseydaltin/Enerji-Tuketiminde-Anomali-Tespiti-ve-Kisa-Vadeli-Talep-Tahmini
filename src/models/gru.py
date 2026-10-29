"""24 saatlik doğrudan tahmin için GRU modeli."""

from torch import nn


class GRUForecaster(nn.Module):
    """Geçmiş pencerenin GRU gizil durumundan ufuk tahminini üretir."""

    def __init__(self, input_size, hidden_size, layers, dropout, horizon):
        """GRU ve doğrusal tahmin katmanını kurar.

        Girdi: input_size özellik sayısı; diğer parametreler config'ten gelir.
        Çıktı: model, ileri geçişte [batch, zaman, özellik] -> [batch, ufuk].
        Neden: LSTM ile benzer tahmin başlığı sayesinde iki kapılı ağı karşılaştırır.
        """
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_size, horizon)

    def forward(self, inputs):
        """GRU son katmanının son gizil durumundan çok adımlı tahmin üretir.

        Girdi: [batch, lookback, özellik], çıktı: [batch, horizon].
        Neden: Geçmiş sırayı modele verir; doğrusal katman 24 hedefi beraber üretir.
        """
        _, hidden = self.gru(inputs)
        return self.output(hidden[-1])
