"""24 saatlik doğrudan tahmin için CNN-LSTM modeli (F11)."""

from torch import nn


class CNNLSTMForecaster(nn.Module):
    """1B evrişim yerel örüntüleri çıkarır, LSTM bunların sırasını modeller."""

    def __init__(self, input_size, channels, kernel_size, pool, hidden_size,
                 layers, dropout, horizon):
        """Conv1d → ReLU → MaxPool1d → LSTM → Linear(horizon) zincirini kurar.

        Girdi: input_size öznitelik sayısı F; channels/kernel_size/pool CNN
        ayarları; hidden/layers/dropout LSTM ayarları (config'ten).
        Çıktı: model; ileri geçişte [batch, L, F] -> [batch, horizon].
        Neden: Evrişim birkaç saatlik yerel desenleri (ör. sabah yükselişi)
        özetler; havuzlama diziyi kısaltıp LSTM'in işini kolaylaştırır.
        Pencerenin tamamı geçmiş olduğundan simetrik padding sızıntı yaratmaz.
        """
        super().__init__()
        self.conv = nn.Conv1d(input_size, channels, kernel_size, padding=kernel_size // 2)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool1d(pool)
        self.lstm = nn.LSTM(channels, hidden_size, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.output = nn.Linear(hidden_size, horizon)

    def forward(self, inputs, future=None):
        """Girdi: [batch, L, F]. Çıktı: [batch, horizon]."""
        features = inputs.transpose(1, 2)               # Conv1d için [batch, F, L]
        features = self.pool(self.activation(self.conv(features)))  # [batch, C, L/pool]
        _, (hidden, _) = self.lstm(features.transpose(1, 2))       # [batch, L/pool, C]
        return self.output(hidden[-1])
