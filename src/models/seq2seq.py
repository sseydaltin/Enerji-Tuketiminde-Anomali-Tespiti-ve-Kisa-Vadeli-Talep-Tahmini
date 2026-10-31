"""24 saatlik tahmin için encoder–decoder (seq2seq) LSTM modeli (F10)."""

from torch import nn


class Seq2SeqForecaster(nn.Module):
    """Encoder geçmişi özetler; decoder her hedef saat için bir çıktı üretir."""

    def __init__(self, input_size, decoder_size, hidden_size, layers, dropout):
        """Encoder LSTM, decoder LSTM ve saat başına çıktı katmanını kurar.

        Girdi: input_size geçmiş öznitelik sayısı F, decoder_size hedef saat
        başına takvim öznitelik sayısı (B setinde 8), hidden/layers/dropout
        config'ten.
        Çıktı: model; ileri geçişte ([batch, L, F], [batch, 24, 8]) -> [batch, 24].
        Neden: Decoder, encoder'ın son durumundan başlar ve her adımda yalnız o
        hedef saatin takvimini alır (önceden bilinir). Önceki tahminini girdi
        olarak almaz; bu yüzden teacher forcing ve hata birikmesi yoktur.
        """
        super().__init__()
        recurrent_dropout = dropout if layers > 1 else 0.0
        self.encoder = nn.LSTM(input_size, hidden_size, layers,
                               batch_first=True, dropout=recurrent_dropout)
        self.decoder = nn.LSTM(decoder_size, hidden_size, layers,
                               batch_first=True, dropout=recurrent_dropout)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, inputs, future):
        """Girdi: inputs [batch, L, F], future [batch, 24, takvim]. Çıktı: [batch, 24]."""
        _, state = self.encoder(inputs)               # (h, c): [layers, batch, hidden]
        decoded, _ = self.decoder(future, state)      # [batch, 24, hidden]
        return self.output(decoded).squeeze(-1)       # [batch, 24]
