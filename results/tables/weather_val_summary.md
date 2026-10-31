Val, 3 seed (42, 7, 2026) ortalama ± std (ddof=1). Kaynak: seed_stability_val.csv. W2 satırı ORACLE'dır: hedef saatlerin gerçek havası kullanıldı, gerçekte bilinemez.

| Model | Set | Rol | MAE (kW) | RMSE (kW) | sMAPE (%) | R² |
|---|---|---|---|---|---|---|
| Seq2Seq-LSTM (F10-opt) | B | referans | 0.4688 ± 0.0060 | 0.6907 ± 0.0072 | 37.77 ± 0.51 | 0.427 ± 0.012 |
| Seq2Seq-LSTM (F10-opt) | W1 | ANA: geçmiş hava | 0.4732 ± 0.0068 | 0.6923 ± 0.0097 | 38.23 ± 0.53 | 0.424 ± 0.016 |
| Seq2Seq-LSTM (F10-opt) | W2 | ORACLE / üst sınır – gerçekte bilinemez, seçimde kullanılmaz | 0.4833 ± 0.0093 | 0.7069 ± 0.0161 | 39.24 ± 1.12 | 0.399 ± 0.027 |
| LSTM (F5-opt) | B | referans | 0.4810 ± 0.0016 | 0.6822 ± 0.0006 | 40.48 ± 0.33 | 0.441 ± 0.001 |
| LSTM (F5-opt) | W1 | ANA: geçmiş hava | 0.4884 ± 0.0025 | 0.6874 ± 0.0024 | 41.13 ± 0.26 | 0.432 ± 0.004 |
