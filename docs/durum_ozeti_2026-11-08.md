# Durum özeti (2026-11-08)

Dal: `dilara/inceleme-zorunlu-duzeltmeler` (yerel). Ayrıntı: `docs/inceleme_ve_iyilestirme_plani_2026-11-08.md`.
Bütün sayıların kaynağı `results/tables/` altındaki dosyalardır (her satırda belirtildi).

## 1. Final sonuçlar (test kümesi 2026-10-31'de bir kez kullanıldı; değişmedi)

### Tahmin, hane testi, 24 saat (`final_forecast_val_vs_test.csv`)

| Model | Test MAE (kW) | Test RMSE (kW) | Test R² |
|---|---|---|---|
| F2 Seasonal Naive 168 | 0,549 | 0,795 | −0,244 |
| F3-cal Ridge | 0,467 | 0,612 | 0,262 |
| F5-opt LSTM (3 seed) | 0,413 ± 0,005 | 0,581 | 0,336 |
| F10-opt Seq2Seq (3 seed) | 0,414 ± 0,007 | 0,612 | 0,263 |
| F12-L336 XGBoost | 0,416 | 0,568 | 0,364 |
| **F13-ens (final, 3 seed)** | **0,406 ± 0,005** | 0,598 | 0,297 |

F13-ens, SN168'e göre +%26,0 (`final_sonuclar.md` §1.3: SN24'e göre +%20,6). H2 karşılandı.

### Anomali, hane testi, ana senaryo, 5 seed (`final_anomaly_val_vs_test.csv`)

| Yöntem | Olay F1 (blok tanımı) | PR-AUC |
|---|---|---|
| A0 z-skoru | 0,398 ± 0,064 | 0,148 |
| **A3-s2s-norm (final)** | **0,384 ± 0,170** | 0,241 |
| A3-s2s-hmean | 0,454 ± 0,036 | 0,264 |
| A5-mean-z | 0,463 ± 0,048 | 0,213 |

H3 (F1 ≥ 0,70) karşılanmadı.

## 2. Geliştirme sonuçları (final değil; yalnız kış-dev + yaz-dev, orta senaryo)

### Aşama 3 (`anomaly_short_event_summary.csv`, `anomaly_eval_audit_summary.csv`)

| Deney | block_f1 kış | block_f1 yaz | block_f1 ort. | ek metrik ort.* |
|---|---|---|---|---|
| S0 (A3-s2s-hmean) | 0,551 | 0,535 | 0,543 | 0,462 |
| S1 (+spike) | 0,551 | 0,535 | 0,543 | 0,462 |
| S2 (+drop) | 0,553 | 0,542 | 0,548 | 0,471 |
| S3 (+değişim) | 0,521 | 0,569 | 0,545 | 0,470 |
| S4 (+spike+drop) | 0,553 | 0,542 | 0,548 | 0,471 |
| S5 (hepsi) | 0,529 | 0,598 | 0,563 | 0,497 |

\* Ek metrik, olay sayımlı F1'dir (tanım ve sınırları: `docs/inceleme_ve_iyilestirme_plani_2026-11-08.md` §1). Birebir alarm–olay eşleştirmesi değildir.

- Önceden konan karar kuralına göre hiçbir aday kabul edilmedi; S0 korundu.
- S5'in ortalama kazancı gürültü sınırında. Yazda arttı, kışta düştü (kış −0,022).
- **Yaz-dev ağırlıkları** bu makinede yeniden eğitildi (hash'ler: `repro_manifest.csv`). Yaz sayıları Şeyda'nın Aşama 2 tablolarıyla doğrudan karşılaştırılamaz.

## 3. Testler (bu makine: macOS, Python 3.11.18, pandas 3.1.0)

| Durum | Sayı | Açıklama |
|---|---|---|
| Geçen | 205 | `pytest tests --ignore=tests/test_stage3_models.py` |
| Atlanan | 7 | Hepsi isteğe bağlı Şebeke (ELD) verisi/ağırlıkları yok (`test_demo_doc_consistency.py`); gerekçe test çıktısında |
| Geçen (ayrı koşu) | 7 | `tests/test_stage3_models.py -k "not xgb and not xgboost"` |
| Çöken | 2 | `test_stage3_models.py` içindeki XGBoost testleri macOS'ta segmentation fault veriyor (ortam sorunu; kod düzeltmesi yapılmadı) |

## 4. Bu dalda düzeltilenler (skora etkisi ölçüldü)

1. pandas 3 zaman birimi, tahmin kayıt/okuma yolunda → kayıtlı tahmin tabloları değişmiyor (val 4213 pencere / 177 gün, test 5481 / 229).
2. Aynı hata ön işlemenin sınır korumasında (±3 saat yerine ±125 gün) → doldurulan değerler değişmiyor (0 saat).
3. Olay metriğine ek tanım (mevcut `block_f1` korunuyor).
4. Etiketsiz arka plan alarmları ayrı raporlanıyor.
5. Yeniden üretim manifestosu (`repro_manifest.csv`).
6. ELD'ye bağlı demo testi düzeltildi.
7. README ve CLAUDE.md yöntem adları kodla uyumlu hâle getirildi.
