# v2: A3-s2s-norm + fiziksel taban kanalı — test kümesinde tek koşu (TEST SONRASI, final değil)

Komut: `python -m src.anomaly_v2_floor_test --device cpu`

## 1. Protokol (koşudan önce yazıldı, 2026-11-09; sonuç görüldükten sonra değiştirilmedi)

**Bağlam.**
- Resmî final test sonucu (A3-s2s-norm, olay F1 = 0,384 ± 0,170) 2026-10-31'de alındı ve **değişmez**.
- Test kümesi o koşuda görüldü. Bu yüzden v2 bağımsız bir test sonucu değildir.
- Taban kanalı test sonrasında, geliştirme verisinde tasarlandı (`docs/anomaly_floor_channel.md`).
- Bu koşu, hocanın açık onayıyla (2026-11-09) bir kez yapılır.

**v2 tanımı.** Bütün parametreler önceden sabittir, testte hiçbir şey seçilmez.
- Kısa kanal: A3-s2s-norm. Pencere w = 3, eşik T = 3,5580; ikisi de `anomaly_protocol_selection.csv` dosyasından, val-dev'de seçilmiş değerler.
- Taban kanalı: f_t = (y_t < taban) VE (y_{t−1} < taban).
  - Taban = train'in (2006-12-16 – 2009-09-30) doldurulmamış saatlerinin 0,5. yüzdeliği.
  - Yalnız geçmişe bakar.
- Alarm = kısa kanal VEYA taban kanalı. Eşdeğer skor: r = max(s̃ / T, 2·f), alarm r ≥ 1 olduğunda.
- Değerlendirilen saatler, resmî koşudakilerle birebir aynıdır (skoru olan ve doldurulmamış saatler).

**Veri.** Resmî koşuyla aynı:
- Test kümesi (2010-04-01 – 2010-11-26).
- Ana senaryo seed'leri 2, 12, 22, 32, 42; zor senaryo seed'leri 4, 14, 24, 34, 44.
- Aynı model ağırlıkları ve aynı OOF std'leri.

**Yeniden üretim denetimi.** Aynı kopyalarda A3-s2s-norm tek başına da hesaplanır. Ana senaryo olay F1'i resmî değerden (0,3835) 0,005'ten fazla saparsa sonuç "yeniden üretilemedi" notuyla raporlanır.

**Raporlanacaklar.**
- Ana ve zor senaryo için: olay F1, olay recall, olay precision ve tür başına recall.
- v2 ile A3-s2s-norm arasında eşleştirilmiş fark.
- Enjeksiyonsuz testte taban kanalının alarm bloğu sayısı.

**Önceden kabul edilen sınırlar.**
- Drop kazancı, enjekte değerin (0,05 kW) tabanın altında olmasına bağlıdır.
- Bu sonuç H3'ün karşılanıp karşılanmadığı değerlendirmesini değiştirmez; raporda ayrı bir satır olarak "v2 (test sonrası)" etiketiyle verilir.
- Final yöntem A3-s2s-norm olarak kalır.

## 2. Sonuçlar (koşu 2026-11-09; log `results/logs/anomaly_v2_floor_test.log`)

- **Yeniden üretim:** Aynı kopyalarda A3-s2s-norm ana senaryo olay F1'i 0,3835. Resmî değerle birebir aynı (TAMAM).
- **Taban:** 0,155 kW. Enjeksiyonsuz test kümesinde taban kanalı **0** alarm bloğu verdi, yani gerçek veride yanlış alarm eklemedi.

Kaynak: `anomaly_v2_floor_test_summary.csv`, `anomaly_v2_floor_test_paired.csv` (5 seed, ort ± std).

| Senaryo | Yöntem | Olay F1 | Olay recall | Olay precision | Drop recall |
|---|---|---|---|---|---|
| Ana | A3-s2s-norm (resmî final) | 0,384 ± 0,170 | 0,358 | 0,424 | 0,10 |
| Ana | **v2: A3 + taban** | **0,506 ± 0,135** | 0,510 | 0,507 | 1,00 |
| Zor | A3-s2s-norm (resmî final) | 0,318 ± 0,073 | 0,286 | 0,364 | 0,00 |
| Zor | **v2: A3 + taban** | **0,460 ± 0,059** | 0,463 | 0,462 | 1,00 |

**Eşleştirilmiş fark (v2 − A3-s2s-norm, olay F1):**
- Ana senaryo: +0,123 ± 0,072. 5 seed'in 5'inde artış var.
- Zor senaryo: +0,141 ± 0,056. 5 seed'in 5'inde artış var.

Diğer türlerin recall değerleri (spike, seviye kayması, desen) iki yöntemde aynı. Kazancın tamamı drop olaylarından geliyor.

**Yorum.**
- Taban kanalı test kümesinde de, geliştirme verisinde olduğu gibi, enjekte edilen bütün kesintileri yakaladı ve gerçek veride yanlış alarm üretmedi.
- Kazanç, enjekte değerin (0,05 kW) tabanın (0,155 kW) altında olmasına bağlı. Gerçek kesintiler bu kadar derin olmayabilir.
- Test kümesi daha önce görüldüğü için bu sonuç bağımsız değildir. Raporda "v2 (test sonrası)" etiketiyle ayrı satır olarak verilir. Final yöntem A3-s2s-norm olarak kalır.
- **H3 (olay F1 ≥ 0,70) v2 ile de karşılanmadı.** Testte spike (0,27) ve desen (0,18) recall değerleri düşük kaldı.
