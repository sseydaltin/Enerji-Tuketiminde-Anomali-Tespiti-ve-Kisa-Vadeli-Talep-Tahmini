# Model ve Sonuç Özeti

**YMÜ425 – Uygulamalı Sinir Ağları** · Enerji Tüketiminde Anomali Tespiti ve Kısa Vadeli Talep Tahmini
Dilara YILDIRIM · Şeyda ALTIN · 2026-11-09

Bu özetteki bütün sayılar `results/tables/` altındaki dosyalardan alınmıştır (her tablonun altında kaynak var).
"Olay F1" bir doğruluk yüzdesi değildir: en az bir saati işaretlenen olayları sayan F1 ölçüsüdür.

---

## 1. Final test sonuçları

Test kümesi (hane, 2010-04-01 – 2010-11-26) bir kez kullanıldı. Modeller ve eşikler testten önce seçildi.

### Talep tahmini (24 saat ufuk)

| Model | Test MAE (kW) | Test RMSE (kW) | Test R² | SN168'e göre MAE farkı |
|---|---|---|---|---|
| Seasonal Naive 168 (SN168) | 0,549 | 0,795 | −0,244 | 0 |
| Seasonal Naive 24 | 0,512 | 0,760 | −0,137 | %6,8 daha düşük |
| Ridge (takvimli) | 0,467 | 0,612 | 0,262 | %15,0 daha düşük |
| LSTM (F5-opt, 3 seed) | 0,413 ± 0,005 | 0,581 | 0,336 | %24,8 daha düşük |
| GRU (F6-C) | 0,417 | 0,581 | 0,335 | %24,1 daha düşük |
| Seq2Seq-LSTM (F10-opt, 3 seed) | 0,414 ± 0,007 | 0,612 | 0,263 | %24,7 daha düşük |
| XGBoost (F12-L336) | 0,416 | **0,568** | **0,364** | %24,2 daha düşük |
| **F13-ens (final; 0,8 Seq2Seq + 0,2 LSTM, 3 seed)** | **0,406 ± 0,005** | 0,598 | 0,297 | **%26,0 daha düşük** |

Kaynak: `final_forecast_val_vs_test.csv`. H2 hedefi (en iyi model RMSE'si SN168'den ≥ %10 düşük) karşılandı.

### Anomali tespiti (sentetik enjeksiyonlu test, ana senaryo, 5 seed)

| Yöntem | Olay F1 | Nokta F1 | PR-AUC |
|---|---|---|---|
| A0 – hareketli z-skoru | 0,398 ± 0,064 | 0,140 | 0,148 |
| A1 – Isolation Forest | 0,378 ± 0,144 | 0,210 | 0,316 |
| A2 – LSTM-Autoencoder | 0,352 ± 0,099 | 0,310 | 0,329 |
| **A3-s2s-norm (final; tahmin artığı)** | **0,384 ± 0,170** | 0,147 | 0,241 |

Kaynak: `final_anomaly_val_vs_test.csv`. **H3 hedefi (F1 ≥ 0,70) karşılanmadı.** Final yöntem testte A0'dan ayırt edilemedi.

---

## 2. Test sonrası geliştirme sonuçları (final değil)

- **Veri:** kış-dev (2009-10 – 12) ve yaz-dev (2009-06 – 09), orta senaryo, 10'ar enjeksiyon seed'i.
- **Kapsam dışı:** test ve val-hold kullanılmadı.
- **Bağımsızlık:** Bu dönemler daha önce defalarca görüldüğü için sonuçlar bağımsız doğrulanmış değildir.
- **Kıyaslanabilirlik:** Test sonuçlarıyla doğrudan kıyaslanamaz.
- **Referans:** S0 = A3-s2s-hmean (aynı kopyalar, aynı ağırlıklar).

| Deneme | Olay F1 kış / yaz | Karar |
|---|---|---|
| S0 (referans) | 0,551 / 0,535 | — |
| Aşama 3: spike / drop / değişim sinyalleri (en iyisi S5) | 0,529 / 0,598 | Kabul edilmedi |
| E1: 6 saatlik işaretli artık | 0,575 / 0,529 | Kabul edilmedi |
| C1: medyan/MAD profil sapması | 0,475 / 0,513 | Kabul edilmedi |
| C2: C1 + A3 birleşimi | 0,478 / 0,546 | Kabul edilmedi |
| OR kanalları (A3 veya C1; en iyisi OR-B) | 0,510 / 0,614 | Kabul edilmedi |
| S0 + taban kuralı | 0,644 / 0,604 | Yalnız deneysel; kazanç taban altındaki enjekte kesintilerden geliyor |
| Gözetimli LR | 0,562 / 0,533 | Kabul edilmedi |
| Gözetimli RF | 0,571 / 0,259 | Kabul edilmedi |
| **Gözetimli HGB** | **0,668 / 0,588** (ortalama 0,628) | Kabul edilmedi (kış spike recall'ü 0,90 → 0,77) |

Kaynaklar: `anomaly_short_event_summary.csv`, `E1_sustained_summary.csv`, `anomaly_candidates_summary.csv`, `anomaly_or_channels_summary.csv`, `anomaly_floor_medium_summary.csv`, `anomaly_supervised_summary.csv`.

Gözetimli modellerde eğitim, değerlendirme dönemlerinden önceki bir dönemde sentetik anomalilerle yapıldı. **0,628 bir geliştirme sonucudur, bağımsız test sonucu değildir.**

---

## 3. Son karar

- **Final tahmin modeli:** F13-ens.
- **Final anomali yöntemi:** A3-s2s-norm.
- **H3 (F1 ≥ 0,70):** karşılanmadı.
- **Geliştirmeler neden final modele alınmadı:**
  - Önceden yazılan kabul koşullarını karşılamadılar (bir mevsimde kayıp veya spike recall kaybı).
  - Kabul edilen tek ek olan taban kuralının kazancı, sentetik kesinti değerinin veride hiç görülmemiş bir düzeyde olmasından kaynaklanıyor.
  - Veri setinde kullanılmamış bir zaman bölümü kalmadığı için hiçbir geliştirme bağımsız olarak doğrulanamadı.

---

## 4. Hocaya gönderilecek kısa metin

> Hocam merhaba, projemizin güncel durumunu kısaca paylaşmak istiyorum. Talep tahmininde final modelimiz F13-ens (0,8 Seq2Seq-LSTM + 0,2 LSTM) oldu. Tek seferlik test kümesinde ortalama mutlak hatası 0,406 kW; bu, mevsimsel naif (bir hafta önceki aynı saat) yönteme göre %26 daha düşük hata demek ve H2 hedefimizi karşılıyor. Anomali tespitinde gerçek etiket olmadığı için değerlendirmeyi sentetik enjeksiyonla yaptık. Final yöntemimiz olan tahmin artığı dedektörü A3-s2s-norm, testte olay F1 = 0,384 elde etti ve basit z-skoru yönteminden (0,398) ayırt edilemedi; bu nedenle H3 hedefi (F1 ≥ 0,70) karşılanmadı. Test sonrasında yalnız geliştirme verisinde yeni sinyaller, sağlam istatistiksel profil, kanal birleşimleri ve sentetik anomalilerle eğitilen gözetimli sınıflandırıcılar denedik. En iyi gözetimli model (HGB) geliştirme verisinde ortalama olay F1 = 0,628'e ulaştı; ancak bu bağımsız bir test sonucu değil ve önceden belirlediğimiz kabul koşulunu karşılamadığı için final modele alınmadı. Veri setinin bütün dönemleri daha önce kullanıldığından bu geliştirmeleri bağımsız olarak doğrulayabileceğimiz bir bölüm kalmadı; bunu ve tek hane verisinin yüksek oynaklığını temel sınırlılıklar olarak raporluyoruz.
