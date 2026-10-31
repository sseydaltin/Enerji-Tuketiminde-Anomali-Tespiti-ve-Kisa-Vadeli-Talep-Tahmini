# Tahmin deneyleri özeti (Aşama 1–5, yalnız doğrulama kümesi)

> Bu belgedeki tüm sayılar `results/` dosyalarından üretildi (`forecast_val_results.csv`,
> `seed_stability_val*.csv`, `optuna_*_best.csv`, `F13-ens_weight_grid_val.csv`).
> **Test kümesi hiçbir aşamada kullanılmadı.** Final değerlendirme (Aşama 6) henüz yapılmadı.
> Tarih: 2026-10-31 · Dal: `seyda/model-fixes`.

## 1. Değerlendirme protokolü (tüm modeller için aynı)
- Val: 2009-10-01 → 2010-03-31, 4.213 pencere (doldurulmuş hedefli 132 pencere çıkarıldı); 168/48/336 → 24 saat.
- Tüm modeller aynı `calculate_metrics` ile, tahminler kW'a geri çevrildikten sonra ölçülür; `verify_forecast_eval` tüm kayıtlı tahminlerin aynı pencere ve aynı kW hedeflerini kullandığını doğrular.
- Model seçimi ve early stopping: val MAE (kW). Optuna (Aşama 4) seçimi val'de değil, train'in son 12 ayında 3 katlı zamansal CV ile yaptı.

## 2. Ana val tablosu (seçilmiş koşular, tek seed = 42)
`vs_SN168_%`: Seasonal Naive 168'e göre MAE iyileşmesi. Süre: eğitim süresi (s).

| exp_id | model | feature_set | lookback | MAE | RMSE | sMAPE | R2 | train_time_s | device | vs_SN168_% |
|---|---|---|---|---|---|---|---|---|---|---|
| F0 | Persistence | A | 168 | 0,9598 | 1,2537 | 74,38 | -0,888 | 0,0 | cpu | -54,4 |
| F1 | SeasonalNaive24 | A | 168 | 0,6913 | 0,9898 | 52,96 | -0,177 | 0,0 | cpu | -11,2 |
| F2 | SeasonalNaive168 | A | 168 | 0,6216 | 0,9053 | 48,32 | 0,016 | 0,0 | cpu | 0,0 |
| F3-a0.1 | Ridge | A | 168 | 0,6053 | 0,8054 | 52,28 | 0,221 | 0,0 | cpu | 2,6 |
| F3-cal | Ridge | B | 168 | 0,5819 | 0,7608 | 49,95 | 0,305 | 0,0 | cpu | 6,4 |
| F4-torch-128x64 | MLP | B | 168 | 0,5045 | 0,7080 | 42,91 | 0,398 | 23,6 | cuda | 18,8 |
| F4-torch-128x64-fut | MLP | B | 168 | 0,5019 | 0,6859 | 42,86 | 0,435 | 45,7 | cuda | 19,3 |
| F5 | LSTM | B | 168 | 0,4893 | 0,6801 | 41,57 | 0,445 | 37,8 | cuda | 21,3 |
| F6 | GRU | B | 168 | 0,4912 | 0,6869 | 41,68 | 0,433 | 49,0 | cuda | 21,0 |
| F6-C | GRU | C | 168 | 0,4878 | 0,6818 | 41,50 | 0,442 | 38,4 | cuda | 21,5 |
| F5-L336 | LSTM | B | 336 | 0,4844 | 0,6785 | 40,94 | 0,447 | 68,9 | cuda | 22,1 |
| F11-L48 | CNN-LSTM | B | 48 | 0,4838 | 0,6818 | 40,73 | 0,442 | 41,2 | cuda | 22,2 |
| F12-L336 | XGBoost | B | 336 | 0,4998 | 0,6806 | 42,65 | 0,444 | 17,4 | cpu | 19,6 |
| F10-L48 | Seq2Seq-LSTM | B | 48 | 0,4747 | 0,6698 | 40,12 | 0,461 | 68,7 | cuda | 23,6 |
| F5-opt | LSTM | B | 48 | 0,4826 | 0,6815 | 40,76 | 0,442 | 47,5 | cuda | 22,4 |
| F10-opt | Seq2Seq-LSTM | B | 48 | 0,4627 | 0,6839 | 37,20 | 0,438 | 16,4 | cuda | 25,6 |
| F13-ens | Ensemble | - | - | 0,4599 | 0,6753 | 37,27 | 0,452 | 0,0 | cpu | 26,0 |

Not: F13-ens'in ağırlıkları val'de seçildiği için val skoru iyimserdir.

## 3. Seed kararlılığı (F9; seed 42, 7, 2026; örneklem std, ddof=1)
| config | model | n | MAE (ort ± std) | RMSE (ort ± std) | sMAPE (ort ± std) | R² (ort ± std) |
|---|---|---|---|---|---|---|
| F10-opt | Seq2Seq-LSTM | 3 | 0,4688 ± 0,0060 | 0,6907 ± 0,0072 | 37,77 ± 0,51 | 0,427 ± 0,012 |
| F5-opt | LSTM | 3 | 0,4810 ± 0,0016 | 0,6822 ± 0,0006 | 40,48 ± 0,33 | 0,441 ± 0,001 |
| F12-L336 | XGBoost | 3 | 0,4995 ± 0,0002 | 0,6806 ± 0,0001 | 42,59 ± 0,06 | 0,444 ± 0,000 |
| F13-ens | Ensemble | 3 | 0,4625 ± 0,0027 | 0,6785 ± 0,0028 | 37,44 ± 0,19 | 0,447 ± 0,005 |

Seed başına val MAE:

| config | seed 7 | seed 42 | seed 2026 |
|---|---|---|---|
| F10-opt | 0,4747 | 0,4627 | 0,4688 |
| F12-L336 | 0,4993 | 0,4998 | 0,4994 |
| F13-ens | 0,4652 | 0,4599 | 0,4626 |
| F5-opt | 0,4810 | 0,4826 | 0,4794 |

**Önemli:** Seq2seq'in seed std'si (0,0060 kW), Aşama 2–3'te görülen birçok farktan büyük (LSTM–GRU 0,002, lookback 48/168/336 ≤ 0,005, CNN-LSTM–LSTM 0,0006). Bu farklar tek seed ile anlamlı sayılmamalı. Seq2seq ile LSTM arasındaki fark (0,4688 vs 0,4810, ≈ 0,012 kW) seed std'lerinin birkaç katı → muhtemelen gerçek.

## 4. Optuna ile seçilen hiperparametreler (Aşama 4, lookback 48)
| model | en iyi deneme | CV MAE | deneme (tamam/kesildi) | hidden | katman | lr | batch | dropout | weight decay | kayıp | shuffle | eğitim süresi (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| seq2seq | 16 | 0,4410 | 20 (14/6) | 128 | 1 | 1.78e-03 | 64 | 0,355 | 5.7e-05 | l1 | True | 1528 |
| lstm | 18 | 0,4568 | 20 (17/3) | 128 | 2 | 6.48e-04 | 64 | 0,341 | 5.2e-05 | huber | False | 1541 |

Arama uzayı: hidden {32, 64, 128}, katman {1, 2, 3}, lr [1e-4, 3e-3] (log), batch {32, 64, 128}, dropout [0, 0,4], weight decay [1e-6, 1e-3] (log), kayıp {MSE, L1, Huber}, shuffle {açık, kapalı}. TPE (seed 42) + MedianPruner. Tam config: `configs/optuna_seq2seq_best.yaml`, `configs/optuna_lstm_best.yaml`.

## 5. Topluluk (Aşama 5)
Üyeler: F10-opt (seq2seq), F5-opt (LSTM), F12-L336 (XGBoost). Ağırlıklar {0, 0,1, …, 1} ızgarasında (66 kombinasyon) val MAE ile seçildi; seed 7 ve 2026'da aynı ağırlıklar kullanıldı. En iyi 5 kombinasyon (seed 42):

| weights | MAE | RMSE |
|---|---|---|
| [0.8, 0.2, 0.0] | 0,4599 | 0,6753 |
| [0.7, 0.3, 0.0] | 0,4600 | 0,6725 |
| [0.7, 0.2, 0.1] | 0,4603 | 0,6701 |
| [0.8, 0.1, 0.1] | 0,4603 | 0,6732 |
| [0.9, 0.1, 0.0] | 0,4607 | 0,6791 |


XGBoost'un ağırlığı 0 çıktı. Topluluk, seq2seq'e göre ortalamada yaklaşık 0,006 kW iyi ve daha kararlı (std 0,0027 vs 0,0060), ama ağırlıklar val'de seçildiği için bu kazanç iyimserdir.

## 6. Denenen konfigürasyon sayısı ve hesaplama maliyeti
- `results/experiments.csv`: **92 satır** = 40 Optuna CV denemesi (gerçek val'i görmez) + 50 val koşusu (8'i seed tekrarı, 3'ü topluluk) + 2 tarihsel satır (ilk F3, sklearn F4).
- Gerçek val kümesinde değerlendirilen farklı konfigürasyon: **42** (50 − 8 seed tekrarı).
- GPU (RTX 2050) eğitim süresi: Aşama 1 ≈ 156 s, Aşama 2 ≈ 516 s, Aşama 3 ≈ 561 s, Aşama 4 ≈ 3.069 s (Optuna) + 64 s (val doğrulaması), Aşama 5 ≈ 166 s → toplam ≈ 4.532 s (≈ 76 dk). 3 saatlik kota aşılmadı; deneme sayısı düşürülmedi. XGBoost CPU'da (koşu başına ≈ 17 s).

## 7. Aşama bazında ana bulgular
1. **Aşama 1:** PR #4'teki LSTM/GRU metrikleri z-skoru üzerinden hesaplanıyordu (Olay 1); düzeltildi. PyTorch MLP sklearn MLP'den iyi (0,5684 → 0,5045). Ridge alpha etkisiz.
2. **Aşama 2:** Shuffle üç modelde kötüleştirdi; Ridge + takvim 0,6053 → 0,5819; gelecek öznitelikleri yalnız MLP'de, C seti yalnız GRU'da küçük kazanç; artık hedef LSTM/GRU'da öğrenilemedi (tanılama: taban değer çıktıya verilince öğreniliyor).
3. **Aşama 3:** Seq2seq LSTM en iyi mimari (0,4747); kazanç uzun ufuklarda. Lookback etkisi gürültü düzeyinde. XGBoost h=1'de ve tepe tüketimde güçlü.
4. **Aşama 4:** Optuna seq2seq'i 0,4747 → 0,4627 (L1 kaybı, shuffle açık), LSTM'i 0,4859 → 0,4826 (Huber, shuffle kapalı) getirdi. Shuffle × lr hipotezinin "lr fazla yüksek" kısmı desteklenmedi; shuffle'ın etkisi mimariye bağlı.
5. **Aşama 5:** 3 seed: seq2seq 0,4688 ± 0,0060, LSTM 0,4810 ± 0,0016, XGBoost 0,4995 ± 0,0002, topluluk 0,4625 ± 0,0027. Seed 42 seq2seq sonucu üç seed'in en iyisiydi.

## 8. Sınırlılıklar
- Çok sayıda konfigürasyon aynı val kümesinde karşılaştırıldı; seçilen modelin val skoru iyimserdir. Yansız ölçüm yalnız test'te (Aşama 6) yapılabilir.
- Val kış (yüksek tüketim), test ilkbahar–sonbahar; val–test farkı beklenebilir.
- Optuna CV katlarında preprocess interpolasyonu kat sınırlarını bilmez (≤ 3 saat; doldurulmuş hedefler metrikten çıkarılıyor).
- Optuna seçimleri tek seed (42) ile yapıldı.

## 9. Savunmada sorulabilecekler
1. **Neden Optuna'yı val'de değil train içi CV'de çalıştırdınız?** Aynı val kümesinde yüzlerce ayar denenseydi val skoru seçime göre şişerdi (val'e aşırı uyum). Train'in son 12 ayını 3 ardışık kata bölüp her katta yalnız geçmişle eğittik; val'i yalnız seçilen ayarı bir kez doğrulamak için kullandık.
2. **Seq2seq neden L1 kaybıyla en iyi oldu, RMSE neden kötüleşti?** L1 koşullu medyanı, MSE koşullu ortalamayı hedefler. Seçim metriğimiz MAE olduğu için L1 ile uyumlu; ama büyük hataları daha az cezaladığı için RMSE ve R² biraz kötüleşti. Hangi metriğin önemli olduğu uygulamaya bağlı (tepe talep için RMSE).
3. **Tek seed'deki farklara neden güvenmiyorsunuz?** Seq2seq'in seed std'si 0,006 kW; birçok ablasyon farkı bundan küçük. Bu yüzden yalnız seed std'sinin birkaç katı olan farkları (ör. seq2seq vs LSTM) gerçek kabul ediyoruz.
4. **Topluluk gerçekten daha iyi mi?** Ortalamada ≈ 0,006 kW iyi ve daha kararlı; ama ağırlıklar val'de seçildiği için bu iyimser. XGBoost'un ağırlığı 0 çıktı; seq2seq ve LSTM'in hataları kısmen birbirini götürüyor.
5. **MedianPruner neyi kesti, adil mi?** İlk 5 denemeden sonra, bir denemenin o ana kadarki kat ortalaması aynı adımdaki tamamlanmış denemelerin medyanından kötüyse kesildi (seq2seq 6, LSTM 3 deneme). Kesilen denemelerin kesildikleri adımdaki ortalama MAE'si 0,51–0,60 idi. İlk kat kış dönemi olduğu için tüm denemelerde yüksek çıkar; karşılaştırma aynı adımdaki denemeler arasında yapıldığı için bu adil, ama yanlış kesme riski sıfır değil. Süre kazancı büyük.
6. **Shuffle hipoteziniz doğrulandı mı?** Kısmen: LSTM'de karıştırma lr'den bağımsız kötü, bu "yakın geçmişe örtük ağırlık" ile uyumlu; seq2seq'te ise en iyi denemeler karıştırmalı ve yüksek lr ile geldi. TPE dengeli örneklemediği için nedensel sonuç çıkarmıyoruz.

## 10. Sonraki adım (yapılmadı)
Aşama 6: seçilen konfigürasyonu train + val üzerinde val'de bulunan epoch sayısıyla yeniden eğiten ve test'i bir kez değerlendiren betik. Kullanıcı onayı olmadan hazırlanmadı/çalıştırılmadı.

## Aşama 3b – Hava durumu öznitelikleri (Open-Meteo ERA5)

Val, 3 seed (42, 7, 2026) ortalama ± std (ddof=1). Kaynak: seed_stability_val.csv. W2 satırı ORACLE'dır: hedef saatlerin gerçek havası kullanıldı, gerçekte bilinemez.

| Model | Set | Rol | MAE (kW) | RMSE (kW) | sMAPE (%) | R² |
|---|---|---|---|---|---|---|
| Seq2Seq-LSTM (F10-opt) | B | referans | 0.4688 ± 0.0060 | 0.6907 ± 0.0072 | 37.77 ± 0.51 | 0.427 ± 0.012 |
| Seq2Seq-LSTM (F10-opt) | W1 | ANA: geçmiş hava | 0.4732 ± 0.0068 | 0.6923 ± 0.0097 | 38.23 ± 0.53 | 0.424 ± 0.016 |
| Seq2Seq-LSTM (F10-opt) | W2 | ORACLE / üst sınır – gerçekte bilinemez, seçimde kullanılmaz | 0.4833 ± 0.0093 | 0.7069 ± 0.0161 | 39.24 ± 1.12 | 0.399 ± 0.027 |
| LSTM (F5-opt) | B | referans | 0.4810 ± 0.0016 | 0.6822 ± 0.0006 | 40.48 ± 0.33 | 0.441 ± 0.001 |
| LSTM (F5-opt) | W1 | ANA: geçmiş hava | 0.4884 ± 0.0025 | 0.6874 ± 0.0024 | 41.13 ± 0.26 | 0.432 ± 0.004 |

- W1 (geçmiş hava, yalnız encoder) iki modelde de B'yi geçmedi: seq2seq farkı seed std'si içinde, LSTM'de hafif zarar.
- W2 **oracle / üst sınırdır** (hedef saatlerin gerçek havası; gerçekte bilinemez) ve model seçiminde kullanılmaz; bu ayarlarla aşırı öğrendi (en iyi epoch 1–2).
- Topluluk yeniden kurulmadı (koşul: W1 en iyi modeli geçerse). Ayrıntı ve teşhis: `docs/deney_gunlugu.md`.
