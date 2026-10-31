# Hane ve şebeke: aynı yöntemler, iki toplulaşma düzeyi (TEST)

- **Hane:** UCI Individual Household Electric Power Consumption, tek ev, ~1 kW. Test dönemi 2010-04-01 – 2010-11-26 (`docs/final_sonuclar.md`).
- **Şebeke (ELD):** UCI ElectricityLoadDiagrams20112014, 2012 öncesi aktif 160 müşterinin toplamı, ~106 MW. Test dönemi 2014-07-01 – 2014-12-31 (`docs/eld_protokol.md`).

ELD'de aynı model ve anomali ayarları household'dan **aynen** alındı; yeni arama yapılmadı, modeller yalnız ELD train'inde yeniden eğitildi. İki veri setinde de test **bir kez** çalıştırıldı ve test sonucuna göre hiçbir şey değiştirilmedi.

Tablolar `results/eld/tables/hane_vs_sebeke_{forecast,anomaly}.csv` dosyalarında (`src/compare_hane_sebeke.py`). Tarih: 2026-10-31.

**Okuma notları:**
- MAE iki ölçek arasında karşılaştırılamaz. Bu yüzden MAPE*, R² ve **aynı veri ve dönemdeki** Seasonal Naive 168'e göre MAE kazancı kullanılır.
- Household'da MAPE* yalnız y > 0,1 kW saatlerde hesaplanır; ELD'de bu eşik etkisizdir.
- F5, F10 ve F13 için değerler 3 seed ortalamasıdır.

## 1. Tahmin, 24 saat ufuk (ana metrik)

| Model | Hane MAPE* | Şebeke MAPE* | Hane R² | Şebeke R² | Hane: SN168'e göre kazanç | Şebeke: SN168'e göre kazanç |
|---|---|---|---|---|---|---|
| F1 Seasonal Naive 24 | 66,3 % | 3,70 % | −0,137 | 0,943 | +6,8 % | +25,0 % |
| F2 Seasonal Naive 168 | 75,2 % | 4,67 % | −0,244 | 0,933 | 0 | 0 |
| F3-cal Ridge | 69,1 % | 3,47 % | 0,262 | 0,960 | +15,0 % | +31,3 % |
| F12-L336 XGBoost | 59,9 % | **2,76 %** | **0,364** | 0,973 | +24,2 % | **+45,1 %** |
| F5-opt LSTM | 53,8 % | 3,48 % | 0,336 | 0,970 | +24,8 % | +31,3 % |
| F10-opt Seq2Seq | 49,9 % | 3,32 % | 0,263 | 0,973 | +24,7 % | +33,4 % |
| **F13-ens (final)** | **49,4 %** | 3,24 % | 0,297 | **0,974** | **+26,0 %** | +35,4 % |

**Diğer seviyeler** (final model F13-ens / en iyi model):

| Seviye | Hane MAPE* | Şebeke MAPE* | Hane R² | Şebeke R² | Hane: SN168'e göre kazanç | Şebeke: SN168'e göre kazanç |
|---|---|---|---|---|---|---|
| (a) h=1, F13-ens | 40,7 % | 2,24 % | 0,527 | 0,990 | +38,7 % | +55,6 % |
| (a) h=1, XGBoost | 42,5 % | **1,65 %** | **0,578** | **0,994** | +41,5 % | **+66,5 %** |
| (c) günlük toplam, F13-ens | 18,9 % | **1,91 %** | 0,330 | **0,917** | +20,4 % | +54,4 % |
| (c) günlük toplam, XGBoost | **17,1 %** | 2,25 % | **0,506** | 0,843 | **+34,1 %** | +50,1 % |

## 2. Anomali tespiti (ana senaryo, σ tabanlı enjeksiyon, test 5 seed)

Pencere ve olay-F1 eşiği her veri setinin kendi val'inde seçildi: hane'de w = 3, şebekede w = 1.

| Yöntem | Hane olay F1 | Şebeke olay F1 | Hane PR-AUC | Şebeke PR-AUC | Hane olay R / P | Şebeke olay R / P |
|---|---|---|---|---|---|---|
| A0 z-skoru | 0,398 ± 0,064 | 0,860 ± 0,017 | 0,148 | 0,761 | 0,43 / 0,39 | 0,81 / 0,92 |
| **A3-s2s-norm (final)** | 0,384 ± 0,170 | **0,927 ± 0,012** | 0,241 | 0,978 | 0,36 / 0,42 | **1,00** / 0,86 |
| A5-mean-z | **0,463 ± 0,048** | 0,908 ± 0,024 | 0,213 | **0,993** | 0,58 / 0,39 | 0,96 / 0,87 |

Şebekede A3-s2s-norm ile A0 arasındaki eşleştirilmiş olay F1 farkı +0,067 ± 0,023 (5 seed'in 5'inde). Hanede aynı fark −0,015 ± 0,184 (5'in 2'sinde).

## 3. Yorum: toplulaştırmanın öngörülebilirliğe etkisi

1. **Toplulaştırma tahmin hatasını bir büyüklük mertebesi azaltıyor.** 24 saatlik MAPE* hanede ~%50, şebekede ~%3; R² 0,30'dan 0,97'ye çıkıyor.
   - Olası neden: 160 müşterinin rastgele ve bireysel dalgalanmaları (cihazın açılıp kapanması, evde olmamak) toplamda büyük ölçüde birbirini götürüyor; geriye düzenli günlük ve haftalık desen kalıyor.
   - Bu, toplamın daha az "gürültülü" olduğu sezgisiyle uyumlu, ama burada yalnız iki veri setiyle gösterildi. Müşteri sayısına göre bir eğri çıkarılmadı.
2. **Naif modeller de iyileşiyor; öğrenen modellerin naife göre kazancı ise büyüyor.** SN168'e göre kazanç hanede +26 %, şebekede +35 % (F13-ens) ve +45 % (XGBoost). Toplamda yapı daha öğrenilebilir; hanede hatanın büyük kısmı hiçbir modelin göremediği rastgelelikten geliyor.
3. **Model sıralaması veri setine göre değişiyor.**
   - Hanede 24 saat MAE'de F13-ens birinci.
   - Şebekede XGBoost (336 saatlik gecikme + takvim) belirgin biçimde birinci; F13-ens ikinci (+35,4 %, XGBoost +45,1 %); seq2seq, LSTM ve Ridge birbirine yakın (+31–33 %).
   - Topluluk ağırlıkları ve derin modellerin ayarları household'da seçildi ve burada değiştirilmedi. ELD-F5-opt üç seed'de de 60 epoch sınırına ulaştı (yakınsamadı).
   - Yani şebekedeki sıralama "household'a göre ayarlanmış derin modeller" ile "ayarsız ağaç modeli" karşılaştırmasıdır. Derin modellerin şebekede daha iyi olamayacağı anlamına gelmez.
4. **Anomali tespiti toplamda çok daha kolay.**
   - Aynı σ tabanlı anomaliler düzgün bir seride belirgin kalıyor: olay F1 0,86–0,93, PR-AUC 0,76–0,99.
   - Hanede ise doğal oynaklık anomalileri gizliyor: olay F1 0,38–0,46.
   - Drop tanımı mutlak (0,05 kW) olduğu için şebekede tam kesinti anlamına geliyor ve hep yakalanıyor; bu, şebeke skorunu bir miktar şişirir.
   - Tahmin artığı yaklaşımı (A3) şebekede A0'dan açıkça iyi. Hanede iyi tahmin zaten zor olduğu için artığın sinyal/gürültü oranı düşük.
5. **Şebeke testinde val'e göre bozulma var ve nedeni büyük ölçüde tatiller.**
   - F13-ens 24 saatlik MAPE*'si val'de 2,18 %, testte 3,24 %. Bozulmanın büyük kısmı Aralık'tan geliyor: Aralık'ta 5,3 %; Temmuz–Kasım'da 2,5–2,7 %.
   - 25 Aralık 2014'te toplam yük ~47 MW (komşu günler ~95 MW; 2012 ve 2013'te de aynı düşüş var). Bu güne denk gelen pencerelerde MAPE %50–75.
   - Tatil gününe değen pencereler (testin %3,2'si) ortalama %13 MAPE veriyor; diğerleri %2,7.
   - Train'de yalnız iki Noel var ve `is_holiday` bayrağı tatilin büyüklüğünü (Noel ile sıradan bir tatil arasındaki farkı) taşımıyor. Hanede bu etki görünmüyordu, çünkü household test dönemi Noel'i içermiyor.

## 4. Sınırlar

- Dönemler ve mevsimler farklı: hane testi Nisan–Kasım 2010, şebeke testi Temmuz–Aralık 2014. Ülke de farklı (Fransa ve Portekiz).
- Tek hane ile 160 müşteri (müşteri tipleri UCI açıklamasında verilmiyor, ticari/sanayi ağırlıklı olabilir – doğrulanmalı) iki uç nokta; aradaki geçiş ölçülmedi.
- ELD'de hiperparametre araması yapılmadı (protokol gereği); derin modellerin ELD sonuçları alt sınır olarak okunmalı.
- Anomaliler sentetiktir; drop'un mutlak tanımı şebeke ölçeğine uygun değildir.
