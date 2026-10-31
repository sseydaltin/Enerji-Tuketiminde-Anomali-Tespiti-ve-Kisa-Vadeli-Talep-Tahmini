# Final sonuçlar: val (dev, hold) ve test yan yana

Test kümesi (2010-04-01 – 2010-11-26) **bir kez** kullanıldı. Değerlendirme kodu testten önce sabitlendi (commit `f5e3efd`); test sonucuna göre hiçbir model, eşik, pencere veya yöntem seçimi değiştirilmedi.

Kararlar: `docs/final_test_protokolu.md`. Tüm sayılar `results/tables/` altındaki dosyalardan:
- `final_forecast_val_vs_test.csv`, `forecast_levels_val.csv`, `forecast_levels_test.csv`
- `final_anomaly_val_vs_test.csv`, `anomaly_protocol_test*.csv`, `anomaly_test_top_events_clean.csv`

Tarih: 2026-10-31.

**Dönemlerin mevsimi farklı:**

| Dönem | Tarih aralığı | Mevsim | Ortalama tüketim (std) |
|---|---|---|---|
| val-dev | 2009-10 – 2009-12 | sonbahar–kış | 1,26 kW (0,92) |
| val-hold | 2010-01 – 2010-03 | kış | 1,31 kW (0,90) |
| test | 2010-04 – 2010-11 | ağırlıklı ilkbahar–yaz | 0,96 kW (0,71) |

Test döneminde gündüz (08–20) ortalaması 1,10 kW; val'de 1,58–1,60 kW. Bu fark aşağıdaki sapmaların çoğunu açıklıyor.

---

## 1. Tahmin (24 saat ufuk)

### 1.1 Tüm modeller: val ve test, 24 saat ortalaması

| Model | Seed | Val MAE (kW) | Test MAE (kW) | Val RMSE | Test RMSE | Val R² | Test R² | SN168'e göre, val | SN168'e göre, test |
|---|---|---|---|---|---|---|---|---|---|
| F0 Persistence | 1 | 0,960 | 0,692 | 1,254 | 0,942 | −0,888 | −0,749 | −54,4 % | −26,0 % |
| F1 Seasonal Naive 24 | 1 | 0,691 | 0,512 | 0,990 | 0,760 | −0,177 | −0,137 | −11,2 % | +6,8 % |
| F2 Seasonal Naive 168 | 1 | 0,622 | 0,549 | 0,905 | 0,795 | 0,016 | −0,244 | 0 | 0 |
| F3-cal Ridge | 1 | 0,582 | 0,467 | 0,761 | 0,612 | 0,305 | 0,262 | +6,4 % | +15,0 % |
| F4 MLP (gelecek öz.) | 1 | 0,502 | 0,432 | 0,686 | 0,584 | 0,435 | 0,328 | +19,3 % | +21,4 % |
| F5-opt LSTM | 3 | 0,481 ± 0,002 | 0,413 ± 0,005 | 0,682 | 0,581 | 0,441 | 0,336 | +22,6 % | +24,8 % |
| F6-C GRU | 1 | 0,488 | 0,417 | 0,682 | 0,581 | 0,442 | 0,335 | +21,5 % | +24,1 % |
| F10-opt Seq2Seq | 3 | 0,469 ± 0,006 | 0,414 ± 0,007 | 0,691 | 0,612 | 0,427 | 0,263 | +24,6 % | +24,7 % |
| F11-L48 CNN-LSTM | 1 | 0,484 | 0,426 | 0,682 | 0,582 | 0,442 | 0,334 | +22,2 % | +22,4 % |
| F12-L336 XGBoost | 1 | 0,500 | 0,416 | 0,681 | **0,568** | 0,444 | **0,364** | +19,6 % | +24,2 % |
| **F13-ens (final)** | 3 | **0,4625 ± 0,0027** | **0,4062 ± 0,0047** | 0,679 | 0,598 ± 0,008 | 0,447 | 0,297 ± 0,019 | **+25,6 %** | **+26,0 %** |

**Kadro:** her ailenin test öncesi val'de seçilmiş temsilcisi. F7–F9 model değil, deney adımıdır (ablasyon, ızgara, seed kararlılığı).

**Model ağırlıkları:**
- Derin modeller kayıtlı, yalnız train ile eğitilmiş checkpoint'lerle çalıştı.
- Ridge ve XGBoost, val'deki koşuyla aynı biçimde yeniden fit edildi. Ridge kapalı çözüm; XGBoost'ta early stopping val ile.
- Aynı kod val tahminlerini birebir yeniden üretiyor: fark ≤ 2,3e-7 kW (`forecast_final_val_reproduction.csv`).

### 1.2 Final model F13-ens ve üyeleri: üç seviye

| Model | Seviye | Val MAE | Test MAE | Val R² | Test R² | Val MAPE* | Test MAPE* |
|---|---|---|---|---|---|---|---|
| F13-ens | (a) h=1 (kW) | 0,382 ± 0,007 | 0,336 ± 0,007 | 0,625 | 0,527 | 36,7 % | 40,7 % |
| F13-ens | (b) 24 saat (kW) | 0,4625 ± 0,0027 | 0,4062 ± 0,0047 | 0,447 | 0,297 | 46,7 % | 49,4 % |
| F13-ens | (c) günlük toplam (kWh) | 5,00 ± 0,17 (177 gün) | 4,38 ± 0,08 (229 gün) | 0,168 | 0,330 | 16,8 % | 18,9 % |
| F10-opt | (c) günlük toplam (kWh) | 5,11 ± 0,17 | 4,62 ± 0,14 | 0,128 | 0,269 | 17,2 % | 20,2 % |
| F5-opt | (c) günlük toplam (kWh) | 4,94 ± 0,09 | 4,03 ± 0,14 | 0,189 | 0,416 | 16,5 % | 17,8 % |
| F12-L336 | (c) günlük toplam (kWh) | – | **3,62** | – | **0,506** | – | 17,1 % |
| F2 SN168 | (c) günlük toplam (kWh) | 6,68 | 5,50 | −0,499 | −0,068 | 24,3 % | 25,5 % |
| F1 SN24 | (c) günlük toplam (kWh) | 6,95 | 4,47 | −0,492 | 0,186 | 24,0 % | 19,0 % |

**Figürler:** `results/figures/F_final_test_mae_by_horizon.png` (ufka göre MAE) ve `F_final_test_mae_by_hour.png` (günün saatine göre MAE).

### 1.3 Val'den sapmalar ve olası nedenleri

1. **Test'te MAE düşük, R² de düşük (bütün modellerde).** Yaz tüketimi daha düşük ve daha az oynak (std 0,71 kW; val'de 0,90–0,92). Mutlak hatalar küçülüyor; ama açıklanacak varyans da küçüldüğü için R² düşüyor. MAE'yi dönemler arasında doğrudan "iyileşme" diye okumamak gerekir. Karşılaştırma, aynı dönemdeki SN168'e göre yapılmalı.
2. **Final model MAE'de birinci, RMSE'de değil.**
   - F13-ens test MAE'de yine birinci (SN168'e göre +26,0 %; val'de +25,6 %).
   - Ancak test RMSE ve R²'de XGBoost (0,568 / 0,364), LSTM ve GRU daha iyi.
   - Olası neden: F10-opt L1 kaybıyla eğitildi (medyan benzeri tahmin), bu yüzden tepeleri küçük tahmin ediyor ve büyük hatalar RMSE'yi şişiriyor. Bu desen val'de de vardı (F10-opt val RMSE 0,691, en kötü derin modeller arasında); testte daha belirgin.
3. **Seq2seq ile LSTM arasındaki fark testte yok.** Val'de F10-opt, F5-opt'tan 0,012 kW iyiydi; testte ikisi eşit (0,414 / 0,413). Val'deki seçim avantajının bir kısmı val dönemine özgüymüş.
4. **Naif sıralama tersine döndü.** Testte SN24, SN168'den iyi (0,512 / 0,549); val'de tersiydi. Olası neden: yazın haftalık desen zayıflıyor (tatil, düzensiz evde bulunma), dünkü gün daha iyi bir tahmin oluyor. Topluluğun SN168'e göre kazancı (+26,0 %), daha güçlü SN24'e göre +20,6 %.
5. **Günlük toplamda en iyi XGBoost.** Testte XGBoost MAE 3,62 kWh, R² 0,51; F13-ens 4,38 kWh, R² 0,33. F13-ens günlük seviyede üyesi F5-opt'tan bile kötü. Topluluk ağırlıkları 24 saatlik saatlik MAE için seçildi; günlük toplamı optimize etmiyor. Seq2seq'in sistematik düşük tahmini, toplamda birikerek büyüyor (olası neden, doğrulanmadı).

## 2. Anomali tespiti

**Final yöntem: A3-s2s-norm.** Tanım: F10-opt'un |h=1 hatası| / OOF std (saat × gün); yumuşatma w = 3, olay-F1 eşiği 3,558. Revizyon 1 kuralıyla, yalnız val-dev'de seçildi.

### 2.1 Olay F1 (ort ± std): val-dev / val-hold (10 seed), test (5 seed)

| Yöntem | Ana: val-dev | Ana: val-hold | **Ana: test** | Zor: val-dev | Zor: val-hold | **Zor: test** |
|---|---|---|---|---|---|---|
| A0 z-skoru | 0,536 | 0,289 | 0,398 ± 0,064 | 0,302 | 0,235 | 0,291 ± 0,076 |
| A1 Isolation Forest | 0,318 | 0,250 | 0,378 ± 0,144 | 0,221 | 0,361 | 0,223 ± 0,093 |
| A2 LSTM-AE | 0,364 | 0,208 | 0,352 ± 0,099 | 0,141 | 0,219 | 0,203 ± 0,097 |
| **A3-s2s-norm (final)** | 0,575 | 0,494 | **0,384 ± 0,170** | 0,346 | 0,431 | **0,318 ± 0,073** |
| A3-ens-norm | 0,617 | 0,559 | 0,401 ± 0,130 | 0,299 | 0,504 | 0,306 ± 0,065 |
| A3-s2s-hmean | 0,562 | 0,417 | 0,454 ± 0,036 | 0,424 | 0,392 | **0,347 ± 0,054** |
| A5-mean-z | 0,642 | 0,462 | **0,463 ± 0,048** | 0,357 | 0,394 | 0,317 ± 0,037 |

**Test, ana senaryo, diğer metrikler:**

| Yöntem | Olay R | Olay P | Nokta P | Nokta R | Nokta F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|---|
| A0 | 0,43 | 0,39 | 0,21 | 0,10 | 0,140 | 0,148 | 0,81 |
| **A3-s2s-norm** | 0,36 | 0,42 | 0,41 | 0,09 | 0,147 | 0,241 | 0,86 |
| A3-ens-norm | 0,39 | 0,43 | 0,41 | 0,10 | 0,163 | 0,245 | 0,86 |
| A3-s2s-hmean | 0,50 | 0,42 | 0,44 | 0,22 | 0,293 | 0,264 | 0,88 |
| A5-mean-z | 0,58 | 0,39 | 0,30 | 0,19 | 0,228 | 0,213 | 0,87 |

Test kopyası başına ortalama 15,4 olay (ana) ve 13,8 olay (zor) var; val dönemlerinde 7–9.

**Tip bazlı olay recall'ü** (ana senaryo; spike / drop / seviye kayması / desen):

| Yöntem | val-dev | val-hold | test |
|---|---|---|---|
| A0 | 0,11 / 0,00 / 0,70 / 0,69 | 0,00 / 0,00 / 0,75 / 0,40 | 0,33 / 0,00 / 0,62 / 0,49 |
| **A3-s2s-norm** | 0,35 / 0,00 / 0,90 / 0,90 | 0,29 / 0,06 / 0,81 / 0,68 | 0,27 / 0,10 / 0,73 / **0,18** |
| A5-mean-z | 0,40 / 0,00 / 1,00 / 0,91 | 0,37 / 0,04 / 0,88 / 0,67 | 0,69 / 0,00 / 0,90 / 0,48 |

**A0'a karşı eşleştirilmiş fark** (test, olay F1; parantez içi, 5 seed'den kaçında yöntem A0'dan iyi):

| Yöntem | Ana senaryo | Zor senaryo |
|---|---|---|
| A3-s2s-norm | −0,015 ± 0,184 (2/5) | +0,027 ± 0,121 (4/5) |
| A3-ens-norm | +0,002 ± 0,156 (2/5) | +0,015 ± 0,133 (2/5) |
| A3-s2s-hmean | +0,056 ± 0,066 (4/5) | +0,055 ± 0,116 (3/5) |
| A5-mean-z | +0,065 ± 0,045 (5/5) | +0,026 ± 0,080 (3/5) |

Eşikten bağımsız PR-AUC'de bütün A3 varyantları testte A0'ın üzerinde: ana senaryoda +0,09 ile +0,12 arası, 5 seed'in hepsinde (`anomaly_protocol_test_paired_vs_A0.csv`).

### 2.2 Val'den sapmalar ve olası nedenleri

1. **Final yöntemin testteki olay F1'i val'in çok altında.** Ana senaryoda val-dev'de 0,575, val-hold'da 0,494, testte 0,384. Testte A0'dan ayırt edilemiyor (−0,015 ± 0,184).
   - Olay recall'ü 0,63 / 0,47'den 0,36'ya düştü; precision ise benzer kaldı (0,42).
   - En büyük kayıp desen anomalilerinde: recall 0,90 / 0,68'den 0,18'e.
2. **Olası neden 1, mevsim farkı ve sabit normalizasyon.**
   - A3'ün std tablosu (saat × gün) train'in bütün mevsimlerinden hesaplandı.
   - Yazın tüketim ve tahmin hatası küçülüyor: F10-opt h=1 MAE val'de 0,387, testte 0,339 kW. Yıllık std ile bölünen yaz hataları küçük skor üretiyor ve kış ağırlıklı val'de seçilen sabit eşik (3,56) testte daha az alarm veriyor.
   - Mevsime duyarlı bir normalizasyon (ör. ay × saat) bu etkiyi azaltabilirdi; protokol gereği denenmedi.
3. **Olası neden 2, enjekte anomaliler yazın daha küçük.**
   - Desen anomalisi geceyi aynı günün gündüz ortalamasına çekiyor. Yazın gündüz düzeyi düşük (1,10 kW, val'de 1,6 kW), bu yüzden desen sapması testte ortalama 0,69 kW; val'de 1,04–1,25 kW.
   - Seviye kayması orana dayalı (×2–2,5) ve yazın mutlak olarak daha küçük (1,31 kW; val 1,42–1,66). Bu sayıların kaynağı, 5'er seed'lik bir enjeksiyon özeti (yalnız enjeksiyon, model yok).
   - Yani test, aynı tanımla daha zor bir görev oldu.
4. **Testte en iyi yöntemler A5-mean-z ve A3-s2s-hmean.** Bu iki yöntem val-dev'de final kuralına göre seçilmemişti. A5-mean-z, A0'ın orana bağlı z-skorunu da kullandığı için mevsim farkından daha az etkileniyor olabilir (hipotez).
5. **Zor senaryoda sıralama daha kararlı.** Tüm yöntemler 0,20–0,35 arasında ve A3 varyantları A0'ın biraz üzerinde.
6. **Belirsizlik büyük.** Testte 5 seed var ve kopya başına ~15 olay düşüyor; A3-s2s-norm'un seed başına olay F1'i 0,20 ile 0,58 arasında değişiyor.
7. **Bağımsızlık notu.** Gürültü kuralı revizyonu val-hold sonuçları görüldükten sonra yapıldı (protokol §5a). Test bundan etkilenmedi ve sonuç ilk ve tek ölçümdür.

### 2.3 Enjeksiyonsuz testte final yöntemin en yüksek skorlu olayları

Eşiği (3,558, w = 3) geçen blok sayısı yalnız **8** (5.481 saat); 10 olay çıkmadı, eşik altındakiler eklenmedi. Tablo `anomaly_test_top_events_clean.csv` dosyasında.
- Alt sayaç değerleri saatlik toplam Wh:
  - Sub_1: mutfak
  - Sub_2: çamaşırhane
  - Sub_3: su ısıtıcı ve klima (UCI veri seti tanımı – doğrulanmalı)
  - "diğer": alt sayaçlarla ölçülmeyen tüketim
- **Açıklamalar hipotezdir**; gerçek etiket yoktur.

| # | Başlangıç | Süre | Maks. skor | Gerçek / h=1 tahmini (kW) | Gün | Veride görülen | Olası açıklama (hipotez) |
|---|---|---|---|---|---|---|---|
| 1 | 2010-07-22 00:00 | 5 sa | 6,21 | 1,82 / 0,74 | Perşembe | Sub_3 466 Wh/sa, diğer 1.257 Wh/sa (test gece ort. Sub_3 ≈ 130–170) | Gece klima / su ısıtıcı + ölçülmeyen yük; sıcak bir yaz gecesi olabilir |
| 2 | 2010-10-31 05:00 | 3 sa | 4,91 | 1,72 / 1,46 | Pazar | Gece boyunca yüksek (00:00'dan 1,4–1,8 kW), 05:00'te 2,91 kW, Sub_1 477 Wh/sa | Yaz saati uygulamasının bittiği gece; olağandışı gece mutfak kullanımı (misafir, dönüş) |
| 3 | 2010-07-01 00:00 | 2 sa | 4,44 | 2,10 / 0,82 | Perşembe | Sub_1 439, Sub_3 451, diğer 1.140 Wh/sa | 1 ile aynı desen: Temmuz'da Perşembe gece yarısı yüksek tüketim |
| 4 | 2010-07-15 00:00 | 2 sa | 4,29 | 1,97 / 1,05 | Perşembe | Sub_3 516, diğer 1.313 Wh/sa | Yine Perşembe gece yarısı. Tekrarlıyor (3 Perşembe); zamanlanmış bir cihaz ya da düzenli bir alışkanlık olabilir, yani "anomali" değil, modelin bilmediği bir rutin |
| 5 | 2010-11-15 14:00 | 1 sa | 4,13 | 0,43 / 2,05 | Pazartesi | Sub_1 ve Sub_3 sıfır | Beklenen öğle tüketimi gerçekleşmedi; evde kimse yok (düşüş tipi) |
| 6 | 2010-05-11 02:00 | 1 sa | 4,08 | 1,46 / 0,51 | Salı | Sub_3 1.145 Wh/sa | Gece su ısıtıcı; gece tarifesinde ısıtma olabilir ama olağandan uzun / güçlü |
| 7 | 2010-11-04 10:00 | 1 sa | 3,94 | 3,63 / 1,75 | Perşembe | Sub_2 1.954, Sub_3 1.079 Wh/sa | Çamaşır makinesi / kurutucu + su ısıtıcı (spike tipi) |
| 8 | 2010-06-21 16:00 | 1 sa | 3,63 | 0,38 / 1,47 | Pazartesi | Bütün alt sayaçlar düşük | Beklenen ikindi tüketimi yok; ev boş |

Bu 8 bloğun hiçbiri doldurulmuş (`is_imputed`) saat değil; hiçbiri resmî tatile denk gelmiyor.

## 3. Genel değerlendirme ve sınırlar

- **Tahmin:** final model F13-ens, testte de MAE'de en iyi model (0,406 ± 0,005 kW, SN168'e göre +26 %); val'deki kazanç korundu. RMSE, R² ve günlük toplamda ise XGBoost ve LSTM daha iyi. "En iyi model" iddiası metriğe bağlı.
- **Anomali:** val'de seçilen A3-s2s-norm, testte A0'ı geçemedi. Mevsim farkı, val'de kalibre edilen bir eşiğin ve yıllık normalizasyonun yaz dönemine taşınamadığını gösteriyor. PR-AUC'de A3 ailesi testte de A0'dan açıkça iyi (sıralama gücü korunuyor). Sorun eşik ve ölçek kalibrasyonunda. Bu bir bulgu olarak raporlanır, düzeltme yapılmadı.
- **Bilinen sınırlar:**
  - F10-opt, F5-opt ve A2'nin early stopping'i tam val'i kullandı.
  - val-hold, revizyon 1'den sonra seçim açısından tam bağımsız değil.
  - Test tek seferlik ve bağımsız.
  - Anomali sonuçları sentetik enjeksiyona dayanıyor.
