# ELD (şebeke düzeyi) protokolü — model eğitiminden ve testten ÖNCE yazıldı

Amaç: household'da seçilmiş tahmin ve anomali yöntemlerini, **hiçbir yeni seçim veya hiperparametre araması yapmadan**, UCI ElectricityLoadDiagrams20112014 (ELD) toplam yüküne uygulamak ve hane düzeyiyle karşılaştırmak. Household sonuçlarına ve household test kümesine dokunulmaz. ELD çıktıları ayrı klasörlere yazılır: `results/eld/`, `models/eld/` ve `results/eld/experiments.csv`.

Dal: `seyda/eld-aggregate` · Tarih: 2026-10-31.

## 1. Veri (`src/data/eld.py`, testler `tests/test_eld.py`)

- **Kaynak.** Portekiz, 370 müşteri, 15 dakikalık kW; ';' ayırıcı, ',' ondalık. Zaman damgası aralık sonudur (ilk satır 00:15).
  - URL: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
  - Lisans: CC BY 4.0 (doğrulanmalı).
- **Müşteri seçimi.** 2012-01-01 00:00 damgasına kadar en az bir sıfır dışı değeri olan müşteriler alındı: **160 / 370**. Geç başlayan müşterilerin başlangıç öncesi değerleri sıfırla doldurulmuş; bunlar toplama katılmaz. Seçilenlerden biri 2014 son çeyreğinde tamamen sıfır (kapanmış olabilir); kural gereği tutuldu.
- **Yaz saati düzeltmesi** (UCI notu; düzeltme toplam yükte, 15 dakikalık düzeyde yapıldı):
  - **Mart geçiş günü:** 01:00–01:45 damgalı dört çeyrek komşuların %0,5–0,9'una düşüyor. 00:45 ile 02:00 arasında doğrusal ara değerle dolduruldu.
  - **Ekim geçiş günü:** aynı dört çeyrek komşuların 1,95–1,98 katı. İkiye bölündü.
  - **Kontrol:** desen beklenen günde görülmezse kod hata verir. 2011'de desen yok; yalnız 2012–2014 düzeltildi. Kayıt: `results/eld/tables/eld_dst_fixes.csv`.
- **Saatlik toplama.** Saat H = (H:00, H+1:00] aralığındaki dört çeyreğin ortalaması (kW).
  - Dönem 2012-01-01 – 2014-12-31: 26.304 saat, eksik saat yok.
  - Düzeltilen 12 saat `is_imputed = 1`. Metriklerden çıkarılır, enjeksiyon yapılmaz.
  - Ortalama yük 105.993 kW, en düşük 42.063 kW, en yüksek 182.820 kW.
- **Takvim.** Household ile aynı öznitelikler (`add_calendar_features`); tatiller Portekiz (PT).
- **Bölme (kronolojik):**
  - train 2012-01-01 – 2013-12-31
  - val 2014-01-01 – 2014-06-30
  - test 2014-07-01 – 2014-12-31
  - Ölçekleyiciler yalnız train'e fit edilir.

## 2. Tahmin modelleri (household seçimleri aynen; yalnız ELD'de yeniden eğitilir)

| Kod | Model | Ayar kaynağı | Seed |
|---|---|---|---|
| ELD-F1 | Seasonal Naive 24 | parametresiz, geçmiş 168 saat | – |
| ELD-F2 | Seasonal Naive 168 | parametresiz | – |
| ELD-F3-cal | Ridge + takvim (F3-cal) | `configs/config.yaml` forecast: lookback 168, lags {1…24, 48, 168}, alpha 0,1; son girdi saatinin B takvimi | – |
| ELD-F12-L336 | XGBoost (F12-L336) | config.yaml forecast.xgboost, lookback 336; early stopping val MAE | 42 |
| ELD-F5-opt | LSTM (F5-opt) | `configs/optuna_lstm_best.yaml` forecast bölümü (Optuna, household) | 42, 7, 2026 |
| ELD-F10-opt | Seq2Seq-LSTM (F10-opt) | `configs/optuna_seq2seq_best.yaml` forecast bölümü | 42, 7, 2026 |
| ELD-F13-ens | Topluluk | 0,8 · F10-opt + 0,2 · F5-opt (household ağırlıkları, yeniden seçilmez) | aynı seed çiftleri |

- **Öznitelikler:** öznitelik seti B (hedef + 8 takvim); F5/F10 için lookback 48. Household'daki ayrıntılar aynen geçerli: eğitim döngüsü, early stopping (val MAE, patience 8), ReduceLROnPlateau, gradient clipping.
- **Persistence (F0) ve diğer aileler** (MLP, GRU, CNN-LSTM) bu karşılaştırmaya dahil değil.
- **Arama yok:** yeni hiperparametre veya topluluk ağırlığı araması yapılmaz.

**Tahmin metrikleri** (household ile aynı fonksiyonlar):
- MAE, RMSE, MAPE*, sMAPE, R²; üç seviyede:
  - (a) h=1
  - (b) 24 saat ortalaması (ana metrik)
  - (c) günlük toplam: 00:00'da başlayan pencerelerin 24 saatlik toplamı (kWh)
- MAPE* eşiği (0,1 kW) bu ölçekte etkisiz; bütün saatler MAPE'ye girer.
- Hane ile karşılaştırma, ölçekten bağımsız metriklerle yapılır: MAPE*, sMAPE, R² ve aynı dönemdeki SN168'e göre MAE kazancı.

## 3. Anomali tespiti

- **Senaryo:** household **ana senaryosu**, aynen.
  - Spike: +U(3, 5)·σ, 1–2 saat.
  - Drop: ~0,05 kW, 2–6 saat; yalnız > 1,0 kW'ta başlar.
  - Seviye kayması: × U(2; 2,5), 12–48 saat.
  - Desen: gece saatleri aynı günün gündüz ortalamasına çekilir.
  - σ = ELD **train**'inde saat × haftanın günü tüketim std'si.
  - Hedef oran %2,5 (en çok %3).
  - **Not:** drop değerleri mutlak (kW) ve hane ölçeğine göre tanımlı. Şebeke düzeyinde bu, toplam yükün ~0'a inmesi, yani tam bir kesinti anlamına gelir ve tespiti kolaydır. Karşılaştırılabilirlik için değiştirilmedi.
- **Seed'ler:** val 1, 11, 21, 31, 41; test 2, 12, 22, 32, 42.
- **Yöntemler** (household tanımlarıyla aynı):
  - **A0:** hareketli z-skoru, 7 gün aynı saat.
  - **A3-s2s-norm:** ELD-F10-opt (seed 42) |h=1 hatası| / OOF std (saat × gün). OOF modelleri:
    - Household'daki 3 katlı düzen kullanılır: train'in son 12 ayı, yani 2013.
    - F10-opt ve F5-opt her katta yalnız kat öncesi veriyle eğitilir; early stopping kat öncesinin son ayıyla yapılır.
    - OOF modelleri yalnız std için kullanılır; skor, tüm train ile eğitilmiş seed 42 modeliyle hesaplanır.
  - **A5-mean-z:** A0 + A3-ens-norm (ELD-F13-ens, seed 42), z ölçeğinde ortalama. Referans: enjeksiyonsuz val.
- **Seçim (yalnız val, ana senaryo, 5 seed):**
  - Pencere w ∈ {1, 3, 6} ve eşik, birleşik olay F1'ini en büyük yapan değer. Izgara household ile aynı (0,70–0,9995 yüzdelikleri, 300 nokta). Eşitlikte pencerede küçük, eşikte yüksek değer seçilir.
  - **Yöntem seçimi yapılmaz:** üç yöntem de raporlanır; birincil yöntem household final yöntemi **A3-s2s-norm**.
- **Metrikler:** ana metrik olay F1. Ayrıca olay recall ve precision, nokta P/R/F1, PR-AUC, ROC-AUC, tip bazlı recall ve A0'a karşı eşleştirilmiş fark.

## 4. Akış

1. `python -m src.data.eld`: veri.
2. `python -m src.eld_run --stage val`: eğitim (3 seed), OOF modelleri, val tahmin ve anomali seçimi.
3. Kullanıcı val sonuçlarını görür. **Test yalnız açık onayla, bir kez:** `python -m src.eld_run --stage test`.
4. `docs/hane_vs_sebeke.md`: aynı modellerin household (`docs/final_sonuclar.md`) ve ELD test sonuçları yan yana; MAPE, R² ve SN168'e göre kazanç.

Test sonucuna göre hiçbir model, ayar, eşik veya pencere değiştirilmez.

## 5. Bilinen farklar ve sınırlar

- **Dönemler farklı mevsimlere düşüyor:**
  - Household: val Ekim–Mart, test Nisan–Kasım.
  - ELD: val Ocak–Haziran, test Temmuz–Aralık.
- **Ölçek ve kapsam farklı:** ELD toplamı 160 müşterinin toplamı (~106 MW); büyük kısmı ticari ve sanayi müşterisi olabilir (UCI açıklamasında müşteri tipi yok, doğrulanmalı). Household tek hanedir (~1 kW).
- **Early stopping val ile yapılıyor** (household'da olduğu gibi). Val sonuçları bu yüzden hafif iyimser; test etkilenmez.
