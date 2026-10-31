# Final test protokolü (test ÖNCESİ sabitlendi)

Bu belge, test kümesi açılmadan önce verilen tüm kararları kaydeder. Test sonucuna göre bu belgedeki hiçbir karar değiştirilmez. Sayılar `results/tables/` altındaki dosyalardan alınmıştır.

**Durum:** test kümesi hiç kullanılmadı. Final değerlendirme yalnız kullanıcı "final değerlendirme" dediğinde aşağıdaki komutla çalıştırılır.

---

## 1. Dört aşamalı düzen

| Aşama | Dönem | Ne yapılır | Ne yapılmaz |
|---|---|---|---|
| **Train / OOF** | 2006-12-16 – 2009-09-30 | Dedektörler ve tahmin modelleri eğitilir; A3'ün std tablosu OOF artıklarından; σ tablosu (ana senaryo) | — |
| **val-dev** | 2009-10-01 – 2009-12-31 | Yumuşatma penceresi, olay-F1 eşiği ve final yöntem **yalnız burada** seçilir (ana senaryo, 10 seed) | — |
| **val-hold** | 2010-01-01 – 2010-03-31 | Seçilen pencere ve eşiklerle **bir kez** ölçüm (ana + zor senaryo, 10'ar seed) | Hiçbir seçim veya ayar değişikliği |
| **Test** | 2010-04-01 – 2010-11-26 | val-dev seçimleriyle tek seferlik ölçüm (ana + zor, 5'er seed) | Hiçbir seçim |

- **Bağımsızlık zaman ayrımıyla sağlanır.** Dört dönem kronolojiktir, kesişmez, aralarında boşluk yoktur. val-dev + val-hold = val (testli).
- **Seed'ler yalnız enjeksiyonu belirler.** Bir seed, anomalilerin yerini, tipini ve büyüklüğünü belirler. Bağımsızlığın kaynağı seed farkı değil zaman ayrımıdır.
- **Bağlam enjeksiyonsuz alınır.** Skor hesabında geçmiş bağlam (son 8 gün) bir önceki dönemden **enjeksiyonsuz** gelir:
  - val-hold için temiz val-dev,
  - test için temiz val-hold.
- **Gelecek kullanılmaz.** val-dev'in bağlamlı tablosu val-hold'a uzanmaz. Bunu bir test kontrol ediyor.

### Bilinen sınırlar (kabul edildi, testi etkilemez)

1. **Checkpoint seçimi tam val ile yapıldı.**
   - Tüm train ile eğitilmiş F10-opt / F5-opt ve A2 (LSTM-AE) checkpoint'leri, early stopping'de **tam val'i** (val-dev + val-hold, enjeksiyonsuz) kullandı.
   - Ağırlıklar val ile güncellenmedi; yalnız epoch seçimi ve lr takvimi val'e bağlı.
   - Bu yüzden val-hold, bu modeller için tam bağımsız değildir. Test bu sorunu taşımaz.
2. **Modeller yeniden eğitilmedi.** Final skorlar mevcut checkpoint'lerle üretilir. OOF modelleri yalnız std kalibrasyonu içindir.

## 2. Train / OOF: in-sample std sorunu ve düzeltmesi

- **Sorun:** A3'ün normalize skoru `|h=1 hatası| / std(saat, gün)` şeklinde. A-1–A-5'te bu std, tüm train ile eğitilmiş modelin **kendi train artıklarından** hesaplandı (in-sample). Model bu saatleri gördüğü için std'nin küçük, val skorlarının şişik olması beklenir.
- **Düzeltme:** Aşama 4'ün 3 katlı zamansal CV düzeni (train'in son 12 ayı) kullanıldı.
  - **Kat dönemleri:** 2008-10 – 2009-01, 2009-02 – 2009-05, 2009-06 – 2009-09.
  - Her katta F10-opt ve F5-opt aynı hiperparametrelerle, **yalnız kattan önceki** veriyle eğitildi.
  - Early stopping için kattan hemen önceki ay kullanıldı (iç val). Böylece katın saatleri modele ne ağırlık, ne epoch seçimi, ne de lr takvimi yoluyla girdi.
  - Kat saatlerinin h=1 (ve 24 ufuk) tahminleri **OOF artıkları**dır. Std tabloları bunlardan hesaplanır.
  - Kodlar: `seyda/model-fixes` dalında `src/train_oof.py`; `seyda/anomaly` dalında `out_of_fold_predictions` ve `fit_detectors(std_source="oof")`.
- **Sonuç** (`results/tables/anomaly_std_in_sample_vs_oof.csv`, `results/figures/P_train_residual_std_in_sample_vs_oof.png`): OOF std, aynı dönemin in-sample std'sinden saat ortalamasında yalnız **%2–3 büyük** (oran: A3-s2s 1,023, A3-ens 1,026; saate göre 0,97–1,05).
  - Beklenen yön doğru, ama etki küçük. In-sample std kaynaklı iyimserlik A-1–A-5 sonuçlarında küçük bir pay.
  - OOF yalnız train'in son 12 ayını kapsar; hücre başına yaklaşık 52 örnek düşer (in-sample'da yaklaşık 145).

## 3. Anomali tanımı: iki senaryo

Anomali, o saatin **olağan oynaklığının dışında kalan sapma** olarak tanımlanır. Ana senaryoda büyüklük, train'den saat × haftanın günü için hesaplanan tüketim std'sine (σ) göre verilir. σ yalnız train'den ve yalnız doldurulmamış saatlerden hesaplanır.

| Tip | Ana (seçim burada) | Zor (A-1–A-5 ayarları, aynen) |
|---|---|---|
| Spike | 1–2 saat, + U(3, 5)·σ | 1–2 saat, × U(2, 4) |
| Drop | 2–6 saat, ~0,05 kW; yalnız gerçek değer > 1,0 kW iken başlar | 2–6 saat, ~0,05 kW |
| Seviye kayması | 12–48 saat, × U(2; 2,5), olay başına tek çarpan | 12–48 saat, × 1,5 |
| Desen | gece 00–05 → aynı günün 08–20 ortalaması | aynı |

- **Hedef oranlar:** her iki senaryoda %2,5, en fazla %3.
- **Zor senaryo ayarları A-1–A-5 ile aynı.** Zor senaryo, A-1–A-5'teki enjeksiyonla aynı rastgele sayı tüketimini kullanır; bunu bir test kontrol ediyor.
- **Literatür atfı:** σ tabanlı enjeksiyon için atıf eksik; kaynak yer tutucusu konuldu, **doğrulanmalı**.

| Senaryo | val-dev seed'leri | val-hold seed'leri | test seed'leri |
|---|---|---|---|
| Ana | 1, 11, 21, …, 91 | 3, 13, 23, …, 93 | 2, 12, 22, 32, 42 |
| Zor | 5, 15, 25, …, 95 | 7, 17, 27, …, 97 | 4, 14, 24, 34, 44 |

Not: A-1–A-5 config'inde test için tek seed (2) vardı. Test seed listeleri bu protokolle eklendi; test hiç çalıştırılmadı.

## 4. Metrikler ve seçim kuralları

- **Ana metrik: olay F1.**
  - Olay = aynı tipten ardışık etiketli saatler.
  - Olay recall'ü = en az bir saati işaretlenen olay oranı.
  - Olay precision'ı = gerçek bir anomaliyle örtüşen alarm segmentleri / tüm alarm segmentleri.
  - Nokta P/R/F1, PR-AUC ve ROC-AUC tabloda kalır. Point-adjust uygulanmaz.
- **Eşik:** val-dev ana senaryonun 10 kopyasında **birleşik** olay F1'ini en büyük yapan değer.
  - Arama, birleşik skor dağılımının 0,70–0,9995 yüzdeliklerinde 300 noktalı bir ızgara üzerinde yapılır.
  - Eşitlikte yüksek eşik seçilir.
- **Yumuşatma:** yalnız geçmişe bakan hareketli ortalama, w ∈ {1, 3, 6}. Seçilen eşikteki ortalama olay F1'i en yüksek pencere seçilir; eşitlikte küçük pencere.
- **Değerlendirilen yöntemler:**
  - A0 z-skoru, A1 Isolation Forest, A2 LSTM-AE
  - A3-s2s-norm, A3-ens-norm (ikisi de OOF std ile)
  - A3-s2s-hmean (ufuk başına OOF std)
  - A5-mean-z: A0 + A3-ens-norm; z ölçeğinin referansı temiz val-dev
- **Final yöntem kuralı:**
  1. val-dev ana senaryoda ortalama olay F1'i en yüksek yöntem referans alınır.
  2. Referansa karşı eşleştirilmiş seed farkının |ortalaması| ≤ std'si olan yöntemler gürültü düzeyinde sayılır.
  3. Referans ile bu yöntemler arasından en basiti seçilir. Basitlik sırası: A0 < A1 < A2 < A3-s2s-norm < A3-s2s-hmean < A3-ens-norm < A5-mean-z.

## 5. Seçim sonucu (val-dev, ana senaryo): `results/tables/anomaly_protocol_selection.csv`

| Yöntem | w | Olay-F1 eşiği | val-dev olay F1 | Referansa (A5-mean-z) fark | Gürültü düzeyi? |
|---|---|---|---|---|---|
| A5-mean-z | 3 | 2,454 | 0,642 ± 0,114 | (referans) | – |
| A3-ens-norm | 3 | 3,447 | 0,617 ± 0,196 | −0,024 ± 0,139 | evet |
| A3-s2s-norm | 3 | 3,558 | 0,575 ± 0,181 | −0,067 ± 0,126 | evet |
| A3-s2s-hmean | 3 | 2,943 | 0,562 ± 0,127 | −0,080 ± 0,089 | evet |
| **A0** | 3 | 6,581 | 0,536 ± 0,206 | −0,106 ± 0,148 | evet |
| A2 | 6 | 0,377 | 0,364 ± 0,189 | −0,277 ± 0,165 | hayır |
| A1 | 1 | 0,607 | 0,318 ± 0,110 | −0,323 ± 0,144 | hayır |

**Kurala göre final yöntem: A0 (hareketli z-skoru, w = 3, eşik 6,581).** A0, referansa karşı gürültü düzeyinde (fark 0,106 < std 0,148) ve basitlik sırasında ilk sırada.

Bu karar val-hold sonucundan bağımsız olarak verildi ve val-hold'a göre değiştirilmedi. val-hold sonuçları ve kuralın bu seçimdeki etkisi deney günlüğündedir.

## 6. Tahmin tarafı: üç seviyeli raporlama (`seyda/model-fixes`)

Final tahmin modeli **F13-ens** (0,8 · F10-opt + 0,2 · F5-opt). Test'te aynı üç seviye raporlanır; yeni model eğitilmez.

- **(a) h=1:** her 24 saatlik tahminin yalnız ilk saati.
- **(b) 24 saat ortalaması:** mevcut ana metrik.
- **(c) Günlük toplam:** ilk hedef saati 00:00 olan pencerelerin 24 saatlik tahmin toplamı ile gerçek günlük toplam. Birim kWh; metrikler MAE, MAPE, R².

Val sonuçları `results/tables/forecast_levels_val.csv` dosyasında (kod: `src/forecast_levels.py`).

## 7. Final değerlendirme komutları (yalnız kullanıcı onayıyla)

```bash
# Anomali (seyda/anomaly): val-dev seçimleri anomaly_protocol_selection.csv'den okunur, seçim yapılmaz
python -m src.anomaly_protocol --split test
```

Tahmin test komutu ayrıca belirlenecek. Test kümesi bu belge yazılırken hiç kullanılmadı.
