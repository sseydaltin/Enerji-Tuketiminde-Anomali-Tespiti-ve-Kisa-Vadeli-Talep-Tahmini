# Proje incelemesi, zorunlu düzeltmeler ve kısa plan (2026-11-08, teslim 2026-11-09)

Dal: `dilara/inceleme-zorunlu-duzeltmeler` (taban `825f7c7`), yerel; push/merge yok.

> Bu belgede yeni bir final sonuç yok. Test kümesi ve val-hold bu çalışmada kullanılmadı.
> Bütün sayılar `results/tables/` dosyalarından geliyor. Yeniden üretim komutları §6'da.

## 0. İnceleme kapsamı

| Durum | Alan |
|---|---|
| Okundu (kod) | `src/data/load.py`, `preprocess.py` (doldurma), `split.py`, `windows.py`, `inject_anomalies.py` (segment kuralı, desen); `src/train_forecast.py` (veri hazırlama, gelecek öznitelikleri, eğitim döngüsü, early stopping); `src/models/seq2seq.py`, `residual_detector.py`; `src/evaluate_anomaly.py`, `anomaly_protocol.py` (bölmeler, test çalıştırma), `anomaly_dev.py`, `anomaly_short_event.py`, `utils/anomaly_metrics.py`, `utils/results_io.py`, `forecast_levels.py`, `evaluate_forecast.py` (kayıt/okuma), `verify_forecast_eval.py` (okuma); `app/services.py` (hane günlük tahmini), `app/model_info_services.py`, `app/pages/model_info.py` (bilinen sorunlar) |
| Kısmen / arama ile | `app/anomaly_services.py` (eşik ve ölçek dosyaları), `app/pages/forecast.py` (zaman alanları), `src/models/lstm.py`, `gru.py`, `mlp.py`, `cnn_lstm.py` (yalnız yapı), `tune_optuna.py` |
| İncelenmedi | `src/models/xgb_forecast.py`, `autoencoder.py`, `anomaly_baselines.py` (iç ayrıntı), `train_oof.py`, `anomaly_dual.py`, `eld_*`, `app/pages/anomaly.py` ve `results.py` (görselleştirme ayrıntısı), notebook'lar, ders PDF'leri, `seyda/rapor` dalındaki öneri formu ve rapor taslağının gövdesi |

Bu yüzden "tüm proje satır satır incelendi" denemez. Veri → bölme → pencere → eğitim → artık → değerlendirme → demo tahmin akışı uçtan uca izlendi. Anomali sayfasının görsel katmanı izlenmedi.

**Sızıntı ve hizalama kontrolleri:**
- Bölmeler kronolojik ve kesişmiyor.
- Ölçekleyiciler yalnız train'e fit ediliyor.
- Val/test hedefleri kendi kümesinde kalıyor; girdi bağlamı önceki kümeden geliyor ve bu belgeli.
- Seq2Seq decoder'ı yalnız hedef saatlerin takvimini alıyor (önceden bilinir).
- Decoder tek yönlü LSTM; bu yüzden h=1 artığı ilk takvime bağlı ve geçerli.
- A3 artıkları yalnız t'den önceki gözlemleri kullanıyor (testle doğrulandı).
- Merkezli pencere ve `bfill` yok.
- Bilinen ve belgelenmiş sınır: F10-opt / F5-opt'un early stopping'i tam val'i kullandı, dolayısıyla kış-dev = val-dev bu modeller için tam bağımsız değil.

## 1. Doğrulanan ve düzeltilen sorunlar

| # | Sorun | Kanıt | Düzeltme | Skora etkisi (ölçüldü) |
|---|---|---|---|---|
| D1 | **Tahmin dosyalarında zaman birimi.** pandas 3.1'de zamanlar µs; `asi8` µs yazıyor, okurken ns sanılıyor → 1970 | 2 test başarısızdı | `dilara/demo`'daki düzeltme (yalnız 5 dosya, ilgisiz değişiklik yok) bu dala uygulandı. Kayıt hep ns, okuma `unit="ns"`. Ek olarak `verify_forecast_eval.py` okuması da düzeltildi | **Yok.** Gerçek veriyle pencere ve gün sayıları kayıtlı tablolarla aynı: val 4213 / 177, test 5481 / 229 (`tests/test_time_alignment.py`). Hedefler tablodaki saatlerle birebir |
| D2 | **Ön işlemede aynı birim hatası.** `fill_missing_hours`, `asi8` (µs) değerini `Timedelta.value` (ns) ile karşılaştırıyor → küme sınırı koruma bölgesi ±3 saat yerine ±125 gün | Korunan saat sayısı: mevcut kodla 6001, doğrusu 7. Yeni test eski kodda başarısız | `Timedelta` karşılaştırması (birimden bağımsız) + µs/ns testi | **Yok.** Ham veriden yeniden üretimde doldurulan değerler aynı (değişen saat 0). O bölgedeki bütün boşluklar 3 saatten uzun ve zaten hafta-önce yöntemiyle doldurulmuş |
| D3 | **Demo testi ELD ağırlığı yoksa başarısız** | `test_demo_model_info` 4 checkpoint okuması bekliyordu | Beklenen sayı yerelde bulunan ELD dosyalarına göre. ELD isteğe bağlı bir özellik (README §4.13; sayfa yokluğu gösteriyor) | Skor yok. Diğer ELD testleri gerekçeyle atlanıyor (7 atlama, hepsi ELD) |
| D4 | **Belge–kod çelişkisi** | README §1 ve CLAUDE.md "GRU artığı" ve "168 s" diyordu; checkpoint'lerde lookback 48 ve artık F10-opt'tan | README §1, CLAUDE.md "Modeller" satırı, Model Bilgisi sayfasındaki bilinen sorun metni güncellendi | Yok |

**Olay metriği: uygulama hatası yok, tanım sorunu var.**
- `event_metrics` belgelenen tanımı doğru uyguluyor:
  - olay = aynı tipten ardışık etiketli saatler (enjeksiyon olaylar arasında en az bir normal saat bırakıyor),
  - recall = en az bir saati işaretlenen olay / tüm olaylar,
  - precision = anomaliyle kesişen alarm bloğu / tüm alarm blokları.
- Küçük bağımsız kontroller `tests/test_event_metric_definitions.py`'de: parçalı alarm, birleştirme, bir alarmın iki olayı kapsaması, kaçırılan olay, yanlış alarm, boş durumlar.
- Sorun tanımın kendisinde. Bir olayın içinde parçalanan alarmlar birden çok "doğru blok" sayılıyor ve precision'ı yükseltiyor; aynı alarmları birleştirmek precision'ı düşürüyor (test: 3/4 → 1/2).
- Bu yüzden ikinci bir tanım **ek olarak** raporlanıyor:

| Ad | Precision | Recall | Gerekçe |
|---|---|---|---|
| `block_f1` (mevcut "olay F1") | doğru alarm bloğu / alarm bloğu | yakalanan olay / olay | Önceki bütün sonuçlar ve seçimler bununla yapıldı |
| `count_f1` (olay sayımlı) | yakalanan olay / (yakalanan olay + yanlış blok) | aynı | Her gerçek olay bir kez sayılır; yanlış blok iki tanımda aynı. **Birebir alarm–olay eşleştirmesi değildir:** bir alarm iki olayı kapsarsa TP = 2; bir olaydaki fazla alarmlar FP sayılmaz. Standart ya da tartışmasız doğru bir metrik olarak sunulmaz |

- İki tanımın da sınırı var:
  - Hiçbiri gereğinden uzun alarmı cezalandırmıyor; nokta precision'ı bunun için tabloda duruyor.
  - `count_f1`'de tek bir alarm iki olayı kapsarsa iki doğru tespit sayılıyor.

## 2. Düzeltmelerin skorlara etkisi

- **Tahmin:** D1 ve D2 kayıtlı hiçbir tahmin sayısını değiştirmiyor (yukarıdaki ölçümler).
- **Anomali, mevcut tanım:** Denetim betiği Aşama 3'ün S0–S5 kopya F1'lerini birebir yeniden üretiyor (120 kopya, fark 0).
- **Anomali, olay sayımlı tanım** (`anomaly_eval_audit_summary.csv`; kış-dev + yaz-dev, orta senaryo, kayıtlı ayarlar, yeni seçim yok):

| Yöntem | block_f1 kış / yaz / ort. | count_f1 kış / yaz / ort. |
|---|---|---|
| A0 z-skoru | 0,52 / 0,31 / 0,414 | 0,51 / 0,27 / 0,391 |
| A3-s2s-norm (final yöntem, Aşama 2 ayarıyla) | 0,54 / 0,53 / 0,535 | 0,44 / 0,49 / 0,466 |
| A3-s2s-hmean (Aşama 2 ayarı) | 0,56 / 0,53 / 0,542 | 0,45 / 0,46 / 0,455 |
| A5-mean-z | 0,61 / 0,48 / 0,545 | 0,55 / 0,41 / 0,482 |
| S0 | 0,55 / 0,54 / 0,543 | 0,44 / 0,48 / 0,462 |
| S2 / S4 | 0,55 / 0,54 / 0,548 | 0,45 / 0,49 / 0,471 |
| S3 | 0,52 / 0,57 / 0,545 | 0,42 / 0,52 / 0,470 |
| S5 | 0,53 / 0,60 / 0,563 | 0,44 / 0,56 / 0,497 |

**Okuma:**
- A3 ailesinde `count_f1`, `block_f1`'den 0,07–0,09 düşük. A0'da fark küçük (0,02), çünkü A0 tek olayda daha az parçalı alarm üretiyor.
- Aşama 3 içindeki sıralama iki tanımda aynı (S5 > S2 = S4 > S3 > S0 = S1), dolayısıyla Aşama 3 kararı değişmiyor.
- Aşama 2'de A3-s2s-norm ile A3-s2s-hmean'in sırası yer değiştiriyor (2./4.). Bu iki varyant arasındaki fark metrik tanımına duyarlı.

**Etiketsiz arka plan alarmları** (enjeksiyonsuz kopyada aynı ayarlarla alarm veren bloklar; `anomaly_eval_audit_background.csv`):
- S0: kış 9 blok (0,098/gün), yaz 3 (0,025/gün).
- A0: kış 5, yaz 13.
- Enjekte kopyalardaki "yanlış" blokların çoğu bu arka plan alarmlarıyla aynı saatte. Örneğin S0: kış 9,0 bloğun 8,9'u, yaz 3,0 bloğun 3,0'ı.
- Bu yüzden enjeksiyonsuz veri **anomaliden arınmış kabul edilmiyor**. Bu bloklar ya gerçek ama etiketsiz olağandışı tüketim, ya da modelin o saatlerde sürekli yanılması olabilir; veri bunları ayırt etmiyor. Metriklerden çıkarılmıyorlar.
- Doğru söylenebilecek olan: gerçek precision, bu blokların kaçının gerçek anomali olduğuna bağlı olarak mevcut değerle daha yüksek bir üst sınır arasındadır.

**Kesinlik düzeltmesi:** Önceki raporda "düşük skorların büyük kısmı ölçüm ve enjeksiyondan geliyor" dedim. Bu bir **hipotez**:
- Kanıtla desteklenen kısım: (a) blok-precision A3'ü şişiriyor; (b) yanlış blokların neredeyse tamamı enjeksiyondan bağımsız.
- Kanıtlanamayan kısım: bu blokların gerçek anomali olduğu ve "kâhin" tavanlarının ulaşılabilir olduğu. Kâhin analizi (`anomaly_eval_audit_oracle.csv`) yalnız teşhistir, başarı tahmini değildir.

## 3. Yeniden üretilebilirlik

- `results/tables/repro_manifest.csv`: veri, config, kullanılan bütün ağırlıklar (F10-opt, F5-opt, A2, OOF modelleri, S-*) için SHA-256 ve paket sürümleri (Python 3.11.18, pandas 3.1.0, numpy 2.3.5, torch 2.15.0, scikit-learn 1.9.1, darwin).
- **Yaz-dev ağırlıkları (S-*):** Bu makinede yeniden eğitildi. Orijinallere erişim yok, bu yüzden **birebir tekrar üretildikleri iddia edilmiyor**.
  - Aşama 2 yaz tablolarıyla farklar: A3-s2s-hmean en fazla 0,046; A2 ve A3-ens-norm en fazla 0,21.
  - Bundan sonraki yeni karşılaştırmalar manifestodaki hash'lere sahip **ortak sabit ağırlıklarla** yapılmalı. Şeyda da aynı dosyaları kullanmalı.
- **Kış-dev:** A0, A1, A2, A3-s2s-norm, A3-s2s-hmean, A5 kayıtlı kopya F1'leriyle birebir aynı.
  - **A3-ens-norm farklı** (en fazla 0,017). Bu yöntem F5-opt ailesine bağlı; bu makinedeki F5-opt / F5-opt-oof dosyaları, Aşama 2 tablosunu üretenlerle aynı olmayabilir. Neden doğrulanamadı.
  - Final yöntem ve S0–S5 bu modeli kullanmıyor.
- **XGBoost:** 2 test macOS'ta segmentation fault veriyor (kök neden doğrulanmadı). F12 sonuçları kayıtlı tablolardan geliyor.

## 4. Güncel durum

**Tahmin (final, test bir kez kullanıldı):**
- F13-ens test MAE 0,406 kW; SN168'e göre +%26,0, SN24'e göre +%20,6.
- RMSE ve R²'de XGBoost daha iyi.
- Zaman hizalaması ve pencere sayıları doğrulandı. Değişiklik yok.

**Anomali:**
- Final (test): A3-s2s-norm olay F1 (blok tanımı) 0,384; A0 0,398. H3 (F1 ≥ 0,70) karşılanmadı.
- Geliştirme (dev, final değil): Aşama 3'te hiçbir aday kabul edilmedi, S0 korundu.
- Olay sayımlı tanımla bütün A3 değerleri daha düşük, sıralama büyük ölçüde aynı.

## 5. En fazla iki kısa deney (önerildi, çalıştırılmadı)

Önkoşullar:
- Metrik iki tanımla raporlanıyor (çözüldü).
- Ortak sabit ağırlıklar manifestoda; Şeyda'nın aynı dosyaları kullanması gerekiyor (açık).

Ortak kurallar:
- Tasarım koşmadan önce belgeye yazılır.
- Ortak baseline: S0 (Aşama 3 ayarı, manifestodaki ağırlıklar).
- Kış-dev + yaz-dev, orta senaryo.
- İki metrik birlikte raporlanır.
- Ağırlık ızgarası yok.
- Test ve val-hold yok.

| | E1 – sürekli işaretli artık | E2 – taban kuralı + duyarlılık |
|---|---|---|
| Hipotez | Geçmişe bakan 6 saatlik ort(z̄) pozitif kısmı, yaz desen ve seviye kaymasını saatlik A3'ten iyi ayırır. Gerekçe: yaz desen sapması 0,50 kW, h=1 MAE 0,34 kW | "Tüketim train 0,5. yüzdeliğinin (≈0,155 kW) altında ≥ 2 saat" kuralı drop'u yakalar. Enjeksiyon 0,05 kW olduğu için başarı tanım gereği olabilir |
| Değişiklik | S6 = max(A3', 1·sustained'); pencere ∈ {3, 6}; ölçek train OOF q99 | S0 ∨ taban kuralı (A6'nın k = 2'si) + tabanın üstünde bir düşüşle duyarlılık senaryosu (değer koşudan önce train dağılımından sabitlenir) |
| Süre | kod 45 dk + koşu ≤ 10 dk | kod 45 dk + koşu ≤ 10 dk |
| Kabul | Aşama 3 karar kuralı, **iki metrikte de**. Yaz desen recall'ü artmalı; kış seviye/desen recall kaybı ≤ 0,05; günlük yanlış alarm artışı ≤ 0,05 | Aynı kural. Ek olarak duyarlılık senaryosunda drop recall'ü raporlanır; orada sıfıra iniyorsa yöntem yalnız "tam kesinti dedektörü" diye sunulur |
| Durdurma | Süre dolarsa ya da iki mevsimde F1 farkı ≤ 0 ise bırakılır | Aynı. Temiz veride yeni arka plan alarmı çıkarsa ayrıca yazılır |

## 6. Yeniden çalıştırma komutları

```bash
pytest tests/ --ignore=tests/test_stage3_models.py          # macOS: XGBoost testleri çöküyor
pytest tests/test_stage3_models.py -k "not xgb and not xgboost"
python -m src.train_summer_dev                               # yaz-dev ağırlıkları (git dışı); hash: repro_manifest.csv
python -m src.anomaly_short_event                            # Aşama 3 (≈75 dk CPU)
python -m src.anomaly_eval_audit                             # iki metrik + arka plan + manifesto (≈1 dk)
```

## 7. Şeyda'ya kısa özet

> Şeyda, inceleme ve zorunlu düzeltmeler hazır (yerel dal `dilara/inceleme-zorunlu-duzeltmeler`, henüz push yok).
> - **pandas 3 zaman birimi hatası iki yerde vardı:** tahmin kayıtları ve ön işlemedeki sınır koruması. İkisi düzeltildi ve testlendi.
>   - Ölçtüm: kayıtlı hiçbir sonuç değişmiyor. Pencere ve gün sayıları tutuyor; doldurulan değerler aynı.
> - **Olay F1 kodu tanımına uygun, ama precision alarm bloklarını sayıyor.** Olay sayımlı F1'i ek metrik olarak ekledim.
>   - A3 ailesi 0,07–0,09 düşüyor.
>   - Aşama 3 sıralaması değişmiyor.
>   - A3-s2s-norm ile A3-s2s-hmean'in sırası metriğe duyarlı.
> - **Yanlış alarmların neredeyse hepsi enjeksiyonsuz veride de çıkıyor.** Bunları "etiketsiz arka plan alarmı" diye ayrı raporluyorum; gerçek anomali mi model hatası mı, ayırt edemiyoruz.
> - **Yaz-dev S-* ağırlıklarını bende yeniden eğittim, seninkilerle aynı değil.** Hash'ler `repro_manifest.csv`'de. Ortak sabit ağırlık kullanalım; ya benim dosyalarımı al ya da seninkileri gönder.
> - **Kış-dev'de A3-ens-norm'da da küçük fark var** (F5-opt ailesi dosyaları farklı olabilir).
