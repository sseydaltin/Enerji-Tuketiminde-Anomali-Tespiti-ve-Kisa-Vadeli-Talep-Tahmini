# Kod haritası: sunumda gösterilen üç ekran

Sunumda canlı açılan üç sayfa (Talep Tahmini, Anomali Tespiti, Gerçek Etiketli Veri) için ekranda görünen her parçanın kodda nerede olduğu. Dosya yolları proje köküne göredir. Demo **hiçbir model eğitmez**: ağırlıklar ve skorlar kayıtlı dosyalardan okunur. Yedek ekran görüntüleri: `docs/sunum/yedek/` (`python scripts/sunum_kontrol.py --ekran-goruntusu`).

## Ortak parçalar (üç sayfada da)

| Ekranda | Kod | Açıklama |
|---|---|---|
| Sol menü ve sayfa adresleri | `app/streamlit_app.py` → `main` | Sayfaları `st.navigation` ile tanımlar, kenar çubuğuna ikonlu bağlantıları `st.page_link` ile çizer. |
| "Bu sayfada ne görüyorsunuz?" kutusu | `app/ui.py` → `page_intro` | Sayfa başlığı ve altındaki 1–2 cümlelik açıklama kutusu. |
| Üstteki veri seti seçimi (Hane / Şebeke) | `app/ui.py` → `dataset_strip` | Yatay seçim şeridi; seçim sayfalar arasında `st.session_state["dataset"]` ile korunur. |
| KPI kartları | `app/ui.py` → `kpi_cards` | En çok 3 kart; CSS ızgarasıyla dizilir, dar ekranda alt alta geçer. Sayıları çağıran sayfa hazırlar. |
| Tablolar (kısa başlık, ipucu) | `app/ui.py` → `compact_table` | Uzun sütun adlarını kısaltır, tam adı sütun ipucuna koyar; değerlere dokunmaz. |
| Renkler, yazı tipi, grafik şablonu | `app/theme.py`, `app/style.css` | Renk körlüğüne uygun palet ve tek Plotly şablonu; aynı model her grafikte aynı renk. |

## 1. Talep Tahmini

| Ekranda | Kod | Açıklama |
|---|---|---|
| Sayfanın tamamı | `app/views/forecast.py` → `render` | Veri setini seçer, kayıtlı modelleri yükler, günü seçtirir; kartları, grafiği ve dağılımı çizer. |
| Kayıtlı modellerin yüklenmesi (hane) | `app/services.py` → `load_f13_models` | `models/F10-opt_best.pt` ve `models/F5-opt_best.pt` ağırlıklarını okur; eğitim yapmaz. |
| Seçilebilir test günleri | `app/services.py` → `valid_test_days` | Önceki 48 saati ve hedef 24 saati doldurulmamış, tam ölçümlü test günleri. |
| Varsayılan gün ("medyana en yakın") | `app/services.py` → `median_day` | SN168'e göre iyileşmesi bütün test günlerinin medyanına en yakın gün; ilk görülen gün ne en iyi ne en kötü örnek olsun diye. |
| Rastgele / Kolay / Zor örnek düğmeleri | `app/views/forecast.py` → `_controls`; `app/services.py` → `best_decile_days`, `worst_decile_days` | Kolay ve zor örnek, topluluk MAE'sinin en iyi ve en kötü %10'luk diliminden seçilir (şebekede zor örnek sabit: Noel, `app/eld_services.py` → `HARD_DAY`). |
| Seçili günün tahmini | `app/services.py` → `predict_test_day` (hane), `app/eld_services.py` → `predict_eld_day` (şebeke) | Seçili günün önceki 48 saatinden 24 saatlik topluluk tahmini (0,8 · Seq2Seq + 0,2 · LSTM) ve SN168 referansı. |
| "Test geneli" satırı | `app/views/forecast.py` → `overall_result_text`; `app/results_services.py` → `load_results_bundle` | Kayıtlı final test tablosundan MAE ve SN168'e göre kazanç; seçili günden bağımsız asıl sonuç. |
| Üç KPI kartı | `app/views/forecast.py` → `_metric_cards`; `app/services.py` → `daily_total_error_pct` | Seçili günün MAE'si, SN168'e göre iyileşmesi ve günlük toplam tüketim hatası. |
| Gerçek tüketim ve tahmin grafiği | `app/views/forecast.py` → `_make_figure` | 48 saat geçmiş + 24 saat hedef; topluluk modeli ve kesikli SN168 çizgisi, tahmin başlangıcı dikey çizgi. |
| "Test günlerinin tamamı" histogramı | `app/views/forecast.py` → `_distribution_section`, `_distribution_figure`; `app/services.py` → `household_comparison` | Bütün test günlerinde SN168'e göre günlük iyileşmenin dağılımı; tek günün temsil gücünü gösterir. |
| Seq2Seq modeli | `src/models/seq2seq.py` → `Seq2SeqForecaster` | Encoder LSTM geçmişi özetler, decoder her hedef saatin önceden bilinen takvimini alır. |
| LSTM modeli | `src/models/lstm.py` → `LSTMForecaster` | Son gizli durumdan tek doğrusal katmanla 24 saatlik tahmin. |

## 2. Anomali Tespiti

| Ekranda | Kod | Açıklama |
|---|---|---|
| Sayfanın tamamı | `app/views/anomaly.py` → `render` | Kayıtlı eşikleri ve modelleri yükler, seçilen ayarlarla analizi çalıştırır, kartları, grafiği ve tabloları çizer. |
| Yöntem, anomali türü, "Gelişmiş ayarlar" | `app/views/anomaly.py` → `_controls` | Üstte yöntem ve tür; senaryo, şiddet ve tarih aralığı açılır bölümde. |
| Kayıtlı eşikler ve OOF std tabloları | `app/anomaly_services.py` → `load_anomaly_calibration` | Doğrulama kümesinde seçilmiş eşik/pencere ve train OOF artıklarından std tabloları; testte değişmez. |
| Sentetik anomali ekleme | `src/core/inject_anomalies.py` → `inject_anomalies`; `app/anomaly_services.py` → `scale_injection` | Gerçek serinin **kopyasına** seed'li olay ekler (ani yükselme, düşüş, seviye kayması, gece örüntüsü); orijinal veri değişmez. |
| Skorların hesaplanması | `app/anomaly_services.py` → `analyze_test_range` | Seçili dönemi skorlar: tahmin artığı, z-skoru vb. ve yumuşatma. |
| Tahmin artığı skoru (final yöntem) | `src/models/residual_detector.py` → `one_step_predictions`, `residual_std_table`, `normalized_scores` | 1 saat sonrası tahmin hatası, saat × gün OOF std'siyle normalize edilir; büyük artık = olağandışı. |
| Üç KPI kartı | `app/views/anomaly.py` → `_metric_cards`; `app/anomaly_services.py` → `event_outcomes` | Yakalanan olay / toplam, yanlış alarm ve eşik; olay tanımı raporlanan metriklerle aynıdır. |
| Alarm verilen saatler | `app/anomaly_services.py` → `flagged_hours` | Yumuşatılmış skorun eşiği aştığı saatler (grafikteki noktalar). |
| Grafik (7 gün, olaya odaklı) | `app/views/anomaly.py` → `_make_figure` | Üstte tüketim, altta skor ve eşik; enjekte olaylar bant olarak; görünüm seçili olayın çevresindeki 7 gün. |
| Olay tablosu (satır seçince grafik oraya gider) | `app/views/anomaly.py` → `_event_table` | Enjekte olaylar: başlangıç, tür, süre, en yüksek skor, yakalandı/kaçırıldı; seçim grafiğin aralığını belirler. |
| Olay düzeyi metrikler | `src/core/anomaly_metrics.py` → `runs`, `event_metrics` | Ardışık alarm saatleri tek olay sayılır; olay en az bir saatte alarm alırsa yakalanmış sayılır. |

## 3. Gerçek Etiketli Veri (LEAD)

| Ekranda | Kod | Açıklama |
|---|---|---|
| Sayfanın tamamı | `app/views/lead.py` → `render` | Seed ve bina seçimi, KPI'lar, grafik, bina sayımı ve açılır sonuç tablosu. |
| Test binalarının verisi ve kayıtlı skorlar | `app/lead_services.py` → `load_test_frame` | Temiz LEAD verisini, test binalarını ve tek seferlik final testin kayıtlı skorlarını (`results/lead/scores/`) hizalar; skor hesaplamaz. |
| Gerçek etiketlerin alt türleri (renkli noktalar) | `src/datasets/lead/features.py` → `evaluation_labels` | Her anomali saatini boşluk kenarı, donmuş okuma ya da davranışsal (other) olarak işaretler; yalnız değerlendirme içindir, model girdisi değildir. |
| Eşikler (kesikli çizgiler) | `app/lead_services.py` → `load_thresholds` | Val'de F1-max ile seçilmiş, testte aynen uygulanmış eşikler. |
| Varsayılan bina ("medyana en yakın") | `app/lead_services.py` → `building_f1`, `median_building` | Final modelin bina bazlı test F1'i medyana en yakın bina; yalnız görüntüleme seçimi. |
| Kolay / Zor örnek düğmeleri | `app/views/lead.py` → `_pick_building` | Final modelin F1'i en yüksek ve en düşük test binası. |
| Varsayılan iki haftalık aralık | `app/lead_services.py` → `busiest_window` | Binada en çok anomali saati içeren 14 gün; "Tüm dönem" ile bütün yıl. |
| Üç KPI kartı | `app/views/lead.py` → `final_kpis`; `app/lead_services.py` → `slice_results` | Final modelin test sonuçları (3 seed ort. ± std): davranışsal PR-AUC, genel PR-AUC, genel ROC-AUC. |
| Grafik (tüketim, etiketler, final alarmları, isteğe bağlı skorlar) | `app/views/lead.py` → `make_figure` | Tek grafik; "Model skorlarını göster" ile final ve seçilen karşılaştırma modelinin skor panelleri. |
| Bina sayımı tablosu | `app/lead_services.py` → `building_subtype_table` | Seçili binada her alt türde kaç saatin alarm aldığı ve normal saatlerdeki yanlış alarmlar. |
| Final modelin seçimi | `src/pipelines/lead_models.py` → `select_model_v2`, `contributes_neural` | Val'de önceden yazılmış kural: davranışsal PR-AUC → genel PR-AUC → gerçekten katkı veren sinir ağı → basitlik. |
| Final test (bir kez) | `src/pipelines/lead_test.py` → `run_test` | Eşik ve ayarlar val'den; test binaları seed başına bir kez skorlandı (`--final-degerlendirme` korumalı). |
| Metrikler | `src/datasets/lead/evaluate.py` → `evaluate`, `f1_max_threshold` | Dilim başına ROC-AUC, PR-AUC, P/R/F1 ve olay recall; eşik val'de F1-max. |

## Veri hazırlama ve bölme — **Dilara anlatacak**

| Adım | Kod | Açıklama |
|---|---|---|
| Ham hane verisini okuma | `src/datasets/household/load.py` → `load_raw_data` | UCI dakikalık verisini okur, eksik değerleri ("?") NaN yapar ve zaman damgasını kurar. |
| Saatliğe indirgeme | `src/datasets/household/preprocess.py` → `preprocess` | Dakikalık ölçümler saatlik ortalamaya (alt sayaç enerjileri toplama) çevrilir; dakikalarının yarısından fazlası eksik saatler geçersiz (NaN) sayılır. |
| Boşlukları doldurma | `src/datasets/household/preprocess.py` → `fill_missing_hours` | Kısa boşluklar aynı küme içinde iki gözlem arasında doğrusal enterpolasyonla, küme sınırına yakın ve uzun boşluklar `lag` saat önceki değerle doldurulur; bu saatler `is_imputed` ile işaretlenir ve doldurulmuş hedefli pencereler değerlendirmeden çıkarılır. |
| Takvim öznitelikleri | `src/datasets/household/preprocess.py` → `add_calendar_features` | Saat, gün, ay (sin/cos), hafta sonu ve tatil; geleceği önceden bilinen bilgilerdir. |
| Kronolojik bölme | `src/core/split.py` → `chronological_split` | Train / val / test tarih aralıkları `configs/config.yaml`'dan; rastgele bölme yok, aralıklar kesişmez. |
| Ölçekleme (yalnız train'e fit) | `src/core/windows.py` → `build_datasets`; `src/pipelines/train_forecast.py` → `prepare_forecast_data` | `StandardScaler` yalnız train verisine fit edilir, val/test'e yalnız uygulanır (sızıntı önlemi). |
| Pencereleme | `src/core/windows.py` → `make_windows` | 48 saatlik geçmişten 24 saatlik hedef pencereleri; hedef penceresi küme sınırını aşmaz. |
| Referans (naif) tahminler | `src/models/baselines.py` → `naive_forecasts` | Persistence, mevsimsel naif 24 ve 168 (SN168): sinir ağlarının geçmesi gereken taban. |
| Şebeke verisi | `src/datasets/eld/eld.py` → `build_hourly`, `fix_dst`, `active_clients` | 15 dakikalık müşteri verisi saatliğe çevrilir, yaz saati düzeltilir, 2011'de aktif müşteriler toplanır. |
| LEAD temizliği | `src/datasets/lead/load.py` → `clean` | Kaggle `train.csv` için tür dönüşümü, geçersiz/çift kayıt temizliği; adımlar günlüğe sayılarla yazılır. |
| LEAD bina bölmesi | `src/datasets/lead/splits.py` → `building_split` | 200 bina, seed'li olarak 120 / 40 / 40'a ayrılır; aynı bina iki parçada bulunmaz. |
