# CLAUDE.md

Bu dosya, bu repoda çalışan Claude Code (ve diğer yapay zekâ asistanları) için kalıcı talimatlardır. Teknik kararlar, deney listesi, sonuçlar ve repo yapısı **`README.md`** içindedir. Herhangi bir iş yapmadan önce README'deki ilgili bölümü oku ve ona uy. README ile bu dosya çelişirse **dur ve kullanıcıya sor**.

## Proje bağlamı
- **Ders:** YMÜ425 – Uygulamalı Sinir Ağları (Fırat Üniversitesi, 2026–2027 Güz)
- **Konu:** Enerji tüketiminde anomali tespiti ve kısa vadeli (24 saat) talep tahmini
- **Ekip:** Dilara YILDIRIM (veri hattı, tahmin, demo; katkısı 2026-11-24'te tamamlandı) · Şeyda ALTIN (EDA, literatür, anomali, şebeke deneyi, görselleştirme; 2026-11-25'ten itibaren projeyi yürütüyor)
- **Veri setleri:** (1) UCI Individual Household Electric Power Consumption – hane, dakikalık → saatlik (ana); (2) UCI ElectricityLoadDiagrams20112014 – şebeke, 160 müşteri toplamı (aynı yöntemlerle ek deney); (3) **LEAD1.0** – gerçek etiketli bina anomalileri, ek doğrulama (`docs/lead_protokol.md`, `docs/lead_sonuclar.md`)
- **Final modeller (yalnız val ile seçildi, test bir kez kullanıldı):** tahmin **F13-ens** = 0,8 · F10-opt (Seq2Seq-LSTM) + 0,2 · F5-opt (LSTM), lookback 48, ufuk 24 · anomali **A3-s2s-norm** = F10-opt'un h=1 artığı / OOF std (saat × gün), w = 3
- **Karşılaştırma modelleri:** Persistence, Seasonal Naive 24/168, Ridge, MLP, GRU, CNN-LSTM, XGBoost · Hareketli z-skoru (A0), Isolation Forest (A1), LSTM-Autoencoder (A2), A5-mean-z; test sonrası A6 (deneysel)
- **LEAD final modeli (val'de §18.2 + §21 ile seçildi, test bir kez):** L1 behavior (LightGBM); sonuçlar `docs/lead_sonuclar.md`.
- **Durum:** Final sonuçlar donmuş (`docs/final_sonuclar.md`, `docs/hane_vs_sebeke.md`, `docs/lead_sonuclar.md`). Hane anomali geliştirmesi 2026-11-25'te donduruldu; LEAD val ve tek seferlik test 2026-11-26'da tamamlandı. Sıradaki iş: final rapor (`docs/rapor/DILARA_RAPOR_REHBERI.md`; belge giriş noktası `docs/README.md`). Basit, okunabilir ve savunulabilir çözüm > karmaşık çözüm.

## En önemli kural: öğrenciler her satırı savunacak
Ders yönergesine göre öğrenciler kodun her parçasını sözlü savunmada açıklamak zorunda; "kodu yapay zekâ yazdı" cevabı kabul edilmiyor. Bu yüzden:
- Kodu **kısa, okunabilir ve açık** yaz; gereksiz soyutlama, sihirli kütüphane kullanımı, tek satırlık karmaşık ifadeler yok.
- Her fonksiyonun başında **Türkçe docstring**: ne yapar, girdi/çıktı şekilleri (tensor shape), neden böyle yapıldığı.
- Önemli satırlarda kısa Türkçe yorum (özellikle sızıntı önlemleri ve tensor boyutları).
- Değişken ve fonksiyon adları İngilizce, yorumlar ve dokümantasyon Türkçe.
- Her görevin sonunda kullanıcıya **3–5 maddelik "savunmada sorulabilecekler"** notu ver: bu kodda ne yapıldı, neden, alternatif ne olurdu.
- Kullanıcı istemedikçe yeni kütüphane ekleme; eklersen `requirements.txt`'i güncelle ve nedenini söyle.

## Kesin yasaklar (bilimsel dürüstlük ve veri sızıntısı)
1. **Sonuç uydurma.** Çalıştırılmamış bir deneyin sonucunu, tahmini bir metriği veya "yaklaşık" bir sayıyı asla rapora, README'ye, slayta veya yoruma yazma. Rapordaki her sayı `results/` altındaki dosyalardan gelmeli.
2. **Kaynak uydurma.** Doğrulanamayan makale, yazar, DOI veya cilt/sayfa bilgisi yazma. Emin değilsen "doğrulanmalı" diye işaretle.
3. **Rastgele bölme yok.** Veri her zaman `configs/config.yaml`'daki tarihlere göre **kronolojik** bölünür. `shuffle` yalnızca eğitim DataLoader'ında, pencereler oluşturulduktan sonra serbesttir.
4. **Fit yalnızca eğitimde.** `StandardScaler`, imputasyon istatistikleri, Isolation Forest, Autoencoder yalnızca eğitim kümesiyle fit/eğitilir. Doğrulama/test'e sadece `transform`.
5. **Test kümesi kutsaldır.** Hiperparametre seçimi, early stopping, eşik seçimi yalnızca doğrulama kümesiyle yapılır. `--split test` / `--stage test` yalnızca kullanıcı açıkça "final değerlendirme" dediğinde çalıştırılır. Yeni yöntemler test kümesinde ve val-hold'da **ölçülmez** (`docs/gelistirme_protokolu.md` §1). Tek istisna, seçim yapmayan ve kullanıcının açıkça onayladığı regresyon kontrolüdür (`scripts/verify_frozen_results.py`).
6. **Gelecek bilgisi yok.** Merkezli hareketli ortalama, `bfill`, gelecekteki değerlere bakan öznitelikler kullanılmaz. Hedef penceresi küme sınırını aşamaz.
7. **Anomali enjeksiyonu** ham kW üzerinde, ayrı kopyalarda, `config.yaml`'daki `protocol.scenarios` seed listeleriyle yapılır (doğrulama ve test seed'leri farklı). Orijinal veri asla yerinde değiştirilmez.
8. **Donmuş sonuçlar değişmez.** `docs/final_sonuclar.md`, `docs/hane_vs_sebeke.md`, `results/tables/final_*`, `anomaly_protocol_*`, `results/eld/` ve `experiments.csv`'deki mevcut satırlar yeniden hesaplanıp değiştirilmez. Kod yapısı değişirse `scripts/verify_frozen_results.py` ile önce/sonra karşılaştırılır.

## Proje yapısı
```
configs/                  # config.yaml (TEK kaynak), eld.yaml, optuna_*_best.yaml
data/raw/ data/processed/ # ham ve işlenmiş veri (git'e girmez)
src/core/                 # config (PROJECT_ROOT, load_config), split, windows, inject_anomalies,
                          # metrics, anomaly_metrics, results_io, seed (set_seed, get_device), console
src/datasets/household/   # load, preprocess, weather (UCI hane)
src/datasets/eld/         # eld (UCI ELD, 160 müşteri toplamı)
src/datasets/lead/        # LEAD1.0: load (temizlik, site), eda, splits, features, evaluate
src/models/               # tahmin modelleri ve referanslar; anomali skorları (A0/A1, AE, A3 artık); lead_classifiers
src/pipelines/            # train_forecast, tune_optuna, train_oof, ensemble, seed_stability, evaluate_forecast,
                          # forecast_levels, verify_forecast_eval, train_autoencoder, evaluate_anomaly,
                          # anomaly_protocol, anomaly_dual, eld_run, final_tables, compare_hane_sebeke,
                          # lead_run (CLI), lead_models, lead_fe, lead_improve, lead_test
src/*.py, src/data/*.py   # GEÇİCİ: yalnız eski `python -m` komutları için yönlendirme (kod içermez)
experiments/archive/      # final sonuç üretmeyen deneyler + testleri (çalışır; çekirdek buradan import ETMEZ)
app/                      # Streamlit demo (streamlit_app.py, views/, *_services.py, ui.py, theme.py, style.css)
notebooks/                # 01_eda, 01_hane, 02_sebeke, 03_lead (eğitimsiz, kayıtlı sonuçlarla birebir kontrol)
scripts/                  # verify_frozen_results.py, lead_results_report.py, sunum_kontrol.py
tests/                    # pytest (pytest.ini: tests + experiments/archive/tests)
models/                   # ağırlıklar (.pt; git'e girmez, yedek repo dışında); ölçek istatistikleri checkpoint içinde
results/                  # experiments.csv, tables/, figures/, logs/, eld/, lead/
docs/                     # README.md (okuma sırası), deney_gunlugu, final sonuçlar, protokoller, literatür, LEAD,
                          # savunma_sorulari, oneri_formu/, rapor/, sunum/, archive/
```
- Yeni veri seti kodu `src/datasets/<ad>/` altına; veri setinden bağımsız adımlar `src/core/` altına.
- Yeni deneyler önce `experiments/` altında yazılır; final sonuç üretmeyen kod `src/`'ye girmez.
- Proje kökü yalnız `src.core.config.PROJECT_ROOT`'tan alınır; `Path(__file__).parents[k]` ile kök hesaplanmaz.

## Komutlar
```bash
pip install -r requirements.txt
python -m src.datasets.household.load
python -m src.datasets.household.preprocess
python -m src.pipelines.train_forecast --model {ridge,mlp,lstm,gru,seq2seq,cnn_lstm,xgboost,all} --exp <exp_id> [--feature_set A|B|C|W1|W2] [--lookback 48] [--seed 42]
python -m src.pipelines.train_autoencoder --exp A2
python -m src.pipelines.evaluate_forecast --split val|test     # test: yalnız final değerlendirme
python -m src.pipelines.anomaly_protocol [--split test]        # test: yalnız final değerlendirme
python -m src.pipelines.eld_run --stage val|test               # test: yalnız final değerlendirme
python scripts/verify_frozen_results.py snapshot --out <repo dışı> | compare <ref> <yeni>
pytest -q                                                       # çekirdek + arşiv; yalnız çekirdek: pytest tests/ -q
python -m streamlit run app/streamlit_app.py --server.address localhost
python -m src.pipelines.lead_run --stage <aşama>            # LEAD; --stage test yalnız --final-degerlendirme ile (yapıldı)
python scripts/sunum_kontrol.py [--ekran-goruntusu]        # sunum günü kontrolü (requirements-dev.txt)
```
Tam liste README §5'tedir.

## Kodlama kuralları
- Python 3.10+, PyTorch, scikit-learn, pandas, numpy, matplotlib, streamlit, pyyaml, holidays, xgboost, optuna, lightgbm (LEAD); geliştirme: playwright.
- Hiperparametreler ve yollar **koda gömülmez**, `config.yaml`'dan okunur; CLI argümanları config'i ezer.
- Her betik başında `set_seed(seed)` (random, numpy, torch, cudnn deterministic).
- Cihaz: `src.core.seed.get_device(args.device)` (CUDA varsa istenen cihaz, yoksa `cpu`). Kod CPU'da da makul sürede çalışmalı.
- Eğitim döngüsü açık yazılır (Lightning vb. kullanılmaz): epoch döngüsü, train/val loss, early stopping (tahminde val MAE izlenir, patience 8), `ReduceLROnPlateau` (factor 0.5, patience 3), gradient clipping (1.0), en iyi **val MAE** modeli `models/{exp_id}_best.pt`.
- Metrikler her zaman **orijinal ölçekte (kW)** hesaplanır (inverse transform). MAPE yalnız `y > 0.1 kW` için; ayrıca sMAPE.
- Doldurulmuş (`is_imputed=1`) hedef içeren pencereler değerlendirme metriklerinden çıkarılır.
- Her eğitim/değerlendirme koşusu `results/experiments.csv`'ye bir satır ekler: `exp_id, date, model, feature_set, lookback, hidden, layers, dropout, lr, batch, epochs_run, best_epoch, seed, val_mae, val_rmse, test_mae, test_rmse, train_time_s, note` (test sütunları final değerlendirmeye kadar boş).
- Figürler: `results/figures/{exp_id}_{ad}.png`, dpi 200; **her figürde eksen adı + birim, model adı, hangi küme (train/val/test) olduğu** başlıkta veya lejantta yazılı. Tablolar `results/tables/*.csv` (+ rapora kolay aktarım için `.md`). Tablo başlığı ve notları ("final sonuç değildir" vb.) **betikten** yazılır, elle eklenmez (`write_markdown(..., title=, note=)`).
- XGBoost çıktısı iş parçacığı sayısına bağlıdır; karşılaştırmalı koşularda `forecast.xgboost.n_jobs` sabitlenir.
- Yeni bir veri işleme adımı yazıldığında ilgili **pytest** testi de yazılır. Zorunlu testler:
  - train/val/test zaman aralıkları kesişmiyor
  - scaler'ın `mean_`'i yalnız eğitim verisinden geliyor
  - pencere şekilleri: X `[N, lookback, F]`, y `[N, 24]`
  - hedef penceresi küme sınırını aşmıyor
  - enjeksiyon aynı seed ile aynı etiketleri üretiyor, orijinal veriyi değiştirmiyor

## Çalışma akışı
1. Göreve başlamadan README'de ilgili bölümü ve durumu (§3) kontrol et; görev orada yoksa kullanıcıya hangi iş paketine (İP1–İP6, LEAD) ait olduğunu sor.
2. Küçük, test edilebilir adımlarla ilerle. Her adımdan sonra çalıştır ve çıktıyı göster.
3. Uzun eğitimlerden önce 1 epoch'luk "smoke test" yap.
4. Önemli bir karar alındığında (eşik değeri, öznitelik seti, hiperparametre, yapı değişikliği) `docs/deney_gunlugu.md`'ye tarih + karar + gerekçe + ilgili exp_id ekle.
5. README'deki kararlar veya yapı değişirse README'yi de güncelle ve değişikliği kullanıcıya bildir.
6. Commit mesajları Türkçe ve açıklayıcı. `main` dalı her zaman çalışır durumda kalmalı; iş `seyda/...` dallarında yapılır.

## Rapor, öneri formu ve sunum işleri
- Öneri formu kuralları README §6'dadır (Arial 9 pt; Özet 150–300 kelime, tek paragraf, tablo/denklem/atıf yok; Özgün Değer 200–750 kelime ve ≥350 boşluksuz karakter; Amaç ve Hedefler ayrı ve ölçülebilir; ≥10 IEEE kaynak, metinde [n] atıf; ≤20 sayfa). Metin yazarken bu sınırları say ve kontrol et.
- Hedeflerdeki sayılar **hedef**tir; gerçekleşmeden "elde edildi" diye yazılmaz.
- Rapor ve sunumdaki sayılar yalnızca `results/tables/` (ve `results/eld/tables/`) dosyalarından alınır. Rapor, kod ve sunum tutarlı olmalıdır.
- Literatürle kıyas yaparken veri seti, çözünürlük, ufuk, bölme yöntemi ve metrik farklarını belirt; koşullar farklıysa doğrudan "daha iyiyiz" deme.

## Kullanıcıyla iletişim
- Kullanıcılar Türkçe yazar; Türkçe cevap ver.
- Açıklamaları bir öğrenciye anlatır gibi yap: önce sezgi, sonra teknik detay.
- Belirsiz durumlarda (özellikle veri sızıntısı riski, test kümesi kullanımı, sonuçların yorumu) varsayım yapma, sor.
