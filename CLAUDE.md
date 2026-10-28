# CLAUDE.md

Bu dosya, bu repoda çalışan Claude Code (ve diğer yapay zekâ asistanları) için kalıcı talimatlardır. Proje planı, deney listesi ve tüm teknik kararlar **`README.md`** içindedir. Herhangi bir iş yapmadan önce README'deki ilgili bölümü oku ve ona uy. README ile bu dosya çelişirse **dur ve kullanıcıya sor**.

## Proje bağlamı
- **Ders:** YMÜ425 – Uygulamalı Sinir Ağları (Fırat Üniversitesi, 2026–2027 Güz)
- **Konu:** Enerji tüketiminde anomali tespiti ve kısa vadeli (24 saat) talep tahmini
- **Ekip:** Dilara YILDIRIM (veri hattı, tahmin, demo) · Şeyda ALTIN (EDA, literatür, anomali, görselleştirme)
- **Veri:** UCI Individual Household Electric Power Consumption (dakikalık → saatlik)
- **Modeller:** LSTM, GRU (tahmin, 168 s → 24 s) · LSTM-Autoencoder + GRU artık dedektörü (anomali)
- **Referanslar:** Persistence, Seasonal Naive 24/168, Ridge, MLP · Hareketli z-skoru, Isolation Forest
- **Süre:** 4 gün. Basit, okunabilir ve savunulabilir çözüm > karmaşık çözüm.

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
5. **Test kümesi kutsaldır.** Hiperparametre seçimi, early stopping, eşik seçimi yalnızca doğrulama kümesiyle yapılır. `--split test` yalnızca kullanıcı açıkça "final değerlendirme" dediğinde çalıştırılır.
6. **Gelecek bilgisi yok.** Merkezli hareketli ortalama, `bfill`, gelecekteki değerlere bakan öznitelikler kullanılmaz. Hedef penceresi küme sınırını aşamaz.
7. **Anomali enjeksiyonu** ham kW üzerinde, ayrı kopyalarda, doğrulama için `seed=1`, test için `seed=2` ile yapılır. Orijinal veri asla yerinde değiştirilmez.

## Proje yapısı
```
configs/config.yaml     # TEK yapılandırma kaynağı (yollar, split tarihleri, pencere, hiperparametreler, seed)
data/raw/               # ham veri (git'e girmez)
data/processed/         # hourly.parquet
src/data/               # load, preprocess, split, windows, inject_anomalies
src/models/             # baselines, mlp, lstm, gru, autoencoder, anomaly_baselines
src/utils/              # seed, config, metrics, plots, exp_logger (stdlib logging ile çakışmasın diye)
notebooks/              # 01_eda.ipynb, 02_results_analysis.ipynb
src/train_forecast.py   src/train_autoencoder.py
src/evaluate_forecast.py src/evaluate_anomaly.py
tests/                  # pytest
app/streamlit_app.py
models/                 # ağırlıklar (.pt) + scaler.pkl
results/experiments.csv results/figures/ results/tables/
docs/                   # literatur.md, deney_gunlugu.md, oneri_formu/, rapor/, sunum/
```

## Komutlar
```bash
pip install -r requirements.txt
python -m src.data.load
python -m src.data.preprocess
python -m src.train_forecast --model {mlp,lstm,gru} --exp F5 [--feature_set A|B|C] [--lookback 168] [--seed 42]
python -m src.train_autoencoder --exp A2
python -m src.evaluate_forecast --split val|test
python -m src.evaluate_anomaly  --split val|test
pytest tests/
streamlit run app/streamlit_app.py
```

## Kodlama kuralları
- Python 3.10+, PyTorch, scikit-learn, pandas, numpy, matplotlib, streamlit, pyyaml, holidays.
- Hiperparametreler ve yollar **koda gömülmez**, `config.yaml`'dan okunur; CLI argümanları config'i ezer.
- Her betik başında `set_seed(seed)` (random, numpy, torch, cudnn deterministic).
- Cihaz: `cuda` varsa kullan, yoksa `cpu`. Kod CPU'da da makul sürede çalışmalı.
- Eğitim döngüsü açık yazılır (Lightning vb. kullanılmaz): epoch döngüsü, train/val loss, early stopping (tahminde val MAE izlenir, patience 8), `ReduceLROnPlateau` (factor 0.5, patience 3), gradient clipping (1.0), en iyi **val MAE** modeli `models/{exp_id}_best.pt`.
- Metrikler her zaman **orijinal ölçekte (kW)** hesaplanır (inverse transform). MAPE yalnız `y > 0.1 kW` için; ayrıca sMAPE.
- Doldurulmuş (`is_imputed=1`) hedef içeren pencereler değerlendirme metriklerinden çıkarılır.
- Her eğitim/değerlendirme koşusu `results/experiments.csv`'ye bir satır ekler: `exp_id, date, model, feature_set, lookback, hidden, layers, dropout, lr, batch, epochs_run, best_epoch, seed, val_mae, val_rmse, test_mae, test_rmse, train_time_s, note` (test sütunları final değerlendirmeye kadar boş).
- Figürler: `results/figures/{exp_id}_{ad}.png`, dpi 200; **her figürde eksen adı + birim, model adı, hangi küme (train/val/test) olduğu** başlıkta veya lejantta yazılı. Tablolar `results/tables/*.csv` (+ rapora kolay aktarım için `.md`).
- Yeni bir veri işleme adımı yazıldığında ilgili **pytest** testi de yazılır. Zorunlu testler:
  - train/val/test zaman aralıkları kesişmiyor
  - scaler'ın `mean_`'i yalnız eğitim verisinden geliyor
  - pencere şekilleri: X `[N, lookback, F]`, y `[N, 24]`
  - hedef penceresi küme sınırını aşmıyor
  - enjeksiyon aynı seed ile aynı etiketleri üretiyor, orijinal veriyi değiştirmiyor

## Çalışma akışı
1. Göreve başlamadan README'de ilgili bölümü ve günün planını (§3) kontrol et; görev planda yoksa kullanıcıya hangi iş paketine (İP1–İP6) ait olduğunu sor.
2. Küçük, test edilebilir adımlarla ilerle. Her adımdan sonra çalıştır ve çıktıyı göster.
3. Uzun eğitimlerden önce 1 epoch'luk "smoke test" yap.
4. Önemli bir karar alındığında (eşik değeri, öznitelik seti, hiperparametre) `docs/deney_gunlugu.md`'ye tarih + karar + gerekçe + ilgili exp_id ekle.
5. README'deki plan veya kararlar değişirse README'yi de güncelle ve değişikliği kullanıcıya bildir.
6. Commit mesajları Türkçe ve açıklayıcı. `main` dalı her zaman çalışır durumda kalmalı.

## Rapor, öneri formu ve sunum işleri
- Öneri formu kuralları README §6'dadır (Arial 9 pt; Özet 150–300 kelime, tek paragraf, tablo/denklem/atıf yok; Özgün Değer 200–750 kelime ve ≥350 boşluksuz karakter; Amaç ve Hedefler ayrı ve ölçülebilir; ≥10 IEEE kaynak, metinde [n] atıf; ≤20 sayfa). Metin yazarken bu sınırları say ve kontrol et.
- Hedeflerdeki sayılar **hedef**tir; gerçekleşmeden "elde edildi" diye yazılmaz.
- Rapor ve sunumdaki sayılar yalnızca `results/tables/` dosyalarından alınır. Rapor, kod ve sunum tutarlı olmalıdır.
- Literatürle kıyas yaparken veri seti, çözünürlük, ufuk, bölme yöntemi ve metrik farklarını belirt; koşullar farklıysa doğrudan "daha iyiyiz" deme.

## Kullanıcıyla iletişim
- Kullanıcılar Türkçe yazar; Türkçe cevap ver.
- Açıklamaları bir öğrenciye anlatır gibi yap: önce sezgi, sonra teknik detay.
- Belirsiz durumlarda (özellikle veri sızıntısı riski, test kümesi kullanımı, sonuçların yorumu) varsayım yapma, sor.
