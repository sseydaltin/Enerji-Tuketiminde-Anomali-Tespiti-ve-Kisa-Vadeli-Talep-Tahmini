# Enerji Tüketiminde Anomali Tespiti ve Kısa Vadeli Talep Tahmini

**YMÜ425 – Uygulamalı Sinir Ağları · 2026–2027 Güz · Fırat Üniversitesi Yazılım Mühendisliği**

| Öğrenci | Numara | Ana sorumluluk |
|---|---|---|
| Dilara YILDIRIM | 230290018 | Veri hattı (yükleme, temizleme, bölme, pencereleme) + **Talep tahmini** (LSTM, GRU, baseline'lar) + Demo uygulaması |
| Şeyda ALTIN | 230290046 | Veri analizi (EDA) + Literatür + **Anomali tespiti** (LSTM-Autoencoder, anomali enjeksiyonu, baseline'lar) + Görselleştirme |

> ⚠️ **Yönergenin altın kuralı:** Görev dağılımı yapılabilir, ancak **her iki üye de projenin tamamından sorumludur** (Yönerge §1, §17). Savunmada Dilara'ya anomali, Şeyda'ya tahmin sorusu gelebilir. Bu yüzden her gün sonunda **30 dakikalık "çapraz anlatım"** zorunludur (bkz. §3).

Bu dosya projenin **tek kaynak kılavuzudur**. Plan, teknik kararlar, deney listesi, öneri formu, rapor, sunum ve savunma hazırlığı burada. Claude Code ile geliştirme kuralları `CLAUDE.md` dosyasındadır.

---

## İçindekiler
1. [Temel kararlar (özet)](#1-temel-kararlar-özet)
2. [Görev dağılımı](#2-görev-dağılımı)
3. [4 günlük plan](#3-4-günlük-plan)
4. [Teknik kılavuz](#4-teknik-kılavuz)
5. [Repo yapısı ve komutlar](#5-repo-yapısı-ve-komutlar)
6. [Proje Öneri Formu kılavuzu](#6-proje-öneri-formu-kılavuzu)
7. [Proje raporu iskeleti](#7-proje-raporu-iskeleti)
8. [Sunum kılavuzu ve puanlama eşlemesi](#8-sunum-kılavuzu-ve-puanlama-eşlemesi)
9. [Savunma hazırlığı – soru bankası](#9-savunma-hazırlığı--soru-bankası)
10. [Kurallar: veri sızıntısı, hazır kod, yapay zekâ kullanımı](#10-kurallar)
11. [Kaynak havuzu (IEEE)](#11-kaynak-havuzu-ieee)
12. [Teslim öncesi son kontrol listesi](#12-teslim-öncesi-son-kontrol-listesi)
13. [Riskler ve B planları](#13-riskler-ve-b-planları)

---

## 1. Temel kararlar (özet)

| Konu | Karar | Gerekçe (savunmada söylenecek) |
|---|---|---|
| **Veri seti** | **UCI Individual Household Electric Power Consumption** (ana). ElectricityLoadDiagrams20112014 yalnızca "gelecek çalışma"da anılır. | Tek hane, ~4 yıl, dakikalık ölçüm (~2,07 milyon satır), alt sayaç (sub-metering) bilgisi var. Saatliğe indirgenince (~34 bin saat) CPU/Colab'da 4 günde eğitilebilir. ElectricityLoadDiagrams 370 müşteri × 15 dk ile 4 gün için fazla büyük ve müşteri başlangıçları (sıfır dolgular) ek temizlik gerektirir. |
| **Zaman çözünürlüğü** | Dakikalık → **saatlik** (ortalama kW = saatlik kWh) | Gürültüyü azaltır, kısa vadeli talep tahmininin pratik ölçeği; eğitim süresi makul. |
| **Görev 1 – Tahmin** | Geçmiş **168 saat** (1 hafta) → gelecek **24 saat** (çok çıktılı, doğrudan tahmin) | Günlük ve haftalık periyodikliği yakalar; "gün öncesi" talep tahmini pratik bir senaryodur. |
| **Görev 2 – Anomali** | **LSTM-Autoencoder** (yeniden yapılandırma hatası) + **GRU tahmin artığı** (residual) yaklaşımı | İki görevi tek bir hatta bağlar; özgün değerin bir parçası. |
| **Anomali etiketi sorunu** | Veri setinde gerçek anomali etiketi **yok** → doğrulama ve test kümelerine **kontrollü sentetik anomali enjeksiyonu** (bilinen etiket) + gerçek veride en yüksek skorlu olayların nitel incelemesi | Precision/Recall/F1/PR-AUC hesaplanabilmesi için zorunlu. Sınırlılık olarak raporlanır. |
| **Veri bölme** | **Zamansal (kronolojik)** bölme, rastgele karıştırma yok | Zaman serisinde geleceği görmek veri sızıntısıdır (Yönerge §7). |
| **Ana modeller** | LSTM, GRU (tahmin) · LSTM-Autoencoder (anomali) | Konu başlığında istenen aile; kapı mekanizmaları uzun bağımlılıkları öğrenir. |
| **Referans yöntemler** | Tahmin: Persistence, Seasonal Naive (24 s / 168 s), Ridge regresyon (klasik ML), MLP (daha basit sinir ağı). Anomali: Hareketli z-skoru, Isolation Forest | Yönerge §9: en az bir anlamlı referans + daha basit ağ + klasik ML. |
| **Araçlar** | Python 3.10+, PyTorch, scikit-learn, pandas, matplotlib, Streamlit, Google Colab (gerekirse GPU) | |
| **Tekrarlanabilirlik** | Tek `configs/config.yaml`, sabit `seed=42` (+ en iyi model 3 seed: 42, 7, 2026), deney günlüğü `results/experiments.csv` | Yönerge §10 son madde. |

---

## 2. Görev dağılımı

### 2.1 Rol tablosu

| İş paketi | Sorumlu | Destek / gözden geçiren |
|---|---|---|
| İP1 – Literatür taraması, kaynakça (≥10 IEEE) | **Şeyda** | Dilara (tahmin makaleleri) |
| İP2 – Veri yükleme, temizleme, saatliğe indirgeme, bölme, ölçekleme, pencereleme | **Dilara** | Şeyda (EDA ile çapraz kontrol) |
| İP2b – Keşifsel veri analizi (EDA) ve figürler | **Şeyda** | Dilara |
| İP3 – Tahmin: baseline'lar + MLP + LSTM + GRU, eğitim, ayar, ablasyon | **Dilara** | Şeyda (kod incelemesi) |
| İP4 – Anomali: enjeksiyon, z-skor, Isolation Forest, LSTM-AE, artık tabanlı dedektör, eşik seçimi | **Şeyda** | Dilara (artık dedektörü final tahmin modelini kullanır: F10-opt / F13-ens) |
| İP5 – Değerlendirme, hata analizi, karşılaştırma tabloları | İkisi birlikte (kendi hattını) | Çapraz okuma |
| İP6 – Streamlit demo | **Dilara** | Şeyda (anomali sayfası) |
| Öneri formu | Şeyda: Özgün Değer + Kaynaklar · Dilara: Yöntem · İkisi: Amaç/Hedefler, Özet (en son) | |
| Rapor | Bölüm bazında §7'deki tabloya göre | Her bölüm diğer kişi tarafından okunur |
| Sunum | §8.3'teki konuşmacı dağılımına göre | Birlikte prova |

### 2.2 Çapraz hâkimiyet kuralları (zorunlu)
- Her gün sonunda **30 dk çapraz anlatım**: Dilara, Şeyda'ya o gün yazdığı kodu ve kararları anlatır; sonra tersi. Anlatan kişi ekran paylaşmadan, kâğıt üzerinde açıklayabilmelidir.
- Her pull request / commit diğer kişi tarafından okunur. "Okudum, anladım" onayı olmadan `main`'e birleştirilmez.
- Gün 4'te karşılıklı **sahte savunma**: §9'daki soru bankasından kişi kendi alanı **dışındaki** sorulara cevap verir.

---

## 3. 4 günlük plan

> Her gün ~9–10 saat varsayılmıştır. **Gün sonu kontrol noktaları (✅) karşılanmadan bir sonraki güne geçilmez**; karşılanmıyorsa §13'teki B planı devreye girer.
>
> Öneri formu ara sınav haftasından önce teslim edileceği için **Gün 2 sonunda bitmiş olmalıdır** ve somut ilerleme (EDA figürleri, ilk baseline sonuçları) içermelidir (Yönerge §14 "Önemli" notu).

### Gün 1 – Kurulum, veri, EDA, literatür

| Saat | Dilara | Şeyda |
|---|---|---|
| 09:00–09:45 | **Birlikte:** repo oluştur, klasör yapısı (§5), `requirements.txt`, `.gitignore` (`data/raw/`, `models/*.pt`), `configs/config.yaml`, `src/utils/seed.py` | |
| 09:45–12:30 | `src/data/load.py`: UCI'dan indir, `;` ayırıcı, `?` → NaN, `Date+Time` → datetime index, tip dönüşümleri. Ham veri özet tablosu (satır sayısı, tarih aralığı, eksik oranı) | Literatür taraması: §11 havuzundan başla, her makale için **literatür matrisi** (`docs/literatur.md`: veri seti, model, metrik, sonuç, güçlü/zayıf yön, bize katkısı) |
| 13:30–16:00 | `src/data/preprocess.py`: eksik veri stratejisi (§4.3), saatliğe indirgeme, `other_energy` türetme, takvim öznitelikleri, `is_imputed` bayrağı | `notebooks/01_eda.ipynb`: §4.2 EDA kontrol listesi (eksik veri ısı haritası, günlük/haftalık/yıllık profiller, dağılımlar, korelasyon, ACF) |
| 16:00–18:00 | `src/data/split.py` (kronolojik bölme) + `src/data/windows.py` (168→24 pencere, ölçekleyici yalnız eğitime fit) + birim testleri: kümeler arası zaman çakışması yok | EDA figürlerini `results/figures/eda_*.png` olarak kaydet; gözlemleri yaz (tatil dönemleri, uzun boşluklar, mevsimsellik) |
| 18:00–19:00 | `src/models/baselines.py`: Persistence + Seasonal Naive (24/168) → yalnız doğrulama sonucu (test için açıkça `--split test`) | Öneri formu: **Özgün Değer** taslağı (literatür boşluğu, araştırma sorusu, hipotez) |
| 19:00–19:30 | **Çapraz anlatım** | |

✅ **Gün 1 bitiş kriterleri:** temiz saatlik veri `data/processed/hourly.parquet`; split tarihleri ve örnek sayıları tablosu; ≥6 EDA figürü; ilk baseline doğrulama MAE/RMSE; ≥10 adaylı literatür matrisi.

### Gün 2 – Modeller ve ilk eğitimler + Öneri formu

| Saat | Dilara | Şeyda |
|---|---|---|
| 09:00–11:00 | Ridge (lag öznitelikleri) + MLP baseline; `src/utils/metrics.py` (MAE, RMSE, MAPE*, sMAPE, R², ufka göre hata) | `src/data/inject_anomalies.py`: 4 anomali tipi (§4.9), tohumlu, etiket maskesi döndürür; doğrulama (seed=1) ve test (seed=2) kopyaları |
| 11:00–13:00 | `src/models/lstm.py`, `src/models/gru.py` + `src/train_forecast.py` (early stopping, ReduceLROnPlateau, grad clip, en iyi val modeli kaydet, `experiments.csv`'ye yaz) | Hareketli z-skoru + Isolation Forest baseline'ları (`src/models/anomaly_baselines.py`) |
| 14:00–16:30 | LSTM ve GRU varsayılan hiperparametrelerle eğitim (§4.7), öğrenme eğrileri | `src/models/autoencoder.py` (LSTM-AE) + `src/train_autoencoder.py` (yalnız eğitim kümesinde, normal veri varsayımı) |
| 16:30–18:30 | Öneri formu: **Yöntem ve Teknikler** (ne/neden/nasıl, değişkenler, iş paketleri) | Öneri formu: **Kaynaklar** (≥10 IEEE, doğrulanmış) + Özgün Değer son hali |
| 18:30–19:30 | **Birlikte:** Amaç ve Hedefler, Proje Özeti (en son), Ekip tablosu, Arial 9 pt biçim kontrolü → **Öneri formu teslime hazır** | |
| 19:30–20:00 | **Çapraz anlatım** | |

✅ **Gün 2 bitiş kriterleri:** LSTM ve GRU eğitilmiş, val sonuçları baseline'larla tek tabloda; LSTM-AE eğitilmiş; enjeksiyon modülü çalışıyor; **öneri formu tamam**.

### Gün 3 – Deneyler, değerlendirme, hata analizi

| Saat | Dilara | Şeyda |
|---|---|---|
| 09:00–12:00 | Hiperparametre ayarı **yalnız doğrulama** üzerinde (§4.8 küçük ızgara: hidden, lookback, dropout) | Eşik seçimi doğrulama kümesinde (F1-max ve yüzdelik 95/99/99.5 duyarlılık analizi); GRU artık (residual) dedektörü |
| 12:00–13:00 | Ablasyon: öznitelik setleri A/B/C | Anomali tiplerine göre recall |
| 14:00–16:00 | En iyi LSTM ve GRU: 3 seed ile yeniden eğit → ortalama ± std; **test kümesinde tek seferlik** son değerlendirme | **Test kümesinde tek seferlik** son değerlendirme: Precision, Recall, F1, ROC-AUC, PR-AUC, karmaşıklık matrisi |
| 16:00–18:00 | Hata analizi: ufka göre hata, saat/gün/mevsim bazlı hata, en kötü 10 gün ve nedenleri, gerçek vs tahmin grafikleri | Hata analizi: kaçırılan/yanlış alarm örnekleri, gerçek (enjeksiyonsuz) test verisinde en yüksek skorlu 10 olay ve yorumları |
| 18:00–19:00 | `results/tables/` altında son tablolar; §4.10 figür listesi tamam | |
| 19:00–19:30 | **Çapraz anlatım** (sonuçların yorumu üzerine) | |

✅ **Gün 3 bitiş kriterleri:** §4.6'daki tüm deneyler `experiments.csv`'de; son karşılaştırma tabloları; tüm figürler eksen adı, birim, küme adıyla kaydedilmiş; hata analizi notları yazılmış.

### Gün 4 – Demo, rapor, sunum, prova

| Saat | Dilara | Şeyda |
|---|---|---|
| 09:00–11:30 | `app/streamlit_app.py`: Tahmin sayfası + model yükleme | Rapor: Giriş, Literatür, Veri, Anomali bölümleri |
| 11:30–13:00 | Streamlit: Anomali sayfası (Şeyda'nın fonksiyonlarıyla) + uçtan uca test | Rapor: Anomali sonuçları, hata analizi, sınırlılıklar |
| 14:00–16:00 | Rapor: Yöntem (veri hattı, tahmin modelleri, hiperparametre tablosu), Tahmin sonuçları | Sunum slaytları (§8.2 plan) |
| 16:00–17:00 | **Birlikte:** Rapor–kod–sunum tutarlılık kontrolü (§12); README'deki komutlar temiz ortamda çalışıyor mu? | |
| 17:00–19:00 | **Birlikte:** 2 tam sunum provası (süre tutarak) + canlı demo provası | |
| 19:00–20:00 | **Sahte savunma:** §9 soru bankası, herkes kendi alanı **dışındaki** sorulara cevap verir | |

✅ **Gün 4 bitiş kriterleri:** Çalışan demo; rapor taslağı tam (şablon yayımlanınca aktarılacak); sunum süresi içinde; iki kişi de §9'daki soruların tamamını cevaplayabiliyor.

> **Not:** Rapor şablonu ve tarihler henüz yayımlanmadı (Yönerge §14, §15.1, §15.3, §16). Şablon gelince içerik §7 iskeletinden aktarılır.

---

## 4. Teknik kılavuz

### 4.1 Veri seti
**UCI – Individual Household Electric Power Consumption** (Hebrail & Berard).
- Fransa'da (Sceaux) tek bir hanenin Aralık 2006 – Kasım 2010 arası **dakikalık** ölçümleri, ~2.075.259 satır, ~%1,25 eksik değer (`?` ile işaretli).
- Sütunlar: `Date`, `Time`, `Global_active_power` (kW), `Global_reactive_power` (kW), `Voltage` (V), `Global_intensity` (A), `Sub_metering_1` (mutfak, Wh), `Sub_metering_2` (çamaşır odası, Wh), `Sub_metering_3` (su ısıtıcı + klima, Wh).
- Türetilen: `other_energy = Global_active_power*1000/60 − Sub_1 − Sub_2 − Sub_3` (alt sayaçlarca ölçülmeyen aktif enerji, Wh/dk).
- Lisans: CC BY 4.0 → raporda kaynak gösterilecek (Yönerge §5).
- ⚠️ Yukarıdaki sayıları raporda kullanmadan önce **kendi EDA çıktınızla doğrulayın**; rapordaki her sayı kodunuzdan gelmeli.

**Neden bu veri seti uygun (Yönerge §5 kriterleri):** probleme uygunluk (gerçek konut tüketimi), yeterli miktar (saatlikte ~34 bin örnek), kalite sorunları bilinir ve yönetilebilir, zamansal bölmeye uygun, LSTM/GRU giriş yapısına uygun, donanımla gerçekleştirilebilir. **Sınırlılıklar:** tek hane (genellenebilirlik düşük), gerçek anomali etiketi yok, hava durumu verisi yok.

### 4.2 EDA kontrol listesi (Yönerge §6)
- [ ] Satır sayısı, tarih aralığı, örnekleme aralığı tutarlılığı (eksik dakikalar var mı?)
- [ ] Sütun bazında eksik oranı + **zaman içinde eksik veri ısı haritası** (uzun kesintileri bul)
- [ ] Tekrarlı zaman damgası kontrolü
- [ ] Tanımlayıcı istatistikler, histogramlar (çarpıklık → log dönüşüm gerekebilir)
- [ ] Ortalama günlük profil (saat × hafta içi/sonu), haftalık profil, aylık/mevsimsel eğilim
- [ ] Isı haritası: saat × haftanın günü ortalama tüketim
- [ ] Alt sayaç payları (ne kadar tüketim hangi alana ait)
- [ ] Korelasyon matrisi (`Global_active_power` ↔ `Global_intensity` neredeyse doğrusal → hedefle birlikte girdi olarak kullanmak sızıntı değil ama bilgi açısından gereksiz olabilir; gerekçelendirin)
- [ ] ACF/PACF: 24 ve 168 gecikmede tepe beklenir → lookback seçiminin gerekçesi
- [ ] Olağandışı dönemler: tatil, uzun süre evde olmama, sıfıra yakın tüketim günleri (gerçek veri üzerindeki nitel anomali analizi için not alın)

### 4.3 Ön işleme (sıra önemli)
1. `?` → NaN, sayısal tiplere dönüşüm, datetime index, sıralama, tekrarları sil.
2. Dakikalık eksik index'i tamamla (`asfreq('1min')`) → eksikler görünür olsun.
3. **Saatliğe indirgeme:** güç/voltaj/akım için `mean`, alt sayaçlar ve `other_energy` için `sum`. Bir saatteki dakikaların **%50'sinden fazlası eksikse** o saat NaN.
4. **Eksik saatler:** ≤3 saatlik boşluk → boşluğun iki yanındaki gözlemler arasında doğrusal interpolasyon; daha uzun boşluk → bir hafta önceki aynı saatin değeri; her durumda `is_imputed=1`. İnterpolasyon her veri kümesinde ayrı uygulanır; küme sınırının ±3 saatindeki eksikler hafta-önce yöntemiyle doldurulur.
   - Doldurulmuş hedef içeren pencereler **değerlendirme metriklerinden çıkarılır** (eğitimde kalabilir). Raporda belirtin.
   - Kısa boşluk interpolasyonu boşluğun sonraki gözlemini kullanır; bu nedenle tümüyle nedensel değildir. Sınır çevresinde veri kümeleri arası bilgi geçişini önlemek için hafta-önce yöntemi kullanılır.
5. **Takvim öznitelikleri:** saat ve haftanın günü için sin/cos kodlama, ay için sin/cos, `is_weekend`, `is_holiday` (Fransa resmi tatilleri, `holidays` paketi).
6. **Ölçekleme:** `StandardScaler` **yalnızca eğitim kümesine fit**, doğrulama/test'e `transform` (Yönerge §6 ve §7 sızıntı uyarısı). Ölçekleyici `models/scaler.pkl` olarak saklanır; demo da aynısını kullanır.
7. İsteğe bağlı: hedefe `log1p` (EDA'da sağa çarpıklık görülürse); metrikler **her zaman orijinal ölçekte (kW)** raporlanır.

### 4.4 Eğitim / doğrulama / test bölmesi
**Kronolojik bölme** (tarihler `config.yaml`'da):

| Küme | Tarih aralığı (öneri) | Yaklaşık oran |
|---|---|---|
| Eğitim | 2006-12-16 → 2009-09-30 | ~%71 |
| Doğrulama | 2009-10-01 → 2010-03-31 | ~%13 |
| Test | 2010-04-01 → 2010-11-26 | ~%16 |

- Rastgele karıştırma **yok**. Her kümedeki örnek (saat ve pencere) sayılarını tabloya yazın.
- Pencereler küme sınırını aşmaz: doğrulama penceresinin girdisi eğitimin son 168 saatini **kullanabilir** (geçmiş bilgi, sızıntı değil), ama **hedefi** asla başka kümeye taşmaz. Bunu raporda açıklayın.
- Test kümesi yalnızca Gün 3 öğleden sonra, **bir kez** kullanılır. Hiperparametre ve eşik seçimleri doğrulama ile yapılır.
- Sınırlılık: test dönemi kış aylarını içermiyor → mevsimsel genelleme sınırlılığı olarak tartışılır. (Alternatif: rolling-origin değerlendirme, "gelecek çalışma".)

### 4.5 Pencereleme
- Tahmin: `X[t-168:t]` (öznitelikler) → `y[t:t+24]` (`Global_active_power`). Adım (stride) 1 saat.
- Autoencoder: 24 saatlik pencere (`[24, n_features]` veya yalnız hedef), stride 1. Nokta skoru = o saati içeren tüm pencerelerin o saatteki yeniden yapılandırma hatalarının ortalaması.

### 4.6 Modeller ve deney listesi

**Tahmin (ufuk 24 saat):**

| Kod | Model | Rolü |
|---|---|---|
| F0 | Persistence (`ŷ[t+h] = y[t]`) | En basit referans |
| F1 | Seasonal Naive 24 s (`ŷ[t+h] = y[t+h−24]`) | Günlük periyot referansı |
| F2 | Seasonal Naive 168 s | Haftalık periyot referansı |
| F3 | Ridge regresyon (geçmiş 168 saat içinden config'teki lag'ler: 1–24, 48, 168) | **Klasik ML referansı** |
| F4 | `MLPRegressor` (düzleştirilmiş pencere, 2 gizli katman) | **Daha basit sinir ağı** |
| F5 | **LSTM** (2 katman, son gizli durum → Linear(24)) | Ana model |
| F6 | **GRU** (aynı yapı) | Ana model / LSTM ile karşılaştırma |
| F7 | Ablasyon: öznitelik seti A (yalnız hedef) / B (+takvim) / C (+alt sayaçlar, voltaj, reaktif güç) | Öznitelik etkisi |
| F8 | Lookback 48 vs 168 | Bağlam uzunluğu etkisi |
| F9 | En iyi model 3 seed (ortalama ± std) | Sonuçların kararlılığı |

#### F3/F4 baseline yöntemi ve doğrulama sonucu

- F3 Ridge, yalnızca `Global_active_power` geçmişinden config'te tanımlı 26
  gecikmeyi (1–24, 48, 168 saat) kullanır; 24 saati tek adımda tahmin eder.
- F4 scikit-learn `MLPRegressor`, aynı geçmiş 168 saatlik tüketim penceresini
  düzleştirir; `[128, 64]` gizli katmanlarıyla 24 saatlik çıktı üretir.
  Model train verisinde fit edilir; dahili rastgele validation ayırması kapalıdır.
- İki modelin ölçekleyicileri yalnız train tüketiminden fit edilir. Eğitim
  hedefleri train pencerelerinden, değerlendirme hedefleri yalnız val
  pencerelerinden gelir. Doldurulmuş hedef içeren val pencereleri çıkarılır;
  eğitim hedefleri train ölçekleyicisiyle normalize edilir ve val tahminleri
  metriklerden önce kW ölçeğine geri çevrilir.
- Metrikler `src/utils/metrics.py` üzerinden MAE, RMSE, MAPE* (`y > 0.1 kW`),
  sMAPE ve R² olarak hesaplanır; ayrıca her 24 tahmin adımı için ayrı sonuç
  yazılır. MAPE* için eşik üstü hedef yoksa değer `NaN`; sMAPE paydası sıfır
  olan (gerçek ve tahminin ikisinin de sıfır olduğu) terim sıfır kabul edilir.
- Çalıştırma: `python -m src.models.forecast_baselines`. Özet
  `results/tables/baseline_results.csv`, ufuk sonuçları
  `results/tables/forecast_horizon_metrics.csv`, deney satırları
  `results/experiments.csv` dosyasına kaydedilir. Bu komut test kümesini
  değerlendirmez.

2026-10-29 tarihinde 4.213 geçerli val penceresinde elde edilen güncel
sonuçlar (`results/tables/baseline_results.csv`):

| Model | Val penceresi | MAE (kW) | RMSE (kW) |
|---|---:|---:|---:|
| Persistence | 4.213 | 0.9598 | 1.2537 |
| Seasonal Naive 24 | 4.213 | 0.6913 | 0.9898 |
| Seasonal Naive 168 | 4.213 | 0.6216 | 0.9053 |
| Ridge (F3) | 4.213 | 0.6053 | 0.8054 |
| MLPRegressor (F4) | 4.213 | 0.5774 | 0.8098 |

F4 `MLPRegressor` koşusu `max_iter=60` sonunda durdu; scikit-learn'in
convergence warning'i kayıt edildi. Bu baseline'da validation ile epoch seçimi
yoktur. Önceki prototipte kullanılan PyTorch MLP'nin değerleri bu F4
koşusundan farklıdır; karşılaştırmada güncel, scikit-learn F4 satırı kullanılmalı.
Test metrikleri bu aşamada hesaplanmamıştır.

#### F5/F6 recurrent model eğitimi

`src/train_forecast.py --model all` tek komutta F3/F4 baseline'ları ile F5
LSTM ve F6 GRU'yu çalıştırır. Recurrent modeller config'in varsayılan B
özellik setini, 168 saatlik girdiyi ve 24 saatlik çıktıyı kullanır; varsayılan
hidden=64, layers=2, dropout=0.2 ayarlarıyla eğitilir. Epoch train/val MSE
geçmişi `results/tables/F5_training_history.csv` ve
`F6_training_history.csv` dosyalarına; öğrenme eğrileri
`results/figures/F5_learning_curve.png` ve `F6_learning_curve.png` dosyalarına
yazılır. Model seçimi val MAE ile yapılır; scheduler val MSE'yi izler.
Checkpoint'ler `models/F5_best.pt` ve `models/F6_best.pt` konumuna kaydedilir.
Ufuk bazındaki MAE, RMSE, MAPE*, sMAPE ve R² değerleri
`results/tables/forecast_horizon_metrics.csv` içindedir. Güncel sonuçlar
aşağıda 4.213 doldurulmamış validation penceresi üzerinden verilmiştir
(`results/tables/baseline_results.csv`):

| Model | MAE (kW) | RMSE (kW) | MAPE* (%) | sMAPE (%) | R² |
|---|---:|---:|---:|---:|---:|
| Ridge (F3) | 0.6053 | 0.8054 | 75.0230 | 52.2753 | 0.2208 |
| MLPRegressor (F4) | 0.5774 | 0.8098 | 56.9711 | 50.8297 | 0.2124 |
| LSTM (F5) | 0.5308 | 0.7354 | 79.3068 | 88.7618 | 0.4421 |
| GRU (F6) | 0.5298 | 0.7299 | 77.2918 | 89.8308 | 0.4505 |

F5, 30 epoch sonunda val MAE ile seçilen 22. epoch checkpoint'ini kullandı
(eğitim süresi 570.4 sn). F6, 26 epoch sonunda seçilen 18. epoch
checkpoint'ini kullandı (672.5 sn). MPS aygıtı paylaşımlı bellek ayırma
hatası verdiğinden GRU koşusu aynı config/seed ile CPU'da tamamlandı. Bu
donanım farkı not edilmelidir; test kümesi kullanılmadı. LSTM ve GRU öğrenme
eğrileri ile epoch kayıpları yukarıda belirtilen PNG/CSV dosyalarındadır.
MAPE* yalnızca gerçek hedef `0.1 kW` üstündeyken hesaplanır; ufuk adımı
başına metrikler karşılaştırmalı tabloya ek olarak ufuk CSV'sinde bulunur.

**Anomali tespiti:**

| Kod | Yöntem | Skor |
|---|---|---|
| A0 | Hareketli z-skoru (168 s pencere, aynı saat-günü bazlı) | \|z\| |
| A1 | Isolation Forest (pencere istatistikleri + takvim) | anomaly score |
| A2 | **LSTM-Autoencoder** (encoder LSTM → gizli vektör → decoder LSTM) | Yeniden yapılandırma MSE |
| A3 | **Tahmin artığı dedektörü** (2026-10-31 kararı: final tahmin modeli F13-ens; 1 adım ileri tahmin için F10-opt seq2seq, ikinci varyant F13-ens = 0,8·F10-opt + 0,2·F5-opt; ağırlıklar yalnız train ile eğitildi) | \|y − ŷ\| (kW) ve \|y − ŷ\| / std(saat × gün, yalnız train artıkları) |
| A4 | Eşik duyarlılığı: val'de F1-max vs yüzdelik 95 / 99 / 99.5 | — |

LSTM-AE bilinçli olarak **yalnız eğitim kümesinde** ve "çoğunlukla normal" varsayımıyla eğitilir. Eğitim verisinde gerçek anomaliler olabileceği (kontaminasyon) bir sınırlılıktır.

### 4.7 Eğitim ayarları (varsayılan, raporda tablo olarak verilecek)

| Parametre | LSTM / GRU (tahmin) | LSTM-AE (anomali) | Gerekçe |
|---|---|---|---|
| Girdi | 168 × n_features | 24 × n_features | ACF tepeleri |
| Gizli boyut / katman | 64 / 2 | Encoder 64→32, decoder simetrik | Veri büyüklüğüne uygun kapasite |
| Dropout | 0.2 (katmanlar arası) | 0.1 | Aşırı öğrenmeye karşı |
| Çıkış aktivasyonu | Doğrusal (regresyon) | Doğrusal | Ölçeklenmiş sürekli hedef |
| İç aktivasyonlar | tanh / sigmoid (kapılar) | aynı | LSTM/GRU standart |
| Kayıp | MSE (alternatif: Huber) | MSE | RMSE ile uyumlu; Huber aykırılara dirençli |
| Optimizer | Adam, lr = 1e-3 | Adam, lr = 1e-3 | Standart, hızlı yakınsama |
| Weight decay | 1e-5 | 0 | Hafif L2 |
| Batch size | 64 | 128 | Bellek/kararlılık dengesi |
| Epoch | en fazla 60 | en fazla 60 | Early stopping belirler |
| Early stopping | **val MAE (kW)**, patience = 8 | val loss, patience = 8 | Tahminde model seçim kriteriyle aynı metrik |
| LR scheduler | ReduceLROnPlateau (factor 0.5, patience 3) | aynı | |
| Gradient clipping | max norm 1.0 | 1.0 | RNN'de patlayan gradyan |
| Model seçim kriteri | **En düşük val MAE** (orijinal ölçek) | En düşük val yeniden yapılandırma kaybı; eşik val F1 ile | |
| Seed | 42 (F9'da 42, 7, 2026) | 42 | Tekrarlanabilirlik |

Her deney `results/experiments.csv`'ye şu sütunlarla eklenir: `exp_id, date, model, feature_set, lookback, hidden, layers, dropout, lr, batch, epochs_run, best_epoch, seed, val_mae, val_rmse, test_mae, test_rmse, train_time_s, note` (`test_*` yalnız final değerlendirmede doldurulur).

### 4.8 Hiperparametre ayarı (yalnız doğrulama kümesinde)
Küçük ve gerekçeli bir ızgara: `hidden ∈ {32, 64, 128}`, `dropout ∈ {0.1, 0.3}`, `lookback ∈ {48, 168}` (F8). Toplam ≤ 12 koşu/model. Yönerge §9: amaç çok model değil, **anlamlı** karşılaştırma.

### 4.9 Anomali enjeksiyonu (değerlendirme protokolü)
Hem doğrulama hem test kümesinin **kopyasına** (seed'ler aşağıda; ilk protokolde val seed=1, test seed=2), saatlerin ~%2–3'ünü etkileyecek şekilde:

| Tip | Uygulama | Gerçek dünya karşılığı |
|---|---|---|
| Ani tepe (spike) | 1–2 saat, değer × U(2, 4) | Arızalı cihaz, kısa devre |
| Düşüş / kesinti (drop) | 2–6 saat, değer → ~0.05 kW | Elektrik kesintisi, sayaç arızası |
| Seviye kayması (level shift) | 12–48 saat, değer × 1.5 | Unutulan ısıtıcı, kaçak tüketim |
| Desen anomalisi (pattern) | Bir günün gece saatleri gündüz seviyesine çekilir | Olağandışı davranış, yetkisiz kullanım |

- Etiketler saat düzeyinde (0/1). Enjeksiyon **ölçeklemeden önce ham kW** üzerinde yapılır; ölçekleyici yine eğitimden gelir.
- **Eşik doğrulama kümesinde seçilir**, test kümesine aynen uygulanır (sızıntı yok).
- **Çok seed'li val protokolü (A-1):** val'e 5 ayrı kopyada seed ∈ {1, 11, 21, 31, 41} ile enjeksiyon yapılır; her yöntemin tek eşiği 5 kopyanın birleşiminde F1-max ile seçilir, metrikler kopya başına hesaplanıp ortalama ± std verilir. Test tek kopyadır (seed=2).
- **Skor yumuşatma (A-3):** her yöntemin skoruna yalnız geçmişe bakan hareketli ortalama (pencere 1/3/6 saat) uygulanır; pencere val'de F1-max eşiğindeki ortalama F1 ile seçilir (eşitlikte küçük pencere), test'e aynen uygulanır.
- **Şiddet taraması (A-4):** seviye kayması ×{1,5; 2; 3} ve spike ×{2; 4; 6}; en iyi 3 yöntem, val'de seçilmiş eşik ve pencere sabit, olay recall'ü ayrı tabloda (`anomaly_val_severity.csv`); ana sonuçları değiştirmez.
- **Çok ufuklu artık ve birleşim (A-5):** A-5a — her saat için F10-opt'un h = 1…24 ufuklu tahminlerinin (başlangıç t−h) her biri kendi train std'siyle (saat × gün) normalize edilir; skor h ∈ {1, 6, 12, 24} ortalaması, 24 ufkun ortalaması veya maksimumu. A-5b — A0 ile en iyi A3 varyantı, enjeksiyonsuz val skor dağılımına göre z-skoru veya rank (ampirik dağılım) ölçeğine getirilip ortalaması / maksimumu alınır (`anomaly_val_combination.csv`).
- **A-5 seçim kuralı (yerini aşağıdaki final protokol aldı):** val F1 (5 seed ortalaması) en yüksek yöntem referanstır; ona karşı eşleştirilmiş seed farkının |ortalaması| ≤ std'si olan yöntemler gürültü düzeyinde sayılır ve aralarından en basit olanı seçilir (basitlik: A0 < A3 tek model h=1 < A3 çok ufuklu < A3 topluluk < birleşim). Seçim: **A3-s2s-norm** (F10-opt, |h=1 hatası| / std saat×gün, w = 3) — gerekçe `docs/deney_gunlugu.md`, A-5.
- **Final protokol (test öncesi sabitlendi; ayrıntı `docs/final_test_protokolu.md`):** train/OOF → val-dev (2009-10-01 – 2009-12-31; pencere, eşik, yöntem seçimi) → val-hold (2010-01-01 – 2010-03-31; bağımsız ölçüm, hiçbir seçim yok) → test. A3'ün normalize skorunda std, tüm train ile eğitilmiş modelin train artıklarından değil, 3 katlı zamansal CV'nin **OOF (kat dışı) artıklarından** hesaplanır; skor yine tüm train ile eğitilmiş F10-opt / F13-ens ile üretilir.
- **Anomali tanımı – iki senaryo:** anomali, o saatin **olağan oynaklığının dışında kalan sapma** olarak tanımlanır; bu yüzden ana senaryoda büyüklük, train'den saat × haftanın günü için hesaplanan tüketim std'sine (σ) göre verilir. Literatürde sentetik anomali büyüklüğünü verinin std'si cinsinden vermek yaygın bir pratiktir [KAYNAK YER TUTUCUSU – doğrulanmalı].

| Tip | Ana senaryo (seçim burada) | Zor senaryo (yukarıdaki tablo, aynı eşiklerle ayrıca raporlanır) |
|---|---|---|
| Spike | 1–2 saat, değer + U(3, 5)·σ(saat, gün) | 1–2 saat, değer × U(2, 4) |
| Drop | 2–6 saat, ~0,05 kW; yalnız gerçek değeri > 1,0 kW olan saatte başlar | 2–6 saat, ~0,05 kW |
| Seviye kayması | 12–48 saat, değer × U(2; 2,5) (olay başına tek çarpan) | 12–48 saat, değer × 1,5 |
| Desen | aynı | aynı |

- **Seed'ler (yalnız enjeksiyon yerini/büyüklüğünü belirler):** ana — val-dev 1, 11, …, 91; val-hold 3, 13, …, 93; test 2, 12, 22, 32, 42. Zor — val-dev 5, 15, …, 95; val-hold 7, 17, …, 97; test 4, 14, 24, 34, 44. Test bağımsızlığı seed farkıyla değil **zaman ayrımıyla** sağlanır.
- **Ana metrik: olay F1.** Olay recall'ü = en az bir saati işaretlenen olay oranı; olay precision'ı = gerçek bir anomaliyle örtüşen alarm segmentleri / tüm alarm segmentleri. Eşik, val-dev ana senaryonun 10 kopyasında birleşik olay F1'ini en büyük yapan değerdir (birleşik skor yüzdeliklerinde 300 noktalı ızgara); pencere ∈ {1, 3, 6} ortalama olay F1 ile seçilir. Nokta P/R/F1, PR-AUC ve ROC-AUC tabloda kalır.
- **Final yöntem kuralı:** val-dev ana senaryoda ortalama olay F1'i en yüksek yöntem referanstır; ona karşı eşleştirilmiş seed farkı |ortalama| ≤ 2·std/√n (standart hata tabanlı; revizyon 1 — ilk sürümdeki |ort| ≤ std seed sayısı arttıkça gevşiyordu) olan yöntemler gürültü düzeyinde sayılır ve aralarından en basiti seçilir (basitlik: A0 < A1 < A2 < A3-s2s-norm < A3-s2s-hmean < A3-ens-norm < A5-mean-z). **Final anomali yöntemi: A3-s2s-norm** (w = 3, olay-F1 eşiği 3,558). Komut: `python -m src.anomaly_protocol` (val), final değerlendirmede `--split test`.
- **Olay bazlı metrikler:** bir olay (aynı tipten ardışık etiketli saatler) en az bir saati işaretlenince yakalanmış sayılır (olay recall'ü); olay precision'ı = gerçek bir anomaliyle kesişen alarm bloklarının oranı. Bu kural yalnız olay metriklerinde kullanılır; nokta metriklerine *point-adjust* (yakalanan olayın tüm saatlerini doğru saymak) uygulanmaz, çünkü point-adjust rastgele skorlara bile yüksek F1 verebildiği için eleştirilmektedir (Kim vd., AAAI 2022 – kaynak doğrulanmalı).
- Raporlanan: Precision, Recall, F1, ROC-AUC, PR-AUC (dengesiz sınıf → PR-AUC daha bilgilendirici), karmaşıklık matrisi, **tip bazında recall**.
- Ek nitel analiz: enjeksiyonsuz test verisinde en yüksek skorlu 10 olay → gerçekte ne olmuş olabilir (tatil, ev boş, mevsim geçişi)?

### 4.10 Metrikler ve neden
- **Tahmin:** MAE (kW, yorumlanması kolay), RMSE (büyük hataları cezalandırır, tepe talep için önemli), R², **MAPE** (sıfıra yakın değerlerde patlar → yalnız `y > 0.1 kW` için hesaplanır veya **sMAPE** ile desteklenir; bunu açıklayın). Ufka göre (h=1…24) MAE eğrisi.
- **Anomali:** Precision (yanlış alarm maliyeti), Recall (kaçırılan anomali maliyeti), F1, ROC-AUC, PR-AUC, karmaşıklık matrisi. Ek olarak olay bazlı recall/precision (§4.9); val'de 5 enjeksiyon seed'inin ortalaması ± std.
- Yönerge §11: yalnız tek metrik raporlanmaz; sınıf dengesizliğinin etkisi tartışılır.

### 4.11 Figür listesi (her figürde eksen adı + birim + model adı + **hangi küme** olduğu yazılı)
1. Eksik veri zaman haritası
2. Saat × gün tüketim ısı haritası
3. Ortalama günlük profil (hafta içi/sonu) ve aylık eğilim
4. ACF (24/168 tepeleri)
5. Bölme şeması (zaman ekseninde train/val/test renkli)
6. LSTM & GRU eğitim/doğrulama kayıp eğrileri (aşırı/yetersiz öğrenme yorumu)
7. Tahmin modelleri karşılaştırma çubuk grafiği (test MAE/RMSE, hata çubuklu)
8. Ufka göre MAE eğrisi (h=1…24)
9. Gerçek vs tahmin: iyi bir hafta + kötü bir hafta
10. Saat / gün / ay bazlı hata dağılımı
11. Anomali skoru zaman serisi + eşik çizgisi + enjekte anomaliler
12. Anomali karmaşıklık matrisleri (A0–A3)
13. ROC ve PR eğrileri (tüm anomali yöntemleri aynı grafikte)
14. Anomali tipine göre recall çubuk grafiği
15. Yanlış alarm / kaçırılan örnek galerisi

### 4.12 Hata analizi soruları (Yönerge §13, rapora cevaplanarak girecek)
- Hangi saatlerde/günlerde/mevsimlerde hata yüksek? Neden (ani cihaz kullanımı, tatil)?
- Ufuk uzadıkça hata nasıl artıyor?
- LSTM ile GRU arasındaki fark anlamlı mı (seed std'sine göre)?
- Hangi anomali tipi zor yakalanıyor? (Beklenti: seviye kayması ve desen anomalisi z-skorda zayıf)
- Train–val–test performans farkı var mı?
- Sonuçlar literatürle nasıl kıyaslanır? (Aynı veri setini kullanan çalışmalarda bölme, çözünürlük ve ufuk farklı olabilir → doğrudan sayı kıyası yapmadan önce koşulları belirtin.)

### 4.13 Demo uygulaması (Streamlit) – Yönerge §15.3, §16
- **Sayfa 1 – Talep Tahmini:** test döneminden bir tarih seç → son 168 saat + LSTM/GRU/Seasonal Naive 24 saatlik tahmin vs gerçek, o örnek için MAE.
- **Sayfa 2 – Anomali Tespiti:** tarih aralığı seç, "sentetik anomali enjekte et" anahtarı, yöntem seçimi (LSTM-AE / GRU artık / IF) → skor grafiği, eşik, işaretlenen saatler tablosu.
- **Sayfa 3 – Model Bilgisi:** mimari özeti, hiperparametre tablosu, test metrikleri.
- Kaydedilmiş ağırlıklar (`models/*.pt`) ve `scaler.pkl` yüklenir; eğitim sunumda yapılmaz (Yönerge §16 izin veriyor). Sunumda kodun **nasıl yüklendiği ve tahmin ürettiği** gösterilecek.

---

## 5. Repo yapısı ve komutlar

```
enerji-anomali-tahmin/
├── README.md                  # bu kılavuz
├── CLAUDE.md                  # Claude Code için geliştirme kuralları
├── requirements.txt
├── .gitignore                 # data/raw/, *.pt (büyükse), __pycache__, .ipynb_checkpoints
├── configs/config.yaml        # yollar, split tarihleri, pencere, hiperparametreler, seed
├── data/
│   ├── raw/                   # household_power_consumption.txt (git'e girmez)
│   └── processed/             # hourly.parquet, split indeksleri
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_results_analysis.ipynb
├── src/
│   ├── data/   load.py · preprocess.py · split.py · windows.py · inject_anomalies.py
│   ├── models/ baselines.py · forecast_baselines.py · mlp.py · lstm.py · gru.py · autoencoder.py · anomaly_baselines.py
│   ├── utils/  seed.py · config.py · metrics.py · plots.py · exp_logger.py
│   ├── train_forecast.py      # python -m src.train_forecast --model all
│   ├── train_autoencoder.py
│   ├── evaluate_forecast.py   # --split val|test
│   └── evaluate_anomaly.py    # --split val|test
├── tests/                     # split çakışması, ölçekleyici sızıntısı, pencere şekilleri
├── app/streamlit_app.py
├── models/                    # en iyi ağırlıklar + scaler.pkl
├── results/
│   ├── experiments.csv
│   ├── figures/
│   └── tables/
└── docs/
    ├── literatur.md           # literatür matrisi
    ├── deney_gunlugu.md       # günlük karar ve gözlem notları
    ├── yontem_ve_teknikler_dilara.md  # tahmin yöntemleri ve deney protokolü
    ├── oneri_formu/
    ├── rapor/
    └── sunum/
```

**Kurulum** (Python 3.10+; proje kökünde):
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/Mac: source .venv/bin/activate)
# NVIDIA GPU varsa önce CUDA'lı torch (yoksa bu satırı atlayın, CPU sürümü kurulur):
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pytest tests/                     # kurulum testleri geçmeli
```

**Temel komutlar** (README'de güncel tutulacak):
```bash
pip install -r requirements.txt
python -m src.data.load            # ham veriyi indir/oku
python -m src.data.preprocess      # data/processed/hourly.parquet
python -m src.train_forecast --model all  # Ridge + MLPRegressor + LSTM + GRU, yalnız val
# MPS belleği yetersizse açık CPU fallback'i:
python -m src.train_forecast --model gru --device cpu
python -m src.train_forecast --model lstm # yalnız LSTM, yalnız val
python -m src.train_forecast --model gru  # yalnız GRU, yalnız val
python -m src.train_autoencoder --exp A2
python -m src.models.baselines          # yalnız val
python -m src.models.baselines --split test
python -m src.models.forecast_baselines # Ridge + MLP, yalnız val
python -m src.evaluate_forecast --split test
python -m src.evaluate_anomaly  --split test
pytest tests/
streamlit run app/streamlit_app.py
```

**Git düzeni:** `main` her zaman çalışır; kişisel dallar `dilara/...`, `seyda/...`; commit mesajları Türkçe ve açıklayıcı ("GRU eğitim betiği: early stopping eklendi").

---

## 6. Proje Öneri Formu kılavuzu

Formdaki ve öneri formu yönergesindeki **tüm kurallar** (çakışan yerlerde ikisini birden karşılayan seçenek seçildi):

### 6.1 Genel biçim
- [ ] Yazı tipi **Arial, 9 punto** (tablolar dahil)
- [ ] Ekler hariç **en fazla 20 sayfa**
- [ ] Akademik, resmi dil; teknik terimler doğru (ilk geçişte Türkçe + İngilizce: "uzun kısa süreli bellek (LSTM)")
- [ ] Kaynaklar **IEEE**, metin içinde **[1], [2]** şeklinde numaralı atıf, kaynakça numara sırasıyla
- [ ] Formdaki italik yönlendirme metinleri teslimden önce silinir

### 6.2 Bölüm bölüm

**Proje Başlığı** – kısa, açık, özgün. Öneri:
> *"LSTM, GRU ve LSTM-Autoencoder ile Hane Düzeyinde Kısa Vadeli Elektrik Talep Tahmini ve Tüketim Anomalisi Tespiti"*

**Proje Ekibi** – tablo: Ad Soyad | Numara | Projedeki Görevi (§2.1'den kısa: "Veri hattı, talep tahmini modelleri, demo uygulaması" / "Veri analizi, literatür, anomali tespiti modelleri, görselleştirme").

**Proje Özeti** – ⚠️ çakışma: yönerge 25–450 kelime, form **150–300 kelime** → **150–300 kelime, tek paragraf**. Tablo, denklem, kaynak gösterimi **yok**. Amaç, yöntem, beklenen sonuçlar, katkı (özgün değer, yönetim/iş planı, yaygın etki) yer almalı. **En son yazılır.**
**Anahtar Kelimeler:** 3–5, virgülle ayrılmış. Öneri: *Enerji tüketimi, talep tahmini, anomali tespiti, LSTM, otokodlayıcı*

**Projenin Özgün Değeri** – ⚠️ form **200–750 kelime**, yönerge **≥350 karakter (boşluksuz)** → 400–700 kelime hedefleyin. İçermesi gerekenler:
1. **Önem:** Akıllı şebeke, talep–arz dengesi, konut tüketiminin yüksek oynaklığı, anomalilerin (kaçak, arıza, verimsizlik) maliyeti → kaynakla destekle.
2. **Literatür ve eleştirel boşluk:** Literatürde tahmin ve anomali çoğunlukla **ayrı** ele alınıyor; gerçek etiket eksikliği nedeniyle anomali yöntemleri çoğu zaman **nicel olarak karşılaştırılamıyor**; bazı çalışmalar rastgele bölme veya tüm veriye ölçekleme gibi **sızıntıya açık** protokoller kullanıyor; hane düzeyi seri, toplulaştırılmış yüke göre çok daha gürültülü. (Bu iddiaları yalnızca okuduğunuz makalelere dayandırın.)
3. **Katkı/yenilik:** (i) Aynı veri hattında tahmin + anomali; (ii) tahmin artığı tabanlı ve yeniden yapılandırma tabanlı anomali tespitinin karşılaştırılması; (iii) tip bazlı, kontrollü sentetik anomali enjeksiyonu ile ölçülebilir değerlendirme; (iv) sızıntısız kronolojik protokol + çok seed'li raporlama. "İlk kez" gibi doğrulanamayan iddialardan kaçının.
4. **Araştırma soruları ve hipotezler** (açık yazılacak):
   - **AS1:** Kapı mekanizmalı tekrarlayan ağlar (LSTM, GRU), hane düzeyi saatlik tüketimin 24 saatlik tahmininde mevsimsel naif ve klasik regresyon yöntemlerinden daha düşük hata üretir mi? **H1:** LSTM/GRU, test RMSE'sinde Seasonal Naive (168 s) ve Ridge'den daha düşüktür.
   - **AS2:** LSTM ve GRU arasında doğruluk ve eğitim maliyeti açısından anlamlı fark var mı? **H2:** GRU, benzer doğruluğu daha az parametre ve daha kısa sürede sağlar.
   - **AS3:** Yeniden yapılandırma tabanlı LSTM-AE, istatistiksel ve klasik ML anomali yöntemlerinden daha yüksek PR-AUC elde eder mi? **H3:** LSTM-AE ve tahmin artığı dedektörü, özellikle seviye kayması ve desen anomalilerinde z-skoru ve Isolation Forest'tan daha yüksek recall sağlar.

**Amaç ve Hedefler** – ≥350 karakter. **Amaç (genel yön)** ve **hedefler (ölçülebilir alt çıktılar)** **ayrı** yazılır; her hedef için *nasıl kontrol edileceği* ve *başarı ölçütü* belirtilir.
> **Amaç:** Hane düzeyi elektrik tüketim verisinden zamana bağlı örüntüleri öğrenen derin öğrenme modelleriyle hem 24 saatlik talep tahmini yapan hem de olağandışı tüketimi tespit eden, sızıntısız ve tekrarlanabilir bir sistem geliştirmek ve bunu referans yöntemlerle bilimsel olarak karşılaştırmak.

| Hedef | Ölçülebilir çıktı | Kontrol yöntemi | Başarı ölçütü (hedef, sonuç değil) |
|---|---|---|---|
| H1 – Veri hazırlama | Temiz saatlik veri, belgelenmiş eksik veri stratejisi, kronolojik bölme | Birim testleri (küme çakışması = 0, ölçekleyici yalnız eğitimde fit) | Tüm testler geçer, eksik oranı ve örnek sayıları raporlanır |
| H2 – Talep tahmini | LSTM ve GRU, 24 saat ufuk | Test MAE/RMSE/sMAPE, 3 seed ort. ± std | En iyi modelin test RMSE'si Seasonal Naive (168 s)'e göre ≥ %10 düşük |
| H3 – Anomali tespiti | LSTM-AE ve tahmin artığı dedektörü | Enjekte anomalili test kümesinde P/R/F1/PR-AUC | En iyi derin yöntemin PR-AUC'si z-skor ve IF'den yüksek; F1 ≥ 0.70 |
| H4 – Analiz | Hata analizi, tip bazlı recall, sınırlılıklar | Rapordaki tablo ve figürler | §4.11 figür listesinin tamamı |
| H5 – Uygulama | Çalışır Streamlit demo | Sunumda canlı gösterim | Kayıtlı modelle örnek girdiden tahmin ve anomali skoru üretimi |

⚠️ Bu sayılar **hedef**tir. Raporda "elde edildi" ancak gerçekten elde edilirse yazılır (Yönerge §19). Ulaşılamazsa nedeni tartışılır; bu da bilimsel bir sonuçtur.

**Yöntem ve Teknikler** – her bileşen için **ne / neden / nasıl** cevaplanır:
- **Veri seti:** kaynak, kapsam, öznitelikler, lisans, neden uygun, sınırlılıklar (§4.1)
- **Veri toplama/işleme:** §4.3 adımları + sızıntı önlemleri
- **Değişkenler** (öneri formu yönergesi açıkça istiyor):
  - *Bağımlı:* gelecek 24 saatin `Global_active_power` değerleri (tahmin); saat düzeyinde anomali etiketi/skoru (anomali)
  - *Bağımsız:* geçmiş 168 saatlik tüketim, alt sayaçlar, voltaj, reaktif güç, takvim öznitelikleri
  - *Kontrol edilen:* veri bölmesi, seed, pencere uzunluğu, eğitim bütçesi (aynı koşullar tüm modellerde)
- **Araştırma tasarımı:** deneysel, karşılaştırmalı; §4.6 deney tablosu
- **Modeller:** LSTM, GRU, LSTM-AE (mimari şema + neden), referanslar
- **Eğitim/test ayrımı:** kronolojik, oranlar ve tarihler, neden rastgele değil
- **Değerlendirme ve istatistik:** §4.10 metrikleri; 3 seed ortalama ± standart sapma; anomali için eşik val'de seçilir
- **Araçlar:** Python, PyTorch, scikit-learn, pandas, Streamlit, Google Colab, Git/GitHub
- **Ön çalışma:** Gün 1–2'de yapılan EDA bulguları (eksik oranı, periyodiklik, figürler) ve **ilk baseline doğrulama sonuçları** → "somut ilerleme" kanıtı (Yönerge §14)
- **İş paketleri** (hangi İP hangi hedefe hizmet eder):

| İP | İçerik | Hedef | Sorumlu | Süre |
|---|---|---|---|---|
| İP1 | Literatür taraması | Tümü | Şeyda | Gün 1 |
| İP2 | Veri hazırlama ve EDA | H1 | Dilara, Şeyda | Gün 1 |
| İP3 | Tahmin modelleri ve referanslar | H2 | Dilara | Gün 2–3 |
| İP4 | Anomali modelleri, enjeksiyon, referanslar | H3 | Şeyda | Gün 2–3 |
| İP5 | Değerlendirme, hata analizi | H4 | İkisi | Gün 3 |
| İP6 | Demo, rapor, sunum | H5 | İkisi | Gün 4 |

**Kaynaklar** – form **en az 10 bilimsel kaynak** istiyor, IEEE formatında, numaralı, metin içinde atıflı. §11'deki havuzu kullanın ama **her birini Google Scholar / DOI ile doğrulayın**.

---

## 7. Proje raporu iskeleti
(Resmî şablon yayımlanınca aktarılacak. İçerik Yönerge §15.1'i birebir karşılar.)

| # | Bölüm | Yazan | İçerik |
|---|---|---|---|
| 1 | Giriş | Şeyda | Problem, önem, amaç, kapsam, araştırma soruları |
| 2 | Literatür | Şeyda | Literatür matrisi özeti, boşluk, yöntem seçimine etkisi |
| 3 | Veri seti ve hazırlama | Dilara | Kaynak, EDA bulguları, ön işleme adımları ve gerekçeleri, bölme tablosu (tarih + örnek sayısı) |
| 4 | Yöntem | Dilara (tahmin), Şeyda (anomali) | Mimariler, referanslar, enjeksiyon protokolü, eşik seçimi |
| 5 | Eğitim | Dilara | **Hiperparametre tablosu** (§4.7), early stopping, scheduler, seed, donanım, süreler |
| 6 | Deneysel tasarım | İkisi | §4.6 deney tablosu |
| 7 | Sonuçlar | İkisi | Karşılaştırma tabloları, figürler, ablasyon |
| 8 | Hata analizi ve tartışma | İkisi | §4.12 soruları, literatürle kıyas (koşul farklarıyla) |
| 9 | Sınırlılıklar ve gelecek çalışmalar | İkisi | Tek hane, sentetik etiket, kış dışı test dönemi, hava durumu yok; kısa boşluk interpolasyonunda sonraki gözlem kullanılır, ancak hedefi doldurulmuş pencereler değerlendirme metriklerine alınmaz; ElectricityLoadDiagrams'a genişletme, Transformer, rolling-origin, olasılıksal tahmin |
| 10 | Sonuç | İkisi | Hipotezlerin kabul/red durumu |
| — | Kaynakça (IEEE), Ekler (kod deposu bağlantısı, ek figürler) | | |

**Rapor–kod–sunum tutarlılığı zorunludur** (Yönerge §15.1). Rapordaki her sayı `results/tables/` dosyalarından kopyalanır, elle yazılmaz.

---

## 8. Sunum kılavuzu ve puanlama eşlemesi

### 8.1 Puanlama formu → yapılacaklar

| Kriter (puan) | Formdaki sorular | Bizim karşılığımız |
|---|---|---|
| **Problem Tanımı ve Akademik Çerçeve (20)** | Problem açık mı? Amaç, motivasyon, kapsam net mi? Literatürle ilişki doğru mu? | Slayt 2–4: problem + araştırma soruları + literatür boşluğu tablosu |
| **Yöntem / Model Tasarımı ve Teknik İçerik (30)** | Yöntem probleme uygun mu? Veri hazırlığı, eğitim ve teknik detaylar doğru mu? Açıklamalar yeterli mi? Ekip modele hâkim mi? | Slayt 5–10: veri, ön işleme, bölme şeması, mimari çizimleri, hiperparametre tablosu. **En çok puan burada** → iki kişi de mimariyi tahtada çizebilmeli |
| **Deneysel Sonuçlar ve Analiz (25)** | Sonuçlar metriklerle mi? Grafik, confusion matrix, örnek çıktılar doğru yorumlanıyor mu? Sonuç–amaç bağı var mı? | Slayt 11–15: karşılaştırma tablosu, kayıp eğrileri, confusion matrix, PR eğrisi, hata analizi; her grafikte "ne görüyoruz → ne anlama geliyor → hipotezle ilişkisi" |
| **Sunum Becerisi ve İletişim (15)** | Akıcı mı? Dengeli katılım? Profesyonel görseller? Sorulara yeterli yanıt? | Konuşma süresi yaklaşık eşit; tutarlı tema; slayt başına tek ana mesaj |
| **Zaman Yönetimi ve Organizasyon (10)** | Süre içinde mi? Akış Problem → Yöntem → Sonuç → Tartışma tutarlı mı? Gereksiz/eksik bilgi yok mu? | Süre açıklanınca slayt sayısı ayarlanır; provada kronometre |

### 8.2 Slayt planı (Yönerge §16'daki 12 adım sırasıyla)
1. Kapak (başlık, isimler, numaralar, ders)
2. Problem, amaç, motivasyon → *§16-1*
3. Araştırma soruları ve hipotezler
4. Literatür ve mevcut yaklaşımlar (tablo) → *§16-2*
5. Veri seti ve temel özellikleri → *§16-3*
6. EDA ve ön işleme (ısı haritası, ACF, eksik veri stratejisi) → *§16-4*
7. Bölme şeması ve sızıntı önlemleri
8. Mimariler: LSTM/GRU tahmin + LSTM-AE (şemalar) → *§16-5*
9. Deneysel tasarım ve referanslar → *§16-6*
10. Eğitim ve hiperparametreler + kayıp eğrileri → *§16-7*
11. Tahmin sonuçları (tablo + ufka göre hata) → *§16-8*
12. Anomali sonuçları (PR eğrisi + confusion matrix + tip bazlı recall)
13. Model karşılaştırmaları ve hata analizi → *§16-9*
14. Yorum ve sınırlılıklar → *§16-10*
15. Genel değerlendirme ve gelecek çalışmalar (hipotezler: kabul/red) → *§16-11*
16. **Canlı demo** → *§16-12* (kodun yapısı, modelin nasıl yüklendiği, örnek girdiden tahmin ve skor üretimi)
17. Kaynaklar + teşekkür

### 8.3 Konuşmacı dağılımı (dengeli katılım)
- **Dilara:** 1–3, 5, 7, 8 (LSTM/GRU kısmı), 10, 11, 16 (demo)
- **Şeyda:** 4, 6, 8 (AE kısmı), 9, 12, 13, 14, 15
- Demo sırasında diğer kişi kodu açıklar (ikisinin de koda hâkim olduğu görünür).

### 8.4 Demo B planı
- İnternet yok: tüm veri ve ağırlıklar yerelde; `streamlit run` önceden açılmış.
- Uygulama çökerse: aynı çıktıyı üreten `notebooks/02_results_analysis.ipynb` hücreleri hazır.

---

## 9. Savunma hazırlığı – soru bankası
Her iki kişi de **hepsini** cevaplayabilmelidir.

**Problem ve veri**
1. Neden bu veri setini seçtiniz, ElectricityLoadDiagrams'ı neden ana veri yapmadınız?
2. Dakikalık veriyi neden saatliğe indirdiniz? Ne kaybettiniz?
3. Eksik verileri nasıl ele aldınız? Doldurulmuş değerler sonuçları nasıl etkiler?
4. `Global_intensity` girdi olarak kullanılabilir mi? Hedefle ilişkisi ne?

**Bölme ve sızıntı**
5. Neden rastgele değil, kronolojik böldünüz? Rastgele bölseydiniz ne olurdu?
6. Ölçekleyiciyi tüm veriye fit etseydiniz sorun ne olurdu?
7. Test kümesini ne zaman ve kaç kez kullandınız?
8. Doğrulama penceresinin girdisinde eğitim verisi var; bu sızıntı mı?

**Modeller**
9. LSTM hücresindeki kapılar (forget, input, output) ne yapar? GRU'dan farkı nedir?
10. Vanishing gradient nedir, LSTM bunu nasıl hafifletir?
11. Neden 168 saatlik pencere? ACF neyi gösterdi?
12. 24 saati tek seferde (çok çıktılı) tahmin etmek ile adım adım tahmin etmek arasındaki fark?
13. Autoencoder nasıl anomali bulur? Neden yalnızca normal veriyle eğitilir?
14. Tahmin artığı tabanlı dedektör ile yeniden yapılandırma tabanlı dedektörün farkı?

**Eğitim**
15. Kayıp fonksiyonu olarak neden MSE? Huber ne zaman tercih edilir?
16. Adam nasıl çalışır? Learning rate'i neden 1e-3 seçtiniz?
17. Early stopping, dropout, weight decay, gradient clipping ne işe yarar?
18. Aşırı öğrenme oldu mu? Hangi grafikten anlıyorsunuz?
19. Model seçim kriteriniz neydi?

**Değerlendirme**
20. RMSE ile MAE neden farklı? Hangisi hangi durumda önemli?
21. MAPE neden sorunlu, nasıl çözdünüz?
22. Anomali etiketi yokken F1'i nasıl hesapladınız? Sentetik enjeksiyonun zayıflığı ne?
23. Eşiği nasıl seçtiniz? Neden test kümesinde değil?
24. Neden ROC-AUC yanında PR-AUC da raporladınız?
25. Confusion matrix'inizi yorumlayın: yanlış alarmların maliyeti nedir?

**Sonuç ve sınırlılıklar**
26. Modeliniz hangi durumlarda başarısız? Örnek gösterin.
27. LSTM ve GRU farkı anlamlı mı? Seed std'si ne diyor?
28. Sonuçlarınızı literatürle nasıl kıyasladınız; neden doğrudan kıyas zor?
29. Gerçek bir sistemde kullanılsa neye ihtiyaç olurdu?
30. Kodda şu fonksiyon ne yapıyor? (rastgele bir dosya açılabilir → her dosyayı ikiniz de bilin)

---

## 10. Kurallar

### 10.1 Veri sızıntısı kontrol listesi (Yönerge §6, §7, §10, §11)
- [ ] Kronolojik bölme, kümeler arası zaman çakışması yok (testle doğrulanır)
- [ ] Ölçekleyici, imputasyon istatistikleri, Isolation Forest, AE: **yalnız eğitime fit**
- [ ] Hiperparametre ve eşik seçimi **yalnız doğrulamada**
- [ ] Test **bir kez**, en sonda
- [ ] Hedef penceresi küme sınırını aşmıyor
- [ ] Enjeksiyon val ve test için farklı seed'le, ayrı kopyalarda
- [ ] Gelecek bilgisi taşıyan öznitelik yok (ör. merkezli hareketli ortalama kullanılmaz; yalnız geçmişe bakan pencereler)

### 10.2 Hazır kod ve kütüphaneler (Yönerge §18)
- PyTorch, scikit-learn vb. serbest; ama her fonksiyonun ne yaptığını ikiniz de bilmelisiniz.
- Dışarıdan alınan her kod parçası için dosyada kaynak yorumu + raporda atıf.
- GitHub'dan hazır bir projeyi olduğu gibi çalıştırmak **kabul edilmez**.

### 10.3 Yapay zekâ araçları (Yönerge §19)
- Claude/ChatGPT ile üretilen kod **okunmadan ve test edilmeden** kullanılmaz.
- **Yapılmayan deney, elde edilmeyen sonuç rapora yazılmaz.** Rapordaki tüm sayılar `results/` dosyalarından gelir.
- Yapay zekânın önerdiği kaynaklar doğrulanmadan kaynakçaya eklenmez.
- "Kodu yapay zekâ yazdı, bilmiyoruz" savunmada kabul edilmez → çapraz anlatım bu yüzden var.

---

## 11. Kaynak havuzu (IEEE)
> ⚠️ Aşağıdaki kaynaklar başlangıç havuzudur. **Her birini Google Scholar veya DOI üzerinden kontrol edin**, cilt/sayfa bilgilerini orijinalinden düzeltin ve yalnızca gerçekten okuduğunuz kaynakları kullanın (Yönerge §19). Ek olarak, aynı veri setini kullanan son yıllardaki 2–3 çalışmayı kendiniz bulun.

**Temel modeller**
- [1] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.
- [2] K. Cho *et al.*, "Learning phrase representations using RNN encoder–decoder for statistical machine translation," in *Proc. EMNLP*, 2014.
- [3] J. Chung, C. Gulcehre, K. Cho, and Y. Bengio, "Empirical evaluation of gated recurrent neural networks on sequence modeling," arXiv:1412.3555, 2014.
- [4] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. ICLR*, 2015.

**Enerji / yük tahmini**
- [5] W. Kong, Z. Y. Dong, Y. Jia, D. J. Hill, Y. Xu, and Y. Zhang, "Short-term residential load forecasting based on LSTM recurrent neural network," *IEEE Trans. Smart Grid*, vol. 10, no. 1, pp. 841–851, 2019.
- [6] D. L. Marino, K. Amarasinghe, and M. Manic, "Building energy load forecasting using deep neural networks," in *Proc. IECON*, 2016.
- [7] T.-Y. Kim and S.-B. Cho, "Predicting residential energy consumption using CNN-LSTM neural networks," *Energy*, vol. 182, pp. 72–81, 2019. *(aynı veri setini kullanır)*
- [8] T. Hong and S. Fan, "Probabilistic electric load forecasting: A tutorial review," *Int. J. Forecasting*, vol. 32, no. 3, pp. 914–938, 2016.
- [9] R. J. Hyndman and A. B. Koehler, "Another look at measures of forecast accuracy," *Int. J. Forecasting*, vol. 22, no. 4, pp. 679–688, 2006. *(MAPE sorunları)*

**Anomali tespiti**
- [10] V. Chandola, A. Banerjee, and V. Kumar, "Anomaly detection: A survey," *ACM Computing Surveys*, vol. 41, no. 3, 2009.
- [11] P. Malhotra, L. Vig, G. Shroff, and P. Agarwal, "Long short term memory networks for anomaly detection in time series," in *Proc. ESANN*, 2015.
- [12] P. Malhotra *et al.*, "LSTM-based encoder-decoder for multi-sensor anomaly detection," arXiv:1607.00148, 2016.
- [13] K. Hundman, V. Constantinou, C. Laporte, I. Colwell, and T. Soderstrom, "Detecting spacecraft anomalies using LSTMs and nonparametric dynamic thresholding," in *Proc. ACM SIGKDD*, 2018.
- [14] F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation forest," in *Proc. IEEE ICDM*, 2008.
- [15] Y. Himeur *et al.*, "Artificial intelligence based anomaly detection of energy consumption in buildings: A review, current trends and new perspectives," *Applied Energy*, vol. 287, 2021.

**Veri setleri**
- [16] G. Hebrail and A. Berard, "Individual household electric power consumption," UCI Machine Learning Repository, 2012. doi: 10.24432/C58K54.
- [17] A. Trindade, "ElectricityLoadDiagrams20112014," UCI Machine Learning Repository, 2015. doi: 10.24432/C58C86.

---

## 12. Teslim öncesi son kontrol listesi
Yönerge §20'deki temel beklentiler:

- [ ] Problem, amaç, kapsam açık
- [ ] Literatür taraması yapıldı, yöntem seçimini etkiledi (yalnızca özet değil)
- [ ] Veri seti belirlendi, özellikleri incelendi, seçim gerekçesi ve sınırlılıkları yazıldı
- [ ] Veri analizi ve ön işleme yapıldı, her adımın gerekçesi var
- [ ] En az bir sinir ağı modeli (LSTM, GRU, LSTM-AE)
- [ ] Eğitim/doğrulama/test bilimsel olarak uygun (kronolojik, sızıntısız), oranlar ve örnek sayıları raporlandı
- [ ] Probleme uygun metrikler, yalnız tek metrik değil
- [ ] En az bir referans/alternatif modelle karşılaştırma (bizde: 5 tahmin + 2 anomali referansı)
- [ ] Tablo ve grafikler eksen adı, birim ve küme bilgisiyle
- [ ] Hata analizi, güçlü/zayıf yönler, sınırlılıklar
- [ ] Hiperparametre tablosu ve seed'ler raporda
- [ ] Rapor, kod ve sunum birbiriyle tutarlı
- [ ] Demo çalışır durumda, temiz ortamda test edildi
- [ ] Kod yeniden çalıştırılabilir (README komutları çalışıyor)
- [ ] Tüm dış kaynaklar (veri, kod, model, makale) atıflı
- [ ] İki üye de §9 soru bankasının tamamını cevaplayabiliyor

---

## 13. Riskler ve B planları

| Risk | Belirti | B planı |
|---|---|---|
| Eğitim çok yavaş | 1 epoch > 2 dk (CPU) | Colab GPU; stride 2–3; hidden 32; ızgarayı küçült |
| LSTM/GRU baseline'ı geçemiyor | Val MAE ≥ Seasonal Naive 168 | Takvim özniteliklerini kontrol et; hedefe log1p; Huber kayıp; lr 5e-4. Yine de geçemezse **dürüstçe raporla ve nedenini tartış** (tek hane gürültüsü) |
| AE her şeyi anomali sayıyor | Val'de precision çok düşük | Eşiği val F1-max ile seç; pencere 24 → 48; skor yumuşatma (3 saatlik ortalama) |
| Gün sonu hedefi kaçtı | ✅ kriterleri eksik | Öncelik sırası: (1) F1, F5, F6 + A1, A2 çalışıyor, (2) değerlendirme + figürler, (3) demo, (4) ablasyon ve seed'ler |
| Demo çöküyor | Sunum günü hata | Notebook yedeği; ekran kaydı videosu |
| Bir üye hastalandı | — | Çapraz anlatımlar sayesinde diğeri devam eder; README güncel tutulur |
