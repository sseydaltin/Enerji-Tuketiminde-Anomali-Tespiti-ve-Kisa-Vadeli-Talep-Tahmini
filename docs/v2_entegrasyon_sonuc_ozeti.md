# A3 + taban (v2) entegrasyonu — sonuç özeti (rapor yazımı için)

Dal: `dilara/v2-taban-entegrasyon` (taban `cb895d1`), yerel · Tarih: 2026-11-09
Kullanıcının bildirimine göre ders hocası bu eklemenin uygulanmasını istedi. v2, **test sonrası geliştirilmiş ek bir değerlendirmedir; final yöntem değildir.**

## 1. Alınan dosyalar ve kod değişiklikleri

**Kaynak:** `~/Downloads/seyda_v2_guncelleme.zip` (SHA-256 `249f271e…`). Bütün içerik değiştirilmeden `results/reference/seyda_v2_zip/` altında saklandı.

| Dosya | İşlem |
|---|---|
| `src/anomaly_v2_floor_test.py` | Alındı ve uyarlandı (aşağıda) |
| `src/plot_h3_type_recall.py` | Aynen alındı. Figür yerelde yeniden üretildi, zip'tekiyle birebir aynı |
| `docs/anomaly_v2_floor_test.md`, `docs/h3_neden_karsilanmadi.md` | Metin aynen alındı. Başa kaynak ve doğrulama notu, v2 belgesine yerel yeniden üretim bölümü (§3) eklendi |
| `results/tables/anomaly_v2_floor_test_{per_seed,summary,paired}.csv`, log | Zip sürümleri referans olarak saklandı. Proje içindekiler **yerel koşunun** çıktılarıdır |
| `results/figures/H3_A3-s2s-norm_dev_type_recall.png` | Yerelde üretildi |

**Kod değişiklikleri:**
- **Bağımlılık:** `src.anomaly_floor_channel` (`FLOOR_QUANTILE`, `floor_flags`) `main`'de yok. `dilara/anomali-adaylar` dalında yazılmıştı; bu dal onun soyundan geldiği için modül hazırdı, taşıma gerekmedi.
- **`floor_flags`:** İsteğe bağlı iki koruma eklendi:
  - `index`: iki kayıt arası tam 1 saat değilse önceki kayıt ardışık sayılmaz,
  - `imputed`: doldurulmuş saat "düşük" sayılmaz.
  Varsayılan çağrı eski davranışla aynı; önceki deney tabloları etkilenmez.
- **Eklenen yardımcı:** `train_floor`.
- **Yeni `src/anomaly_v2.py`:** Tek kural fonksiyonu `v2_score`. Deney betiği ve demo bunu kullanır.
- **`src/anomaly_v2_floor_test.py` uyarlamaları:**
  - kural `v2_score`'dan gelir,
  - `--out-dir` seçeneği,
  - kullanılan dosyaların SHA-256 özetleri,
  - enjeksiyonsuz testte taban / A3 / v2 alarm blokları ayrı ayrı,
  - korumalı ile korumasız taban bayrağının karşılaştırması.
- **Demo:**
  - `app/anomaly_services.py`: `analyze_test_range(..., v2=True)`.
  - `app/pages/anomaly.py`: "A3 + taban (v2) – test sonrası, final değil" seçeneği, taban çizgisi ve açıklama notu.
  - `app/ui.py`: etiket. Mevcut yöntem adları değişmedi.
- **Testler:** `tests/test_anomaly_v2.py` şunları sınar:
  - iki ardışık düşük saat alarm verir, tek düşük saat vermez,
  - zaman boşluğu ardışıklığı bozar,
  - doldurulmuş saat düşük sayılmaz,
  - OR birleşimi ve ortak maske,
  - taban 0,005 yüzdeliğidir (0,5 değil),
  - demo kuralı deneyle aynıdır.

## 2. Deney nasıl çalıştı

- **Komut:** `python -m src.anomaly_v2_floor_test --device cpu`. Önce `--out-dir` ile ayrı bir dizine yazıldı ve zip'le karşılaştırıldı.
- **Veri:** `data/processed/hourly.parquet`. Test 2010-04-01 – 2010-11-26 (5 550 değerlendirilen saat).
- **Seed'ler:** ana senaryo 2, 12, 22, 32, 42; zor senaryo 4, 14, 24, 34, 44.
- **Ağırlıklar:** F10-opt, F5-opt ve OOF modelleri; A2. Tahmin modelleri yeniden eğitilmedi. SHA-256 özetleri `results/tables/anomaly_v2_floor_test_inputs.csv` dosyasında.
- **A3-s2s-norm ayarı:** pencere 3, eşik 3,5580; ikisi de `anomaly_protocol_selection.csv`'den (val-dev seçimi). Testte hiçbir ayar seçilmedi.
- **Taban:** train'in doldurulmamış saatlerinin 0,005 yüzdeliği = **0,1554 kW**. Koşuda train'den hesaplandı, sabitlenmedi.
- **Ardışıklık:** Test saatleri boşluksuz bir saatlik ızgara. Zaman boşluğu ve doldurulmuş saat koruması bu kopyalarda hiçbir bayrağı değiştirmedi (0 saat).
- **Maske:** İki yöntem aynı saatlerde değerlendirildi: A3 skoru olan ve doldurulmamış saatler. Etiketler yalnız metrik hesabında kullanıldı.

## 3. Sonuçlar (test, olay metrikleri, 5 seed ort. ± std)

| Senaryo | Yöntem | Olay precision | Olay recall | Olay F1 |
|---|---|---|---|---|
| Ana | A3-s2s-norm (resmî final) | 0,424 ± 0,164 | 0,358 ± 0,180 | 0,384 ± 0,170 |
| Ana | **A3 + taban (v2)** | 0,507 ± 0,125 | 0,510 ± 0,158 | **0,506 ± 0,135** |
| Zor | A3-s2s-norm | 0,364 ± 0,083 | 0,286 ± 0,073 | 0,318 ± 0,073 |
| Zor | **A3 + taban (v2)** | 0,462 ± 0,082 | 0,463 ± 0,062 | **0,460 ± 0,059** |

Kaynak: `results/tables/anomaly_v2_floor_test_summary.csv`.

**Seed başına olay F1 (A3 → v2)** (`anomaly_v2_floor_test_per_seed.csv`):

| Ana seed | 2 | 12 | 22 | 32 | 42 |
|---|---|---|---|---|---|
| A3 | 0,526 | 0,200 | 0,231 | 0,380 | 0,580 |
| v2 | 0,590 | 0,364 | 0,452 | 0,427 | 0,698 |

| Zor seed | 4 | 14 | 24 | 34 | 44 |
|---|---|---|---|---|---|
| A3 | 0,261 | 0,324 | 0,231 | 0,370 | 0,405 |
| v2 | 0,435 | 0,524 | 0,392 | 0,429 | 0,519 |

**Eşleştirilmiş fark (v2 − A3, aynı kopya):**
- Ana senaryo: +0,123 ± 0,072. **5/5 seed'de artış.**
- Zor senaryo: +0,141 ± 0,056. **5/5 seed'de artış.**
- Kaynak: `anomaly_v2_floor_test_paired.csv`.

**Tür başına olay recall'ü** (A3 / v2):

| Senaryo | Spike | Drop | Seviye kayması | Desen |
|---|---|---|---|---|
| Ana | 0,27 / 0,27 | **0,10 / 1,00** | 0,73 / 0,73 | 0,18 / 0,18 |
| Zor | 0,22 / 0,22 | **0,00 / 1,00** | 0,35 / 0,35 | 0,40 / 0,40 |

**Enjeksiyonsuz test** (`anomaly_v2_floor_test_clean.csv`):
- Taban kanalı **0** alarm bloğu verdi.
- A3 tek başına **8** blok, v2 de **8** blok verdi.
- Yani taban kanalı gerçek veride alarm **eklemedi**; ama sistemin gerçek veride alarmı sıfır değil.

## 4. Zip referansıyla karşılaştırma

Yerel koşu, zip tablolarıyla **birebir** aynı:
- `per_seed`: 20 satır.
- `summary`: 4 satır.
- `paired`: 2 satır.
- Bütün sayısal sütunlarda en büyük fark 0.
- Kontrol listesindeki değerlerin hepsi tuttu:
  - A3 ana 0,383509, v2 ana 0,506143,
  - A3 zor 0,318348, v2 zor 0,459568,
  - 5/5 seed'de artış,
  - drop recall 0,10 → 1,00 (ana) ve 0,00 → 1,00 (zor),
  - taban alarm bloğu 0.
- A3 ana senaryo sonucu resmî değerle aynı (0,3835; tolerans 0,005).

## 5. Kazancın kaynağı ve sınırları

- **Kazanç yalnız drop türünden geliyor.** Diğer üç türün recall'ü iki yöntemde aynı.
- **Kazanç sentetik drop değerine bağlı.** Enjekte drop değeri sabit **0,05 kW**, taban 0,155 kW'ın altında. Bu yüzden taban kuralı bu olayları neredeyse tanım gereği yakalıyor.
  - Geliştirme verisindeki duyarlılık senaryosunda drop değeri tabanın üstüne (train 5. yüzdelik ≈ 0,21 kW) çekilince kazanç sıfırdı (`docs/anomaly_floor_channel.md`).
  - Gerçek kesintilerin bu kadar derin olacağı gösterilmedi.
- **Değerlendirme sınırı:** Test kümesi 2026-10-31'de resmî koşuda görüldü ve taban kuralı ondan sonra tasarlandı. Bu yüzden bu sonuç **bağımsız, hiç görülmemiş test sonucu değildir.** Geliştirme protokolü de yeni yöntemler için test kullanımını yasaklıyordu; bu koşu bilinçli bir sapmadır (`docs/deney_gunlugu.md`).
- **Final yöntem değişmedi.** Final yöntem A3-s2s-norm; resmî test tabloları (`anomaly_protocol_test*.csv`, `final_anomaly_val_vs_test.csv`) değişmedi.

## 6. Kullanılan olay F1 tanımı

- **Olay:** aynı türden ardışık etiketli saatler.
- **Olay recall:** en az bir saati alarm alan olay / tüm olaylar.
- **Olay precision:** en az bir etiketli saatle kesişen alarm bloğu / tüm alarm blokları.
- **Olay F1:** bu ikisinin harmonik ortalaması (`src/utils/anomaly_metrics.event_metrics`; denetimde `block_f1` adıyla anılır).

Bu bir doğruluk (accuracy) veya "başarı yüzdesi" değildir. 0,506, "olayların %50,6'sı doğru" anlamına gelmez.

## 7. H3 hedefi

- **Kaynak:** README §6.2 "Amaç ve Hedefler" tablosu (öneri formu taslağı): "Enjekte anomalili test kümesinde … F1 ≥ 0.70".
- Proje kendi koyduğu bir hedeftir; hocanın zorunlu eşiği olarak kaydedilmemiştir.
- v2 ile de karşılanmadı: ana 0,506, zor 0,460.

## 8. Tahmin bölümü (değişmedi; doğrulanmış mevcut sonuç)

- F13-ens (0,8 Seq2Seq + 0,2 LSTM, 3 seed), hane testi: **MAE 0,4062 ± 0,0047 kW**, RMSE 0,598, R² 0,297.
- SN168'e göre MAE %26,0 daha düşük (SN168 MAE 0,5489 kW).
- Kaynak: `results/tables/final_forecast_val_vs_test.csv`.

## 9. Demo

- **Başlatma:** `python -m streamlit run app/streamlit_app.py --server.address localhost`
- **Kullanım:** Anomali Tespiti sayfası → Yöntem: "A3 + taban (v2) – test sonrası, final değil".
- **Gösterim:** v2, A3 ile aynı test kopyası ve aynı eşikle gösterilir. Tüketim grafiğinde taban çizgisi vardır.
- **Kural:** Demo da `src/anomaly_v2.v2_score` fonksiyonunu kullanır.
- **Uygulama kontrolü** (2026-11-09; Streamlit AppTest, tarayıcıda görsel kontrol yapılmadı):
  - Dört sayfa hatasız açıldı.
  - Anomali sayfasında, yalnız "Ani düşüş" enjeksiyonu, ana senaryo, seed 2, bütün test dönemi: A3 **0 / 35** olay, v2 **35 / 35** olay yakaladı. Yanlış alarm bloğu ikisinde de 9.
  - Bu bir uygulama kontrolüdür, deney sonucu değildir: tek tip enjeksiyon olduğu için §3'teki tablolarla kıyaslanmaz.
