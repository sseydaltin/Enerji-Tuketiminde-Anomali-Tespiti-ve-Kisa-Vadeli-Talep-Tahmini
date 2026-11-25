# Sentetik anomalilerle eğitilen gözetimli dedektör (yöntem değişikliği; geliştirme, final değil)

Dal: `dilara/anomali-sentetik-siniflandirici` (taban `70bd1a0`) · Komut: `python -m src.anomaly_supervised`

## 0. Hedefin ve bölmelerin geçerliliği (koşudan önce, 2026-11-09 00:30)

**H3 (README §6.2, öneri formu):** "Enjekte anomalili **test kümesinde** P/R/F1/PR-AUC … F1 ≥ 0,70."
- Öneri aşamasında F1'in türü yazılmamış.
- Final protokolü (`docs/final_test_protokolu.md`, test öncesi, 2026-10-31) ana metriği **olay F1** = `block_f1`, ana senaryoyu σ tabanlı "ana" = geliştirmedeki "orta" olarak sabitledi.
- Test sonucu (bir kez, 2026-10-31): A3-s2s-norm olay F1 0,384, nokta F1 0,147 → **iki tanımla da karşılanmadı.**
- Hedef başka bir senaryoya veya metriğe taşınmıyor. Raporda `block_f1` (orta senaryo) ana, nokta F1 ek olarak verilir.

**Bölme geçmişi:**

| Dönem | Kullanım geçmişi | Yeni yöntem için durum |
|---|---|---|
| Train 2006-12 – 2008-09 | Tahmin modelleri bununla eğitildi (F10-opt, F5-opt; S-* için 2009-05'e kadar) | Bu dönemde A3 artıkları in-sample; sınıflandırıcı eğitimi için uygun değil |
| OOF katları 2008-10 – 2009-09 | A3 std kalibrasyonu; kat 3 (2009-06 – 09) = yaz-dev, defalarca değerlendirildi | Kat 1–2 (2008-10 – 2009-05) dedektör olarak hiç değerlendirilmedi; artıklar kat dışı → **sınıflandırıcı eğitim dönemi** |
| val-dev = kış-dev 2009-10 – 12 | Seçim; F10-opt early stopping; çok kez görüldü | Geliştirme değerlendirmesi |
| val-hold 2010-01 – 03 | Birden çok kez görüldü (revizyon 1, A6) | **Kullanılmaz** (protokol) |
| Test 2010-04 – 2010-11-26 (verinin sonu) | Bir kez kullanıldı, sonuçları biliniyor | **Kullanılmaz**: yeni yöntem için bağımsız değil |

**Sonuç: UCI hane verisinde hiç kullanılmamış bir zaman bölümü YOK.**
- Veri 2010-11-26'da bitiyor; test'ten sonra dönem kalmadı.
- Aynı zaman aralığına yeni enjeksiyon seed'i uygulamak bağımsız veri değildir.
- ELD'deki 210 kullanılmamış müşteri farklı bir alandır (toplu yük) ve bu bilgisayarda yok.
- **Bu yüzden bu çalışmadaki hiçbir sonuç bağımsız doğrulanmış F1 değildir.** Hedefe yalnız geliştirmede ulaşılırsa böyle raporlanır.

## 1. Protokol (koşudan önce yazıldı; sonuç görüldükten sonra değiştirilmedi)

**Neden yöntem değişikliği?** Önceki tanılar şunu gösterdi:
- Tek bir skoru (A3) eşiklemek, tiplerin farklı imzalarını tek bir ölçekte yarıştırıyor:
  - spike: kısa ve büyük artık,
  - drop: alt yükün altı,
  - desen: gece düzeyinin olağan profilin üstüne çıkması,
  - seviye kayması: süren işaretli artık.
- Elle kurulan birleşimler (max/OR) bir mevsimde kazandırıp ötekinde kaybettirdi.
- Gözetimli bir sınıflandırıcı bu imzaları birlikte ağırlıklandırmayı **eğitim döneminden** öğrenebilir.

**Özellikler** (saat t için; yalnız t ve öncesi; merkezli pencere yok):

| Özellik | Anlam |
|---|---|
| `a3` | A3-s2s-hmean = ort_h \|z_h\| |
| `zbar`, `zbar_m3`, `zbar_m6` | işaretli ort. normalize artık ve geçmişe bakan 3/6 saatlik ortalaması |
| `z1` | h=1 işaretli normalize artık |
| `change` | \|Δy − μ_Δ\| / σ_Δ (train tablosu) |
| `c1z`, `c1z_m6` | (y − medyan) / ölçek, son 2 haftanın aynı saat + gün tipi (C1-L2, işaretli) ve 6 saatlik ortalaması |
| `ratio` | log((y + 0,05) / (medyan + 0,05)) |
| `low2` | log((max(y_t, y_{t−1}) + 0,05) / (taban + 0,05)); taban = train 0,5. yüzdelik |
| `a0z` | işaretli A0 z-skoru (son 7 günün aynı saati) |
| `hour_sin`, `hour_cos`, `daytype` | takvim |

- Ay özelliği yok: yaz sınıflandırıcısının eğitim döneminde yaz ayı yok.
- Ham kW düzeyi yok: sabit enjeksiyon değerini ezberlemesin.
- Alt sayaçlar yok: enjeksiyon yalnız toplamı değiştiriyor, yapay iz olur.
- Enjeksiyon kimliği, seed veya tip özellik olarak kullanılmaz.

**Eğitim verisi** (mevsime göre ayrı model, değerlendirme dönemlerinden önce):

| Değerlendirme | Eğitim (fit) | Boşluk | İç doğrulama (eşik) | Artıklar |
|---|---|---|---|---|
| kış-dev | 2008-10-01 – 2009-07-24 | 7 gün | 2009-08-01 – 2009-09-30 | F10-opt-oof1..3 (kat dışı) |
| yaz-dev | 2008-10-01 – 2009-03-24 | 7 gün | 2009-04-01 – 2009-05-31 | S-F10-L48-oof1..2 (kat dışı) |

- Kış modelinin iç doğrulaması yaz-dev dönemine (2009-08 – 09) düşer. Yaz-dev o modelle değerlendirilmez; yaz modeli ayrıdır.
- **Fit kopyaları:** 8 enjekte kopya, seed 7001–7008, **çeşitlendirilmiş** ayarlar (kopya başına döngüyle):
  - drop değeri ∈ {0,05; 0,10; 0,20; 0,30} kW,
  - spike σ katı (2–6),
  - seviye kayması ×(1,3–2,5),
  - desen alpha ∈ {0,5; 0,75; 1,0}.
  Etiket = saat başına `is_anomaly`.
- **İç doğrulama kopyaları:** 4 kopya, seed 7101–7104, değerlendirme tanımıyla aynı **orta** senaryo ayarları. Yalnız eşik seçimi; değerlendirme dönemine dokunmaz.
- σ tablosu, taban, C1 tabanı, değişim tablosu, std tabloları: o mevsimin train kesiminden (yaz için ≤ 2009-05-31).

**Model aileleri (en fazla üç; küçük, sabit ayarlar; ızgara yok):**
1. **LR:** lojistik regresyon (medyan doldurma + standartlaştırma, class_weight dengeli, C = 1).
2. **RF:** rastgele orman (300 ağaç, min_samples_leaf 20, class_weight dengeli; NaN için medyan doldurma).
3. **HGB:** HistGradientBoosting (max_iter 300, lr 0,05, max_leaf_nodes 31, class_weight dengeli; NaN doğal).

Hepsi seed 42. XGBoost bu Mac'te çöktüğü için kullanılmaz.

**Seçim (değerlendirme verisi kullanılmadan):**
- Her aile için eşik, iç doğrulama kopyalarında mevcut kuralla seçilir: birleşik block_f1'i en büyük yapan olasılık yüzdeliği, 300 nokta.
- Yumuşatma yok.
- Aile seçimi iç doğrulama block_f1 ortalamasıyla (iki mevsim ortalaması) yapılır.
- Bütün ayarlar ve eşikler dev değerlendirmesinden **önce dondurulur**.

**Değerlendirme (geliştirme):**
- Kış-dev + yaz-dev, orta senaryo, config seed'leri (10 + 10). Aynı kopyalarda referanslar kayıtlı ayarlarıyla: A0, A3-s2s-norm, S0-yerel (A3-s2s-hmean), S0+taban (deneysel).
- Raporlananlar: block_f1 (ana), count_f1, nokta P/R/F1, precision, recall, tip recall'ü, yanlış alarm/gün, gecikme, arka plan alarmları, eğitim ve çıkarım süresi, S0'a karşı eşleştirilmiş fark (%95 GA).
- Üç ailenin hepsi raporlanır; seçilen aile önceden belirlidir.
- Kış-dev ve yaz-dev daha önce defalarca görüldü. Bu yüzden sonuç "geliştirme sonucu"dur, bağımsız değildir.
- Aile seçimi dev'e bakmadan yapıldığı için seçim yanlılığı önceki deneylerden düşüktür, ama sıfır değildir: özellik tasarımı önceki dev tanılarından esinlendi.

**Kabul (deneysel seçenek olarak):**
- Önceki c1–c5 (S0-yerel'e karşı) + c7 (spike kaybı ≤ 0,05).
- **F1 hedefi:** iki mevsimde de block_f1 ≥ 0,70 ise "hedefe geliştirmede ulaşıldı" denir; aksi hâlde "ulaşılamadı".

**Uygulamaya ekleme:**
- Kabul edilse bile demo anomali sayfası test dönemini puanlıyor; yeni modeli orada göstermek test performansını açığa çıkarır.
- Bu yüzden demoya canlı olarak eklenmez. Yalnız Sonuçlar/Model Bilgisi metninde "deneysel, geliştirme" etiketiyle anılabilir (zaman kalırsa).

**Süre:** 00:29 – 02:29. Log `results/logs/anomaly_supervised.log`. Tek koşu; sonuçtan sonra özellik veya ayar değişikliği yok.

## 2. Sonuçlar (koşu 2026-11-09 00:32–00:34; log `results/logs/anomaly_supervised.log`)

- İlk başlatma bir dizin hatasıyla (`sigma` NumPy dizisi) özellik üretmeden durdu. Düzeltilip baştan çalıştırıldı; sonuç üretilmeden önce.
- Manifest denetimi geçti.
- Eğitim verisi:
  - kış: 56 560 saat (8 kopya), anomali oranı %2,7,
  - yaz: 33 576 saat, anomali oranı %2,7.
- Veri üretimi kış 6,7 s, yaz 5,7 s.

**İç doğrulama (eğitim dönemi, eşik seçimi; `anomaly_supervised_inner_selection.csv`):**

| Aile | Kış (birleşik block_f1, eşik) | Yaz | Ortalama |
|---|---|---|---|
| LR | 0,594 (0,946) | 0,505 (0,937) | 0,549 |
| RF | 0,923 (0,863) | 0,560 (0,965) | 0,742 |
| **HGB** | 0,971 (0,932) | 0,625 (0,904) | **0,798 → seçildi, donduruldu** |

**Geliştirme değerlendirmesi** (kış-dev / yaz-dev, orta senaryo, seed'ler 1…91 / 201…291; `anomaly_supervised_summary.csv`):

| Yöntem | block_f1 | count_f1 | nokta F1 | block P | olay R | spike | drop | seviye | desen | YA/gün | gecikme (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 0,518 / 0,310 | 0,506 / 0,275 | 0,328 / 0,215 | 0,51 / 0,26 | 0,57 / 0,44 | 0,14 / 0,03 | 0 / 0 | 0,90 / 0,95 | 0,77 / 0,40 | 0,041 / 0,097 | 3,8 / 6,6 |
| A3-s2s-norm | 0,541 / 0,529 | 0,443 / 0,489 | 0,289 / 0,139 | 0,42 / 0,63 | 0,80 / 0,46 | 0,90 / 0,80 | 0,15 / 0 | 1,00 / 0,90 | 0,90 / 0,00 | 0,123 / 0,025 | 2,5 / 5,6 |
| S0-yerel | 0,551 / 0,535 | 0,442 / 0,481 | 0,372 / 0,183 | 0,46 / 0,66 | 0,71 / 0,45 | 0,90 / 0,77 | 0 / 0 | 1,00 / 0,90 | 0,76 / 0,00 | 0,098 / 0,025 | 2,2 / 5,4 |
| S0+taban (deneysel) | 0,644 / 0,604 | 0,540 / 0,551 | 0,458 / 0,239 | 0,51 / 0,69 | 0,88 / 0,55 | 0,90 / 0,77 | 1,00 / 1,00 | 1,00 / 0,90 | 0,76 / 0,00 | 0,097 / 0,025 | 1,9 / 4,6 |
| SUP-LR | 0,562 / 0,533 | 0,465 / 0,486 | 0,459 / 0,265 | 0,46 / 0,66 | 0,76 / 0,45 | 0,71 / 0,71 | 0,04 / 0 | 1,00 / 0,95 | 0,96 / 0,03 | 0,100 / 0,024 | 1,8 / 6,1 |
| SUP-RF | 0,571 / 0,259 | 0,457 / 0,259 | 0,564 / 0,086 | 0,42 / 0,80 | 0,90 / 0,16 | 0,82 / 0,07 | 1,00 / 0 | 1,00 / 0,60 | 0,93 / 0,00 | 0,146 / 0,000 | 1,7 / 10,2 |
| **SUP-HGB (seçilen)** | **0,668 / 0,588** | 0,553 / 0,478 | 0,525 / 0,350 | 0,57 / 0,58 | 0,84 / 0,61 | 0,77 / 0,77 | 1,00 / 0,42 | 1,00 / 0,90 | 0,81 / 0,31 | 0,082 / 0,057 | 1,5 / 4,0 |

**SUP-HGB − S0-yerel** (eşleştirilmiş, %95 GA, n = 10):

| | Kış | Yaz |
|---|---|---|
| block_f1 | +0,117 [0,016; 0,218] | +0,053 [−0,029; 0,135] |
| count_f1 | +0,111 [0,004; 0,218] | −0,003 [−0,082; 0,077] |
| nokta F1 | +0,153 [0,080; 0,226] | +0,167 [0,115; 0,219] |
| spike recall | −0,131 [−0,284; 0,022] | 0 |
| desen recall | +0,05 | +0,31 [0,07; 0,54] |
| yanlış alarm / gün | −0,016 | +0,032 |

- İki mevsim ortalaması block_f1 farkı +0,085 ± 0,029.
- PR-AUC: SUP-HGB 0,560 / 0,474; S0-yerel 0,348 / 0,237.
- Süreler (`anomaly_supervised_timing.csv`): HGB eğitimi 2,1 / 1,8 s; çıkarım kopya başına yaklaşık 0,01 s (veri üretimi hariç).

**Karar** (`anomaly_supervised_decision.csv`):
- **SUP-HGB kabul edilmedi.** c1–c5 sağlandı, **c7 sağlanmadı**: kış spike recall'ü 0,90 → 0,77.
- SUP-LR ve SUP-RF de kabul edilmedi. RF yazda çöktü: recall 0,16.
- **F1 ≥ 0,70 hedefine ulaşılamadı:** hiçbir yöntem iki mevsimde de block_f1 ≥ 0,70 vermedi. En yüksek kış değeri 0,668 (SUP-HGB), en yüksek yaz değeri 0,604 (S0+taban).
- Protokol gereği özellik, ayar veya eşik değiştirilmedi; ikinci koşu yapılmadı.

**Tanı: iç doğrulama ile dev arasındaki fark** (kış 0,971 → 0,668; salt-okunur, seçim değiştirmez; `scratchpad/shift_diag.py`):

| Küme | Kaynak | A3 90. yüzdelik | A3 99. yüzdelik | z̄ std |
|---|---|---|---|---|
| kış | fit (OOF) | 1,74 | 3,37 | 1,04 |
| kış | iç doğrulama (OOF, 2009-08/09) | **1,28** | 2,59 | 0,73 |
| kış | dev (tüm-train modeli) | 1,61 | 3,55 | 1,03 |
| yaz | fit (OOF) | 1,67 | 3,21 | 1,05 |
| yaz | iç doğrulama (OOF, 2009-04/05) | 1,41 | 3,23 | 0,87 |
| yaz | dev (tüm-train modeli) | 1,21 | **2,31** | 0,78 |

- Kış iç doğrulaması sakin bir yaz dönemine düşüyor. Artıklar küçük olduğu için anomaliler orada kolay ayrılıyor; iç doğrulama F1'i iyimser çıkıyor ve orada seçilen eşik dev'e taşınmıyor.
- Yazda dev artıklarının ölçeği eğitimdekinden küçük.
- Dönemler (ve OOF ile tüm-train modeli) arasındaki artık ölçek kayması, eğitilen sınırların genelleşmesini sınırlıyor. Bu bir hipotezdir; nedensel olarak doğrulanmadı.

**Sınırlar:**
- Dev kümeleri defalarca görüldü; bağımsız doğrulama yok.
- Özellik tasarımı önceki dev tanılarından esinlendi.
- Eğitim sentetik anomalilerle yapıldı. Sonuçlar gerçek etiketli saha başarısı değildir.
- Eğitim çeşitliliğindeki drop değerlerinden biri (0,05 kW) değerlendirmedeki değerle aynı.
