# S0 + fiziksel taban (kesinti) kanalı — son sınırlı deney (geliştirme, final değil)

Dal: `dilara/anomali-adaylar` · Komut: `python -m src.anomaly_floor_channel`

## 1. Protokol (koşudan önce yazıldı, 2026-11-09 00:17; sonuç görüldükten sonra değiştirilmedi)

**Tanı (koşudan önce, `scratchpad/drop_diag.py`):** Orta senaryodaki 24 drop olayı (kış 16, yaz 8) hiçbir yöntemde yakalanmadı.
- Enjekte değer 0,05 kW; bu, train'in en düşük gözlenen saatlik değerinden (0,124 kW) de düşük. Olaylar ortalama 1,0–1,8 kW'tan bu değere iniyor.
- A3 ölçekli skoru drop'larda en fazla 0,80 / 0,81 (eşik 1,16): hane tüketiminin olağan tahmin hatası içinde kalıyor.
- C1-L2 (w = 6) en fazla 6,9 / 4,3 (eşik 7,05): 6 saatlik ortalama 2–6 saatlik düşüşü seyreltiyor.

**Aday (önceki yöntemlerden farklı bilgi: mutlak fiziksel alt yük).**
- f_t = (y_t < taban) VE (y_{t−1} < taban). Yalnız t ve t−1'e bakar.
- Taban = ilgili kümenin train kesiminin doldurulmamış saatlerinin 0,5. yüzdeliği (kış 0,155, yaz 0,154 kW).
- Ardışık 2 saat koşulu (k = 2), A6'nın test sonrası val-dev'de seçtiği değerdir. Bu yüzden kış-dev k için bağımsız değildir; burada yeni seçim yapılmaz.
- Alarm = S0 alarmı (A3-s2s-hmean / q_A3 ≥ 1,1636, w = 1) VEYA f_t. Eşdeğer skor r = max(a / 1,1636, 2·f); alarm r ≥ 1.
- Kanal seçimi için etiket veya tip kullanılmaz.

**Bilinen risk.**
- Enjekte değer tabanın altında olduğu için orta senaryoda drop recall'ü büyük ölçüde **tanım gereği** yükselir.
- Kural enjeksiyon değerini kullanmaz; taban yalnız train dağılımından gelir. Ama genel drop başarısı iddia edilemez.
- **Duyarlılık senaryosu** (yalnız rapor, seçim yok): Orta senaryo, aynı seed'ler, tek fark drop değeri = o kümenin train 5. yüzdeliği (kış 0,213, yaz 0,207 kW; tabanın üstünde).

**Karşılaştırma:** S0-yerel ve S0+taban. Aynı veri, ağırlıklar (manifest hash denetimi), seed'ler; kış-dev + yaz-dev, 10 seed / küme. Hiçbir eşik veya parametre seçilmez.

**Kabul ölçütü (orta senaryo, S0-yerel'e karşı; hepsi):**
- c1–c5: önceki deneylerle aynı.
- c7: her mevsimde spike recall kaybı ≤ 0,05.
- c8: duyarlılık senaryosunda her mevsimde block_f1 farkı ≥ −0,01 (taban kanalı başka koşulda zarar vermemeli).

**Kabul edilirse:**
- Yalnız "deneysel — geliştirme verisinde değerlendirildi; drop kazancı enjeksiyon değerine bağlı" etiketiyle ayrı bir yapılandırma (`configs/experimental_s0_floor.yaml`) olarak sunulur.
- Final yöntem (A3-s2s-norm) ve demo değişmez.
- Bağımsız doğrulama (görülmemiş veri, tek ölçüm) yapılmadan final başarı denmez.

**Süre:** 00:45'e kadar (kod + koşu + rapor). Log `results/logs/anomaly_floor_channel.log`. Bu deneyden sonra yeni hipotez yok.

## 2. Sonuçlar (koşu 2026-11-09 00:17, 0,6 dk; log `results/logs/anomaly_floor_channel.log`)

- Manifest denetimi geçti.
- Taban: kış 0,155, yaz 0,154 kW. Duyarlılık senaryosunda drop değeri: kış 0,213, yaz 0,207 kW (`anomaly_floor_levels.csv`).
- Enjeksiyonsuz kümelerde taban kuralı **0** alarm bloğu verdi.

**Orta senaryo** (kış / yaz, 10 seed; `anomaly_floor_medium_summary.csv`):

| | S0-yerel | S0+taban |
|---|---|---|
| block_f1 | 0,551 / 0,535 | **0,644 / 0,604** |
| count_f1 | 0,442 / 0,481 | 0,540 / 0,551 |
| block precision | 0,464 / 0,659 | 0,513 / 0,687 |
| count precision | 0,340 / 0,529 | 0,403 / 0,572 |
| olay recall | 0,706 / 0,453 | 0,883 / 0,545 |
| spike | 0,90 / 0,77 | 0,90 / 0,77 |
| drop | 0,00 / 0,00 | **1,00 / 1,00** |
| seviye kayması | 1,00 / 0,90 | 1,00 / 0,90 |
| desen | 0,76 / 0,00 | 0,76 / 0,00 |
| yanlış alarm / gün | 0,098 / 0,025 | 0,097 / 0,025 |
| gecikme (saat) | 2,2 / 5,4 | 1,9 / 4,6 |
| PR-AUC | 0,348 / 0,237 | 0,444 / 0,279 |

**Eşleştirilmiş fark** (%95 GA, n = 10; `anomaly_floor_medium_paired.csv`):

| | Kış | Yaz |
|---|---|---|
| block_f1 | +0,093 [0,029; 0,158] | +0,069 [0,023; 0,115] |
| count_f1 | +0,098 [0,031; 0,165] | +0,070 [0,023; 0,117] |
| yanlış alarm / gün | −0,001 [−0,004; 0,001] | 0,000 |

İki mevsim ortalaması block_f1 farkı +0,081 ± 0,018.

**Duyarlılık senaryosu** (drop değeri tabanın üstünde): S0+taban, S0 ile **birebir aynı**. Drop recall'ü iki mevsimde 0, block_f1 0,551 / 0,535.

**Karar** (`anomaly_floor_decision.csv`): c1–c5, c7 ve c8'in hepsi sağlandı → **önceden yazılan kurala göre kabul edildi.**

**Kabulün anlamı:**
- Kazancın tamamı orta senaryodaki 24 drop olayının (enjekte değer 0,05 kW) taban kuralıyla yakalanmasından geliyor. Duyarlılık senaryosunda kazanç sıfır.
- Bu yöntem **yalnız tam kesinti (alt yükün altına inen) olayları** yakalayan bir ek kanaldır. Genel drop başarısı veya genel F1 artışı iddia edilemez.
- Kural enjeksiyon değerini kullanmıyor (taban yalnız train'den). Ama orta senaryodaki recall tanım gereği yüksek.
- Gerçek veride bu düzeyde kesinti görülmedi (temiz geliştirme kümelerinde 0 alarm). Yaz tatilinde boş ev gibi durumlar başka dönemlerde yanlış alarm üretebilir; bu bilinmiyor.
- k = 2 değeri test sonrası A6'da kış-dev (= val-dev) üzerinde seçilmişti; kış-dev bu parametre için bağımsız değil.

**Durum:**
- Deneysel yapılandırma: `configs/experimental_s0_floor.yaml`.
- **Final yöntem A3-s2s-norm değişmedi; demo değişmedi.**
- Bağımsız doğrulama yapılmadı (görülmemiş veride tek ölçüm gerekir; test kümesi daha önce görüldüğü için bu amaçla kullanılamaz).
