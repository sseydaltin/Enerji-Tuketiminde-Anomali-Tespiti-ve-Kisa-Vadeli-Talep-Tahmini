# Kanal başına ayrı yumuşatma: A3 (w=1) VEYA C1-L2 (w=6) — KEŞİFSEL deney

Dal: `dilara/anomali-adaylar` · Komut: `python -m src.anomaly_or_channels` · Geliştirme sonucu, final değil.

## 1. Protokol (koşudan önce yazıldı, 2026-11-09 00:09; sonuç görüldükten sonra değiştirilmedi)

**Keşifsel uyarı.** Bu hipotez, aynı geliştirme kopyalarında görülen sonuçlardan doğdu (`docs/anomaly_candidates.md` §2: C1 yaz desenini yakaladı, spike'ı kaybetti). Aşağıdaki kanal eşikleri de aynı 20 kopyada seçildi. Bu yüzden buradaki her kazanç **seçim yanlılığı** taşır ve bağımsız doğrulanmış başarı sayılmaz. Önceki 90 dakikalık bütçenin devamıdır (bitiş 01:26).

**Neden C2'den farklı?** C2 = smooth_w(max(A3', C1')): birleşimden **sonra** ortak pencere (seçilen w = 3). Bu yüzden A3'ün tek saatlik sıçramaları seyrelir, C1 de 6 saatlik pencere almaz. Burada her kanal kendi penceresiyle eşiklenir, kararlar OR ile birleşir; birleşimden sonra yumuşatma yok.

**Tanım.**
- Kısa kanal: a_t = A3-s2s-hmean_t / q_A3 (w = 1). Alarm: a_t ≥ T₁.
- Uzun kanal: c_t = ort(C1-L2_{t−5..t}) (w = 6, yalnız geçmiş). Alarm: c_t ≥ T₂.
- Alarm = (a_t ≥ T₁) VEYA (c_t ≥ T₂). Eşdeğer skor r_t = max(a_t / T₁, c_t / T₂); alarm r_t ≥ 1. PR-AUC bu skorla hesaplanır. Bir kanalın NaN olması diğerini engellemez.
- Kanal seçimi için anomali tipi veya etiket **kullanılmaz**; iki kanal her saatte birlikte çalışır.

**Aşama A – sabit eşikler (yeni arama yok):**
- T₁ = 1,1636 (S0'ın Aşama 3 eşiği).
- T₂ = 7,0525 (C1-L2'nin tek başına seçilen eşiği, w = 6; `anomaly_candidates_selection.csv`).

**Aşama B – ek eşik seçimi (yalnız Aşama A kabul edilmezse çalışır):**
- Gerekçe: Tek başına seçilen T₂, C1'in kendi yanlış alarmlarını dengeler. OR'da bu alarmlar A3'ünkilere eklenir; daha yüksek bir T₂ gerekebilir.
- Aday listesi (çok küçük, sabit): T₁ sabit = 1,1636; T₂ ∈ {1,25; 1,5; 2,0} × 7,0525.
- Seçim kuralı (mevcut eşik kuralıyla aynı amaç): 20 kopyada birleşik (pooled) block_f1 en büyük olan; eşitlikte yüksek T₂.
- **Seçim yanlılığı:** Seçim ve rapor aynı kopyalarda yapılır.

**Karşılaştırılanlar** (aynı veri, ağırlıklar, seed'ler; kış-dev + yaz-dev, orta senaryo, 10 seed / küme):
- S0-yerel
- C1-L2 (w = 6, T₂)
- OR-A
- OR-B (yalnız Aşama B çalışırsa)

**Kabul ölçütü.** Önceki c1–c5 (S0-yerel'e karşı) ile bu hipoteze özgü iki koşul. Hepsi sağlanmalı:
1. İki mevsim ortalaması block_f1 farkı > 2·se.
2. Her iki mevsimde block_f1 farkı > 0.
3. Her mevsimde yanlış alarm artışı ≤ 0,05/gün.
4. Her mevsimde level_shift ve pattern recall kaybı ≤ 0,05.
5. İki mevsim ortalaması count_f1 farkı > 0.
6. Yaz desen recall'ü S0'dan yüksek.
7. Her mevsimde spike recall kaybı ≤ 0,05.

Aşama B kabul edilse bile sonuç "seçim yanlılıklı keşifsel bulgu" diye raporlanır. Final modele ve demoya eklenmez.

**Ek raporlar:**
- Olay başına hangi kanalın yakaladığı: yalnız A3 / yalnız C1 / ikisi / hiçbiri; tip ve mevsim bazında.
- OR'un S0'a eklediği yanlış alarm blokları (S0'ın alarmıyla kesişmeyen yanlış bloklar) ve bunların kaçının enjeksiyonsuz veride de alarm verdiği (etiketsiz arka plan).

**Süre ve durdurma.**
- Kod + koşu + rapor 01:26'ya kadar. Betik 01:15'ten sonra yeni aşamaya geçmez.
- Log: `results/logs/anomaly_or_channels.log`.
- Kabul edilmezse arama genişletilmez, başka yöntem başlatılmaz.

## 2. Sonuçlar (koşu 2026-11-09 00:09, 0,8 dk; log `results/logs/anomaly_or_channels.log`)

- Manifest denetimi geçti. S0-yerel ve C1-L2, `anomaly_candidates` koşusuyla birebir aynı.
- Aşama A kabul edilmediği için Aşama B çalıştı:

| T₂ çarpanı | 20 kopyada birleşik block_f1 |
|---|---|
| ×1,25 | **0,5607 → seçildi** (T₂ = 8,816) |
| ×1,5 | 0,5591 |
| ×2,0 | 0,5507 |

- **OR-B seçim yanlılığı taşır:** eşik, raporlandığı kopyalarda seçildi.

**Kopya ortalamaları** (kış / yaz; `anomaly_or_channels_summary.csv`):

| | S0-yerel | C1-L2 | OR-A (sabit eşikler) | OR-B (T₂ seçildi) |
|---|---|---|---|---|
| block_f1 | 0,551 / 0,535 | 0,475 / 0,513 | 0,490 / **0,605** | 0,510 / **0,614** |
| count_f1 | 0,442 / 0,481 | 0,450 / 0,491 | 0,402 / 0,536 | 0,415 / 0,545 |
| block precision | 0,464 / 0,659 | 0,409 / 0,510 | 0,364 / 0,539 | 0,385 / 0,575 |
| count precision | 0,340 / 0,529 | 0,383 / 0,469 | 0,283 / 0,447 | 0,295 / 0,471 |
| olay recall | 0,706 / 0,453 | 0,613 / 0,556 | 0,785 / 0,711 | 0,785 / 0,681 |
| spike | 0,90 / 0,77 | 0,10 / 0,14 | **0,90 / 0,77** | 0,90 / 0,77 |
| drop | 0,00 / 0,00 | 0,00 / 0,00 | 0,00 / 0,00 | 0,00 / 0,00 |
| seviye kayması | 1,00 / 0,90 | 0,90 / 0,95 | 1,00 / 0,95 | 1,00 / 0,95 |
| desen | 0,76 / 0,00 | 0,96 / 0,67 | **0,96 / 0,67** | 0,96 / 0,53 |
| yanlış alarm / gün | 0,098 / 0,025 | 0,067 / 0,039 | 0,145 / 0,055 | 0,136 / 0,048 |
| gecikme (saat) | 2,2 / 5,4 | 3,9 / 6,0 | 2,0 / 4,1 | 2,2 / 4,3 |
| PR-AUC | 0,348 / 0,237 | 0,251 / 0,235 | 0,296 / 0,267 | 0,304 / 0,275 |

**S0-yerel'e karşı eşleştirilmiş fark** (%95 GA, n = 10):

| | OR-A kış | OR-A yaz | OR-B kış | OR-B yaz |
|---|---|---|---|---|
| block_f1 | −0,061 [−0,113; −0,009] | +0,070 [0,007; 0,132] | −0,040 [−0,093; 0,012] | +0,079 [−0,007; 0,165] |
| count_f1 | −0,040 [−0,079; −0,002] | +0,055 [−0,014; 0,124] | −0,027 [−0,067; 0,013] | +0,064 [−0,022; 0,149] |
| block precision | −0,100 | −0,120 | −0,079 | −0,084 |
| olay recall | +0,079 | +0,258 [0,134; 0,381] | +0,079 | +0,228 |
| desen recall | +0,20 [0,01; 0,39] | +0,67 [0,44; 0,90] | +0,20 | +0,53 |
| spike recall | 0 | 0 | 0 | 0 |
| yanlış alarm / gün | +0,047 [0,041; 0,052] | +0,030 [0,027; 0,033] | +0,038 | +0,023 |

**Hangi kanal yakaladı?** OR-A, 10 seed toplamı, olay sayısı (`anomaly_or_channels_attribution.csv`):

| Tip | Kış: yalnız A3 / yalnız C1 / ikisi / hiçbiri | Yaz: yalnız A3 / yalnız C1 / ikisi / hiçbiri |
|---|---|---|
| spike | 15 / 0 / 3 / 3 | 15 / 0 / 5 / 5 |
| drop | 0 / 0 / 0 / 16 | 0 / 0 / 0 / 8 |
| seviye kayması | 1 / 0 / 10 / 0 | 0 / 1 / 17 / 1 |
| desen | 0 / 6 / 20 / 1 | 0 / 19 / 0 / 12 |

**Birleşimin eklediği yanlış alarmlar** (kopya başına; `anomaly_or_channels_added_false_alarms.csv`):
- OR-A: S0'ın alarmına dokunmayan yeni yanlış bloklar kışta 4,3, yazda 3,6. Bunların kışta 4,2'si, yazda 3,6'sı enjeksiyonsuz veride de alarm veriyor (etiketsiz arka plan).
- OR-B: kış 3,5, yaz 2,7.

**Karar** (`anomaly_or_channels_decision.csv`): **OR-A ve OR-B kabul edilmedi.**
- Sağlanmayan koşullar: c1 (ortalama farkı gürültünün üstünde değil; OR-A +0,004 ± 0,018, OR-B +0,019 ± 0,022) ve c2 (kışta block_f1 düşüyor).
- Sağlanan koşullar: c3, c4, c5, c6 (yaz deseni arttı), c7 (spike korunuyor).

**Soruya cevap: "Yaz desen recall'ünü artırırken A3'ün spike başarısını ve kabul edilebilir yanlış alarm yükünü koruyabildik mi?"**
- **Desen ve spike: evet.** Yaz desen recall'ü 0 → 0,67 (OR-A). Spike recall'ü iki mevsimde de S0 ile aynı (0,90 / 0,77). Yaz block_f1 +0,070 (GA sıfırı dışlıyor).
- **Yanlış alarm: kısmen.** Artış önceden konan 0,05/gün sınırının altında. Ama kışta +0,047 (sınıra çok yakın, %48 artış), yazda +0,030 (yaklaşık 2,2 kat).
- **Bedel:** C1 kanalı kışta S0'ın zaten yakaladığı desenlere çoğunlukla ek bir şey katmıyor (20 olayı ikisi birlikte yakalıyor). Buna karşılık kopya başına ~4 ek yanlış blok getiriyor; bunlar etiketsiz arka plan alarmları. Bu yüzden kış block_f1 düşüyor (−0,061).
- **Sonuç:** Ortalama artmıyor (+0,004). Bir mevsimdeki kazanç diğerindeki kayıpla dengeleniyor.
- Drop iki kanalda da yakalanmıyor (0 / 24 olay).

Protokol gereği arama genişletilmedi, başka yöntem başlatılmadı. Final model ve demo değişmedi.
