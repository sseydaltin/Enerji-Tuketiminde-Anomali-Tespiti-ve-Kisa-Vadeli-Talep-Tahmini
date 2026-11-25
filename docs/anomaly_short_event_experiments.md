# Aşama 3 – A3 tabanlı kısa olay sinyalleri (spike / drop / değişim)

Dal: `seyda/gelistirme` · İş paketi: İP4 (anomali) · Protokol: `docs/gelistirme_protokolu.md`

> **Durum:** Geliştirme (seçim) sonucu. Bağımsız değerlendirme yapılmadı; buradaki hiçbir sayı
> final test başarısı değildir. val-hold ve hane test kümesi bu aşamada **hiç** kullanılmadı.
>
> **Aşama 3 sonucu:** S1–S5 adaylarının hiçbiri önceden sabitlenen karar kuralını geçmedi;
> **A3-s2s-hmean (S0) korunuyor.** Ayrıntı §3–§4.

## 1. Tasarım (sonuçlara bakılmadan, 2026-11-08'de sabitlendi)

Bu bölüm kod yazılıp çalıştırılmadan önce yazıldı; sonuçlar görüldükten sonra değiştirilmedi.

**Araştırma sorusu.** Yönlü tahmin artıkları ve kısa süreli tüketim değişim sinyalleri, A3-s2s-hmean'in
olay F1'ini ve özellikle spike/drop recall'ünü artırır mı?

**Değişmeyenler.** A3-s2s-hmean skoru (çok ufuklu, ufuk başına OOF std'siyle normalize, 24 ufkun
ortalaması), tahmin modeli ağırlıkları (kış-dev: F10-opt; yaz-dev: S-F10-L48), train / kış-dev / yaz-dev
sınırları, enjeksiyon seed'leri ve senaryo tanımları, `event_metrics` ve `alarm_diagnostics`.

**Sinyaller** (hepsi yalnız t ve öncesindeki gözlemlere bakar; merkezli pencere yok):

| Ad | Tanım | Neden |
|---|---|---|
| A3 | `A3-s2s-hmean` = ort_h \|y_t − ŷ_{h,t}\| / σ_h(saat, gün) | mevcut dedektör |
| spike | max(z̄_t, 0), z̄_t = ort_h (y_t − ŷ_{h,t}) / σ_h(saat, gün) | işaret korunur: yalnız beklenenden **yüksek** |
| drop | max(−z̄_t, 0) | yalnız beklenenden **düşük**; tüketim dağılımı sağa çarpık olduğundan negatif kuyruk daha incedir ve ayrı ölçeklenince görünür olur |
| change | \|Δy_t − μ_Δ(saat, gün)\| / σ_Δ(saat, gün), Δy_t = y_t − y_{t−1} | beklenmeyen saatlik değişim; μ, σ train'den (iki saat de doldurulmamış) |

σ_h tabloları mevcut OOF tablolarıdır (kış-dev: 3 kat; yaz-dev: 2009-05-31'e kadarki 2 kat).
change tablosu kış-dev için tüm train'den, yaz-dev için yalnız 2009-05-31'e kadarki train'den.

**Ortak ölçek.** Her sinyal, train'in OOF saatlerindeki (doldurulmamış, enjeksiyonsuz) kendi **99. yüzdeliğine**
bölünür: 1,0 = "train'de 100 saatte bir aşılan düzey". Monoton ölçekleme olduğu için tek başına A3'ün
sıralamasını ve olay F1'ini değiştirmez.

**Deneyler.** Birleşik skor = max(A3', w_s·spike', w_d·drop', w_c·change') (yalnız deneyin içerdiği sinyaller):

| Deney | Sinyaller |
|---|---|
| S0 | A3 (referans) |
| S1 | A3 + spike |
| S2 | A3 + drop |
| S3 | A3 + change |
| S4 | A3 + spike + drop |
| S5 | A3 + spike + drop + change |

Maksimum seçildi, çünkü kısa bir olay yalnız tek bir sinyalde belirgin olabilir; ortalama onu seyreltir.

**Geliştirme kümesinde seçilenler** (yalnız **orta** senaryo, kış-dev + yaz-dev'in 20 kopyası birlikte):
- ağırlıklar w ∈ {0,5; 0,75; 1; 1,5; 2} (her ek sinyal için ayrı; S5'te 125 bileşim),
- yumuşatma penceresi ∈ {1, 3, 6} saat (yalnız geçmişe bakan ortalama, mevcut `smooth_scores`),
- alarm birleştirme boşluğu g ∈ {0, 1, 2} saat (iki alarm bloğu arasındaki ≤ g saatlik boşluk doldurulur;
  olay tanımı değişmez, yalnız alarm maskesi),
- eşik: mevcut kural, birleşik kopyalarda olay F1'i en büyük yapan yüzdelik (ızgara 0,70–0,9995, 300 nokta).

Ölçüt: kopya başına olay F1 ortalaması (iki mevsim 10'ar kopya → eşit ağırlık). Eşitlikte küçük pencere,
küçük boşluk, sonra 1'e yakın ağırlıklar (Σ|log w| küçük). **S0 da aynı pencere/boşluk ızgarasından
geçer** (kazancın sinyalden mi, yumuşatma/birleştirmeden mi geldiği ayrılsın diye). Ayrıca
**S0-kayıtlı** (Aşama 2 ayarı: w=1, g=0, kayıtlı eşik) yeniden üretim kontrolü olarak raporlanır.

Kolay ve zor senaryolar, orta senaryoda seçilen ayarlarla **ek rapordur**. Kolay senaryonun desen tanımı
hatalı olduğu biliniyor (karar bekliyor); kolay sonuçlar yorumlanırken bu not düşülür.

**Raporlananlar.** Her deney × mevsim × senaryo: olay F1, olay precision, olay recall, tip bazında olay
recall'ü (spike, drop, level_shift, pattern), günlük yanlış alarm, tespit gecikmesi (ort., medyan), PR-AUC.
Kopya başına sonuçlar saklanır. S0'a göre **eşleştirilmiş farklar** (aynı seed = aynı enjeksiyon):
ortalama, std, %95 güven aralığı (t dağılımı, n = 10; iki mevsim ortalamasında se = √(se_k² + se_y²) / 2).

**Karar kuralı (önceden).** Sk, S0'a karşı "kabul" edilir ancak:
1. iki mevsim ortalaması olay F1 farkı > 2·se (protokolün gürültü kuralı),
2. her iki mevsimde ortalama fark > 0 (kazanç iki mevsimde de korunuyor),
3. her mevsimde günlük yanlış alarm artışı ≤ 0,05 (≈ 20 günde 1 ek yanlış alarm),
4. her mevsimde level_shift ve pattern olay recall'ü S0'dan en fazla 0,05 düşük.

Kabul edilenler arasında en yüksek iki mevsim F1'i seçilir; daha basit bir kabul edilmişle farkı gürültü
içindeyse basit olan (sıra S0 < S1 = S2 = S3 < S4 < S5). Hiçbiri kabul edilmezse S0 kalır.
Spike/drop sinyalinin "tek başına katkısı" S1/S2'nin S0'a karşı spike/drop recall farkı ve F1 farkıyla yanıtlanır.

**Komut:** `python -m src.anomaly_short_event`

## 2. Yeniden üretim notu (sonuçlardan önce okunmalı)

- **Kış-dev:** S0-kayıtlı (Aşama 2 eşiği, w=1) kayıtlı kopya F1'leriyle **birebir** aynı (en büyük fark 0).
- **Yaz-dev:** S-* ağırlıkları git'te olmadığından bu makinede CPU ile `python -m src.train_summer_dev` ile yeniden
  eğitildi (aynı config ve seed; iç val MAE S-F10-L48 0,3751 / kayıtlı 0,3735, S-F5-L48 0,3915 / 0,3909).
  Ağırlıklar bit düzeyinde aynı değil: kayıtlı eşikle yaz-dev olay F1'i 0,528 (kayıtlı 0,553; kopya başına en büyük
  fark 0,046; `anomaly_short_event_s0_reproduction.csv`). Bu yüzden **yaz-dev sayıları Aşama 2 tablolarıyla
  doğrudan karşılaştırılmaz**. Bu aşamadaki bütün deneyler (S0 dahil) aynı yeniden eğitilmiş ağırlıkları kullandığı
  için **S0–S5 arasındaki eşleştirilmiş karşılaştırma geçerlidir**. Eğitim betiğinin `experiments.csv` ve
  `summer_dev_models.csv` üzerine yazdığı satırlar git'ten geri alındı (mevcut kayıtlar korunur).
- S0'ın eşiği iki kümede yeniden seçildiği için kış-dev S0 değerleri de Aşama 2'den biraz farklıdır
  (ör. olay F1 0,551 / Aşama 2'de 0,556).

## 3. Sonuçlar (orta senaryo = seçim senaryosu; 10 seed / küme)

Ortak ölçek (train OOF 99. yüzdelik; kış / yaz): A3 3,26 / 3,21 · spike 3,26 / 3,20 · drop 2,08 / 1,96 · change 2,98 / 3,00.
Seçilen ayarlar: bütün deneylerde pencere 1, boşluk 0 (yumuşatma ve alarm birleştirme seçilmedi).
Ağırlıklar: S1 w_s 0,75 · S2 w_d 1 · S3 w_c 0,75 · S4 w_s 0,75, w_d 1 · S5 w_s 1, w_d 1, w_c 0,75.

| Deney | Kış F1 | Yaz F1 | Ort. F1 | Fark − S0 (± se) | Kış YA/gün | Yaz YA/gün | Spike rec. (k / y) | Drop rec. (k / y) | Desen rec. (k / y) |
|---|---|---|---|---|---|---|---|---|---|
| S0 | 0,551 | 0,535 | 0,543 | — | 0,098 | 0,025 | 0,90 / 0,77 | 0,00 / 0,00 | 0,76 / 0,00 |
| S1 | 0,551 | 0,535 | 0,543 | 0,000 ± 0,000 | 0,098 | 0,025 | 0,90 / 0,77 | 0,00 / 0,00 | 0,76 / 0,00 |
| S2 | 0,553 | 0,542 | 0,548 | +0,005 ± 0,014 | 0,108 | 0,034 | 0,81 / 0,74 | 0,44 / 0,67 | 0,72 / 0,00 |
| S3 | 0,521 | 0,569 | 0,545 | +0,002 ± 0,014 | 0,099 | 0,026 | 0,81 / 0,83 | 0,00 / 0,08 | 0,72 / 0,00 |
| S4 | 0,553 | 0,542 | 0,548 | +0,005 ± 0,014 | 0,108 | 0,034 | 0,81 / 0,74 | 0,44 / 0,67 | 0,72 / 0,00 |
| S5 | 0,529 | 0,598 | 0,563 | +0,020 ± 0,020 | 0,102 | 0,026 | 0,80 / 0,77 | 0,36 / 0,75 | 0,66 / 0,00 |

YA = yanlış alarm. Drop recall'ü yalnız drop olayı içeren kopyalarda tanımlıdır (her kümede 6/10 kopya); bu yüzden
güven aralıkları geniştir. Seviye kayması recall'ü bütün deneylerde kış 1,00, yaz 0,90–0,95.
Kaynak: `results/tables/anomaly_short_event_summary.csv`, `_by_season.csv`, `_by_type.csv`, `_paired.csv`.

**Eşleştirilmiş farklar (S0'a karşı, %95 GA, orta):**
- S2 drop recall'ü: kış +0,44 [0,04; 0,85], yaz +0,67 [0,13; 1,21]. Olay F1: kış +0,002 [−0,05; 0,06], yaz +0,007 [−0,03; 0,04].
  YA/gün: kış +0,010, yaz +0,008 (iki mevsimde de GA sıfırın üstünde).
- S3 olay F1: kış −0,030 [−0,063; 0,003], yaz +0,034 [−0,020; 0,088]. YA/gün farkı ≈ 0 (+0,001).
- S5 olay F1: kış −0,022 [−0,093; 0,049], yaz +0,063 [0,005; 0,121]; kış desen recall'ü −0,10 [−0,28; 0,07].

**Karar kuralı (§1):** hiçbir deney kabul edilmedi → **S0 (A3-s2s-hmean) kalır** (`anomaly_short_event_decision.csv`).
S2/S4 ölçüt 1'i (gürültü üstü kazanç), S3 ölçüt 1–2'yi, S5 ölçüt 1, 2 ve 4'ü (kış desen recall'ü −0,10 > 0,05) karşılamadı.

**Zor senaryo (ek, orta ayarlarla):** hiçbir deney S0'ı geçmedi. S0 ort. F1 0,263; S2/S4 −0,024 (iki mevsimde),
S3 ≈ 0, S5 kış −0,028 / yaz +0,009. S2 kış drop recall'ü 0,10 → 0,27; spike recall'ü bütün varyantlarda düşük (≤ 0,32).
**Kolay senaryo (ek; desen tanımı hatalı, karar bekliyor):** S0 ile farklar küçük; S2/S4 kış drop recall'ü 0,13 → 0,53.

## 4. Aşama sonu soruları

1. **Spike sinyali tek başına katkı sağlıyor mu?** Hayır. S1, S0 ile bütün kopyalarda aynı alarmları üretti. Nedeni
   yapısal: spike = max(z̄, 0) ≤ ort|z| = A3 ve iki sinyalin 99. yüzdelikleri neredeyse eşit (3,26 / 3,26), dolayısıyla
   w ≤ 1'de spike' hiçbir saatte A3'ü geçemez; w > 1 seçimde F1'i düşürdü. A3 kısa spike'ları orta senaryoda zaten
   yakalıyor (recall kış 0,90, yaz 0,77).
2. **Drop sinyali tek başına katkı sağlıyor mu?** Drop recall'ünü belirgin artırıyor (0 → 0,44 / 0,67; GA sıfırı
   dışlıyor), ama olay F1'ine katkısı gürültü içinde (+0,005 ± 0,014): ek doğru tespitler, ek yanlış alarm blokları
   ve hafif spike recall kaybıyla (eşik yükseldi) dengeleniyor. Zor senaryoda F1'i düşürdü.
3. **Birinci fark sinyali gereksiz yanlış alarm üretiyor mu?** Hayır, YA/gün değişmedi (+0,001; S5'te +0,004 kış).
   Ancak etkisi mevsime göre ters: yazın F1 +0,034, kışın −0,030 (kışın kaçırılan spike/desen olayları).
4. **Birleşim A3'ten daha başarılı mı?** Ortalamada en yüksek S5 (+0,020), fakat fark gürültü sınırında (2·se = 0,040)
   ve kış F1'i ile kış desen recall'ü düşüyor. Önceden belirlenen kurala göre A3'ten daha başarılı değil.
5. **Kazanç iki mevsimde de korunuyor mu?** Yalnız S2/S4'te iki mevsimde de ortalama fark pozitif, ama çok küçük.
   S3 ve S5'in kazancı yalnız yazda; kışta kayıp var.
6. **Yanlış alarm ve gecikme:** drop sinyali YA/gün'ü ≈ +0,01 artırıyor (≈ 100 günde 1 ek); değişim sinyali
   artırmıyor. Ortalama gecikme farklarının GA'ları sıfırı içeriyor (S2 yaz −0,6 saat, S5 kış +0,25 saat).
7. **Normalizasyon geliştirmesine ihtiyaç devam ediyor mu?** Evet. Yaz desen recall'ü bütün deneylerde 0,00 ve
   mevsimler arası fark (kış F1 > yaz F1 S0'da, tersi S3/S5'te) sinyal eklemeyle kapanmadı. Ayrıca drop
   sinyalinin recall kazancının F1'e dönüşmemesi, eşiğin tek global değer olmasıyla da ilişkili olabilir
   (doğrulanmalı; Aşama 6 eşik/alarm konusu).

**Bu sonuçlar geliştirme kümelerindendir; final test başarısı değildir. val-hold ve test kullanılmadı.**
