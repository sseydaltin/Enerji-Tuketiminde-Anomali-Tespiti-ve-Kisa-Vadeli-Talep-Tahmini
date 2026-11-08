# Anomali aday yöntemleri (geliştirme deneyi, final değil)

Dal: `dilara/anomali-adaylar` (taban `b17ba84`) · Komut: `python -m src.anomaly_candidates`

## 1. Protokol (koşudan önce yazıldı, 2026-11-08 23:58; sonuç görüldükten sonra değiştirilmedi)

**Sabit başlangıç.**
- Kümeler: kış-dev (2009-10-01 – 2009-12-31) ve yaz-dev (2009-06-01 – 2009-09-30).
- Senaryo: orta senaryo. Enjeksiyon seed'leri config'teki `development.scenarios.medium.seeds` listesinden.
- Ağırlıklar: `results/tables/repro_manifest.csv` hash'leri koşu başında denetlenir.
- Metrikler: `block_f1` (mevcut olay F1), `count_f1` (olay sayımlı; birebir eşleştirme değil), precision, recall, tip recall'ü, günlük yanlış alarm, tespit gecikmesi, PR-AUC.
- val-hold ve test kullanılmaz. Tahmin modeli eğitilmez.

**Referanslar (aynı kopyalarda yeniden ölçülür, seçim yapılmaz):**
- **S0-yerel** (A3-s2s-hmean): w = 1, eşik 1,1636 · q_A3 (Aşama 3 seçimi). Kabul ölçütünün referansı budur.
- **A0** (hareketli z): w = 6, eşik 3,989 (Aşama 2 seçimi).
- **A3-s2s-norm** (final yöntemin dev hâli): w = 1, eşik 4,241 (Aşama 2 seçimi).

**Mevcut hatalar (önceki bulgular):**
- (a) A0'ın referansı son 7 günün ortalama ve std'si. Önceki günlerdeki anomaliler ve uç değerler ortalamayı kaydırıyor, std'yi şişiriyor. 7 örnekle std gürültülü. Yazın yanlış alarm yüksek (kopya başına 11,5 blok).
- (b) A3'ün ölçeği bütün train dönemi üzerinden saat × gün std'si; yaz ölçeğine uyum sağlamıyor. Yaz desen recall'ü 0.
- (c) Drop'lar eşiğin hemen altında kalıyor.

**Adaylar ve hedefledikleri hata:**

| Aday | Tanım | Hedeflenen hata |
|---|---|---|
| **C1 – sağlam profil sapması** | Gün tipi d = hafta içi / (hafta sonu veya tatil); takvimden, önceden bilinir. Referans kümesi R_t = son L haftadaki aynı saat ve aynı gün tipindeki gözlemler (t − 24k, k = 1…7L); yalnız geçmiş. m_t = medyan(R_t). s_t = max(1,4826 · MAD(R_t), taban(saat, d)); taban = train'de hesaplanan s'nin o hücredeki 25. yüzdeliği. Skor \|y_t − m_t\| / s_t; en az 4 referans yoksa NaN | (a) medyan/MAD, referanstaki anomalilerden etkilenmez; L haftalık daha çok örnek; train'den öğrenilen taban sahte büyük z'yi önler. (b) Ölçek son haftalara uyar (mevsime adaptif). Gece referansları sakin olduğu için gece desen sapması görünür olabilir. (c) Mutlak düzey sapması drop'u doğrudan görür |
| **C2 – C1 + A3 birleşimi** | max(A3-s2s-hmean / q_A3, C1 / q_C1); q'lar train'in (A3 için OOF) 99. yüzdeliği; ağırlık 1 | Model artığı ve tipik profil farklı bilgi taşır; biri kaçırınca öteki yakalayabilir |

**Reddedilen seçenekler:**
- **CUSUM/EWMA ve ayrı kısa/uzun kanallar:** Max birleşimi (OR kuralı) ve geçmişe bakan ortalama olarak Aşama 3 (S1–S5) ve E1'de denendi, kabul edilmedi. Eşdeğerini tekrar etmiyorum.
- **Alt sayaçlarla tutarlılık özelliği:** Enjeksiyon yalnız toplam gücü değiştiriyor, alt sayaçlara dokunmuyor. Bu özellik enjeksiyon yapaylığını ezberler; yasak.
- **Asimetrik/kantil tabanlı tahmin aralığı:** Drop için ayrı ölçek Aşama 3 S2'de denendi; recall arttı, F1 artmadı.
- **Yeni öğrenilen model:** Eğitim bütçesi yok, A1/A2 zaten kötü.

> **Düzeltme notu (2026-11-09 00:08, koşudan sonra eklendi; yukarıdaki protokol metni değiştirilmedi):**
> CUSUM/EWMA'nın ve asimetrik tahmin aralığının önceki yöntemlerle "eşdeğer" olduğu **kanıtlanmadı**.
> - Aşama 3 (S2: ayrı drop ölçeği) ve E1 (geçmişe bakan işaretli ortalama) bu fikirlere yalnız **benzer**.
> - CUSUM'un birikimli toplamı ve sıfırlama kuralı, E1'in sabit pencereli ortalamasıyla aynı değildir.
> - Kantil tabanlı aralık, S2'nin q99 ölçeğiyle aynı değildir.
> - Bu seçenekler zaman bütçesi ve öncelik nedeniyle seçilmedi; denenmiş ve başarısız olmuş sayılmamalıdır.

**Parametre bütçesi (küçük, sabit):**
- C1: L ∈ {2, 4} hafta. Skor yumuşatma w ∈ {1, 3, 6} (mevcut seçim kuralı).
- C2: C1'in seçilen L'si. w ∈ {1, 3, 6}.
- Eşik her zaman mevcut kuralla seçilir: iki kümenin kopyaları birlikte, birleşik block_f1'i en büyük yapan yüzdelik, 0,70–0,9995, 300 nokta.
- Pencere seçimi kopya ortalaması block_f1 ile; eşitlikte küçük w.

**Pilot (eleme).**
- Her kümenin ilk 3 seed'i (toplam 6 kopya); eşik ve pencere bu 6 kopyada seçilir.
- Aday, pilotta S0-yerel'den (aynı 6 kopya) iki mevsim ortalaması block_f1'de **0,03'ten fazla düşükse** elenir.
- C2 yalnız C1 elenmezse veya C1'in tip recall'ü S0'ı tamamlıyorsa (S0'ın kaçırdığı bir tipte C1 recall'ü ≥ 0,3) çalıştırılır.

**Tam değerlendirme.** Elenmeyen adaylar 10 seed / küme ile, seçim 20 kopyada yeniden yapılarak değerlendirilir.

**Kabul ölçütü (S0-yerel'e karşı, E1 ile aynı + count_f1):**
1. İki mevsim ortalaması block_f1 farkı > 2·se.
2. Her iki mevsimde ortalama block_f1 farkı > 0.
3. Her mevsimde günlük yanlış alarm artışı ≤ 0,05.
4. Her mevsimde level_shift ve pattern recall kaybı ≤ 0,05.
5. İki mevsim ortalaması count_f1 farkı > 0.

Birden fazla aday kabul edilirse en basiti (C1 < C2) seçilir; ancak C2'nin farkı C1'e göre gürültü sınırının (2·se) üstündeyse C2 seçilir. Hiçbiri kabul edilmezse S0 kalır.

**Süre.**
- Toplam 90 dakika (23:57 – 01:27). Pilot ≤ 10 dk, tam değerlendirme ≤ 20 dk koşu.
- Betik 75. dakikadan sonra yeni adıma geçmez, o ana kadarki çıktıları yazar.
- Log: `results/logs/anomaly_candidates.log`.

**Durdurma.** Kabul edilmeyen aday için L, w veya ağırlık aranmaz; yeni aday eklenmez.

## 2. Sonuçlar (koşu 2026-11-09 00:00–00:02; log `results/logs/anomaly_candidates.log`)

**Akış:**
- Manifest denetimi geçti.
- Pilot (3 seed / küme, S0-yerel pilot block_f1 0,517):
  - C1-L2: 0,507 → devam.
  - C1-L4: 0,482 → **elendi** (S0'dan 0,035 düşük).
  - C2-L2: 0,513 → devam. C1, S0'ın kaçırdığı yaz deseninde recall ≥ 0,3 verdiği için çalıştırıldı.
- Tam değerlendirme (10 seed / küme, 20 kopyada seçim):
  - C1-L2: w = 6, eşik 7,052.
  - C2-L2: w = 3, eşik 0,786.
- İlk koşu C2 kararı satırında bir tip hatası yüzünden durdu. Düzeltilip baştan çalıştırıldı; sonuçlar deterministik, pilot değerleri aynı çıktı.
- Toplam koşu yaklaşık 1 dakika. Aday başına seçim + değerlendirme yaklaşık 9 saniye.

**Referans kontrolü:** S0-yerel, A0 ve A3-s2s-norm, `anomaly_eval_audit` ve E1 değerleriyle birebir aynı.

**Kopya ortalamaları** (orta senaryo, kış / yaz; `anomaly_candidates_summary.csv`):

| Yöntem | block_f1 | ort. | count_f1 | block P | count P | recall | spike | drop | seviye | desen | YA/gün | gecikme (s) | PR-AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S0-yerel | 0,551 / 0,535 | 0,543 | 0,442 / 0,481 | 0,464 / 0,659 | 0,340 / 0,529 | 0,706 / 0,453 | 0,90 / 0,77 | 0,00 / 0,00 | 1,00 / 0,90 | 0,76 / 0,00 | 0,098 / 0,025 | 2,2 / 5,4 | 0,348 / 0,237 |
| A0 | 0,518 / 0,310 | 0,414 | 0,506 / 0,275 | 0,508 / 0,260 | 0,491 / 0,217 | 0,567 / 0,440 | 0,14 / 0,03 | 0,00 / 0,00 | 0,90 / 0,95 | 0,77 / 0,40 | 0,041 / 0,097 | 3,8 / 6,6 | 0,276 / 0,141 |
| A3-s2s-norm | 0,541 / 0,529 | 0,535 | 0,443 / 0,489 | 0,418 / 0,625 | 0,320 / 0,535 | 0,797 / 0,462 | 0,90 / 0,80 | 0,15 / 0,00 | 1,00 / 0,90 | 0,90 / 0,00 | 0,123 / 0,025 | 2,5 / 5,6 | 0,258 / 0,193 |
| **C1-L2** | 0,475 / 0,513 | 0,494 | 0,450 / 0,491 | 0,409 / 0,510 | 0,383 / 0,469 | 0,613 / 0,556 | 0,10 / 0,14 | 0,00 / 0,00 | 0,90 / 0,95 | **0,96 / 0,67** | 0,067 / 0,039 | 3,9 / 6,0 | 0,251 / 0,235 |
| **C2-L2** | 0,478 / 0,546 | 0,512 | 0,405 / 0,485 | 0,361 / 0,554 | 0,291 / 0,449 | 0,753 / 0,552 | 0,61 / 0,53 | 0,24 / 0,00 | 1,00 / 0,95 | 0,96 / 0,36 | 0,132 / 0,043 | 2,1 / 5,1 | 0,371 / 0,291 |

**S0-yerel'e karşı eşleştirilmiş fark** (%95 GA, n = 10; `anomaly_candidates_paired.csv`):

| | C1-L2 kış | C1-L2 yaz | C2-L2 kış | C2-L2 yaz |
|---|---|---|---|---|
| block_f1 | −0,075 [−0,151; 0,000] | −0,022 [−0,104; 0,060] | −0,072 [−0,128; −0,017] | +0,011 [−0,071; 0,092] |
| count_f1 | +0,008 [−0,085; 0,100] | +0,010 [−0,082; 0,101] | −0,038 [−0,094; 0,018] | +0,004 [−0,071; 0,078] |
| desen recall | +0,20 [0,01; 0,39] | **+0,67 [0,44; 0,90]** | +0,20 [0,01; 0,39] | +0,36 [0,09; 0,63] |
| spike recall | −0,80 [−0,99; −0,61] | −0,63 [−0,98; −0,29] | −0,29 [−0,55; −0,02] | −0,24 [−0,55; 0,07] |
| yanlış alarm / gün | −0,030 | +0,013 | +0,034 | +0,018 |
| gecikme (saat) | +1,7 [0,4; 3,0] | +0,6 | −0,1 | −0,4 |

**Karar** (`anomaly_candidates_decision.csv`): **hiçbir aday kabul edilmedi. S0 kalır.**

| Koşul | C1-L2 | C2-L2 |
|---|---|---|
| c1: ortalama block_f1 farkı > 2·se | −0,049 ± 0,025 → hayır | −0,031 ± 0,022 → hayır |
| c2: iki mevsimde kazanç | hayır (iki mevsimde de düşüş) | hayır (kış −0,072) |
| c3: yanlış alarm artışı ≤ 0,05 | evet | evet |
| c4: seviye/desen kaybı ≤ 0,05 | **hayır** (kış seviye kayması 1,00 → 0,90) | evet |
| c5: count_f1 farkı > 0 | evet (+0,009) | hayır (−0,017) |

**Okuma:**
- **C1 ilk kez yaz desenini yakalıyor.** Recall 0 → 0,67; güven aralığı sıfırı dışlıyor. Kışta desen recall'ü 0,96. Mevsime uyumlu, sağlam ölçek hipotezi desen tipi için destekleniyor. Bu yalnız geliştirme verisinde ve çok kez kullanılmış kümelerde görüldü; bağımsız doğrulanmış değil.
- **Bedeli spike.** C1'in seçilen 6 saatlik yumuşatması tek saatlik sıçramaları bastırıyor (spike recall 0,90 → 0,10). Sonuç olarak block_f1 iki mevsimde de düşük.
- **C2 iki bilgiyi birleştiriyor ama tek ortak pencere ve tek eşik ödünleşimi çözmüyor.** Yazda küçük, belirsiz bir kazanç var (+0,011). Kışta kayıp var (−0,072) ve yanlış alarm artıyor (+0,034/gün).
- A0, S0'dan yazın açıkça kötü (block_f1 −0,225), kışın ayırt edilemiyor.

**Protokol gereği yapılmayanlar:** L, w, ağırlık veya eşik aranmadı; yeni aday eklenmedi.

Bu sonuçlardan çıkan ve **ayrıca önceden yazılmadan çalıştırılmaması gereken** hipotez:

> Kanal başına ayrı yumuşatma, yani alarm = (A3 ham, w = 1, ≥ T₁) VEYA (C1, w = 6, ≥ T₂), spike'ı A3'ten, sürekli sapmayı C1'den alabilir.

- Aşama 3 ve E1'deki birleşimler tek bir ortak pencere kullanıyordu; bu birleşim denenmedi.
- Seçim iki eşik içerdiği için aşırı uyum riski daha yüksek. Bağımsız bir kontrol olmadan sonucu iyimser okunmalı.
