# E1 – sürekli işaretli artık (geliştirme deneyi, final değil)

Dal: `dilara/inceleme-zorunlu-duzeltmeler` · Komut: `python -m src.anomaly_e1_sustained`

## 1. Protokol (koşudan önce yazıldı, 2026-11-08 23:34; sonuç görüldükten sonra değiştirilmedi)

**Hipotez.** Geçmişe bakan 6 saatlik işaretli artık ortalaması, uzun süre aynı yönde kalan sapmaları
(özellikle yaz deseni ve seviye kayması) saatlik A3-s2s-hmean skorundan daha iyi ayırır.
Gerekçe: saatlik gürültü işaretiyle birlikte ortalanınca kısmen birbirini götürür; sürekli sapma kalır.
Yaz desen sapması 0,50 kW, h=1 MAE 0,34 kW (`docs/inceleme_ve_iyilestirme_plani_2026-11-08.md`).

**Skor.**
- z̄_t = (1/24) Σ_{h=1..24} (y_t − ŷ_{h,t}) / σ_h(saat_t, gün_t). Bu, Aşama 3'teki işaretli ortalama artıktır; z̄ = spike − drop.
- u_t^(w) = | (1/w) Σ_{k=0..w−1} z̄_{t−k} |. Pencerede eksik saat varsa NaN.
- **İşaretin kullanımı:** Ortalama işaretli değerlerle alınır (zıt yönlü hatalar birbirini götürür); mutlak değer ortalamadan **sonra** alınır. Böylece sürekli fazla tüketim de sürekli düşük tüketim de yakalanır.
- E1-w = max( A3_t / q_A3 , u_t^(w) / q_u^(w) ). A3 = A3-s2s-hmean; q = ilgili sinyalin train OOF saatlerindeki 99. yüzdeliği. Ağırlık 1, sabit.

**Yalnız geçmiş bilgi.**
- u_t, z̄_{t−w+1} … z̄_t'ye bakar; z̄_t, y_t ile t'den önce yapılmış tahminleri kullanır. Merkezli pencere yok.
- Test: y_{t+1}'i değiştirmek u_t'yi değiştirmemeli (`tests/test_anomaly_e1_sustained.py`).
- q değerleri yalnız train OOF saatlerinden (kış-dev: 3 kat; yaz-dev: 2009-05-31'e kadarki 2 kat).

**Adaylar (en fazla iki).** w ∈ {3, 6}. Yumuşatma 1 ve alarm birleştirme boşluğu 0 sabit; S0'ın seçilmiş ayarıyla aynı.

**Ortak koşullar (S0 ile aynı).**
- Kış-dev + yaz-dev, orta senaryo, aynı enjeksiyon seed'leri.
- Aynı ağırlıklar: koşu başında `results/tables/repro_manifest.csv` hash'leri denetlenir, uyuşmazsa koşu durur.
- Eşik: mevcut kural, iki kümenin 20 kopyasında birleşik olay F1'ini (blok tanımı) en büyük yapan yüzdelik; 0,70–0,9995, 300 nokta.
- İki aday arasında seçim kopya başına ortalama block_f1 ile; eşitlikte küçük w.
- Test ve val-hold kullanılmaz.

**Referans: S0-yerel.** A3-s2s-hmean, Aşama 3'te seçilen eşikle (1,1636 · q_A3), bu makinedeki ağırlıklarla yeniden ölçülür. Bu bir **yerel geliştirme referansıdır**. Şeyda'nın farklı ağırlıklarla ürettiği sonuçlarla kazanç hesabı yapılmaz.

**Kabul ölçütü (Aşama 3 kuralı + bu hipoteze özgü iki koşul).** Seçilen E1 adayı kabul edilir ancak aşağıdakilerin hepsi sağlanırsa:
1. İki mevsim ortalaması block_f1 farkı > 2·se.
2. Her iki mevsimde ortalama block_f1 farkı > 0.
3. Her mevsimde günlük yanlış alarm artışı ≤ 0,05.
4. Her mevsimde level_shift ve pattern recall'ü S0'dan en fazla 0,05 düşük.
5. İki mevsim ortalaması olay sayımlı F1 (count_f1) farkı > 0.
6. Yaz desen recall'ü S0'dan yüksek.

**Süre bütçesi.** Kod + koşu + değerlendirme toplam 60 dakika (23:34 – 00:34).
- Betik, başlangıçtan 50 dakika sonra yeni adıma geçmez; o ana kadarki çıktıları yazar ve "süre doldu" kaydı düşer.
- İlerleme ve süre `results/logs/E1_sustained.log` dosyasına yazılır.

**Durdurma.** Kabul ölçütü sağlanmazsa w veya ağırlık aranmaz, başka sinyal eklenmez. E2 başlatılmaz.

## 2. Sonuçlar (koşu 2026-11-08 23:35, süre 0,4 dk; log `results/logs/E1_sustained.log`)

- Manifest denetimi geçti (9 dosya).
- Ölçek q99 (kış / yaz): A3 3,261 / 3,208 · u3 2,764 / 2,644 · u6 2,467 / 2,401 (`E1_sustained_scales.csv`).
- Aday seçimi (iki küme, block_f1): E1-w3 0,545 (eşik 1,172) · **E1-w6 0,552 (eşik 1,142) → seçildi**.
- S0-yerel, Aşama 3'teki S0'ı aynı ağırlıklarla birebir veriyor (kış 0,551, yaz 0,535).

**Kopya ortalamaları, orta senaryo, 10 seed / küme** (`E1_sustained_summary.csv`):

| | S0-yerel kış | E1-w6 kış | S0-yerel yaz | E1-w6 yaz |
|---|---|---|---|---|
| block_f1 | 0,551 | 0,575 | 0,535 | 0,529 |
| count_f1 (olay sayımlı) | 0,442 | 0,476 | 0,481 | 0,464 |
| block precision | 0,464 | 0,464 | 0,659 | 0,604 |
| count precision | 0,340 | 0,360 | 0,529 | 0,471 |
| olay recall | 0,706 | 0,775 | 0,453 | 0,472 |
| recall spike / drop | 0,90 / 0,00 | 0,90 / 0,00 | 0,77 / 0,00 | 0,83 / 0,00 |
| recall level_shift / pattern | 1,00 / 0,76 | 1,00 / 0,93 | 0,90 / 0,00 | 0,90 / 0,00 |
| yanlış alarm blok / kopya (arka planda da olan) | 9,0 (8,9) | 9,0 (8,9) | 3,0 (3,0) | 4,0 (4,0) |
| yanlış alarm / gün | 0,098 | 0,098 | 0,025 | 0,034 |
| ort. tespit gecikmesi (saat) | 2,20 | 2,36 | 5,42 | 5,27 |
| PR-AUC (nokta, eşiksiz) | 0,348 | 0,387 | 0,237 | 0,324 |

**Eşleştirilmiş fark, E1-w6 − S0-yerel** (%95 GA, n = 10; `E1_sustained_paired.csv`):

| | Kış | Yaz |
|---|---|---|
| block_f1 | +0,025 [−0,004; 0,053] | −0,006 [−0,031; 0,018] |
| count_f1 | +0,033 [−0,004; 0,070] | −0,017 [−0,042; 0,009] |
| count precision | +0,020 [−0,004; 0,044] | −0,058 [−0,073; −0,043] |
| pattern recall | +0,167 [−0,026; 0,359] | 0,000 |
| yanlış alarm / gün | 0,000 | +0,008 (her seed'de 1 ek blok) |

**Karar** (`E1_sustained_decision.csv`): **kabul edilmedi.**

| Koşul | Sonuç |
|---|---|
| c1: iki mevsim ortalaması block_f1 farkı > 2·se | +0,009 ± 0,008 → sağlanmadı |
| c2: her iki mevsimde fark > 0 | yaz −0,006 → sağlanmadı |
| c3: yanlış alarm artışı ≤ 0,05/gün | sağlandı |
| c4: level_shift / pattern kaybı ≤ 0,05 | sağlandı |
| c5: count_f1 ortalama farkı > 0 | +0,008 → sağlandı |
| c6: yaz desen recall'ü S0'dan yüksek | 0 → 0 → sağlanmadı |

E1-w3 de kabul edilmedi.

**Okuma:**
- Kazanç yalnız kışta: desen recall'ü 0,76 → 0,93 arttı.
- Hipotezin asıl hedefi olan **yaz deseni hiç yakalanmadı** (0 → 0).
- Yazda her seed'de bir ek yanlış blok çıktı (arka plan alarmı; enjeksiyonsuz veride de var), bu yüzden yaz precision'ı düştü.
- Ortalama block_f1 artıyor ama bir mevsim (yaz) kötüleşiyor. Yöntem başarılı ilan edilmez.
- Eşikten bağımsız PR-AUC iki mevsimde de arttı. Bu, sıralama gücünde bir iyileşme işareti; ancak kabul ölçütünün parçası değil ve sonradan ölçüt olarak eklenmez.
- Protokol gereği w veya ağırlık aranmadı, E2 başlatılmadı.
