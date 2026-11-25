# LEAD val kapanışı — test DEĞİL (2026-11-26)

Bu belge LEAD1.0 val aşamasının kapanışıdır. **Test kümesi hiç kullanılmadı.** Bütün sayılar `results/lead/tables/` dosyalarından betikle alınmıştır (`val_results.csv`, `val_final_summary*.csv`, `val_selection*.csv`, `val_split_composition.csv`, `fe_ablation.csv`). Protokol: `docs/lead_protokol.md` (§18 geçerli seçim kuralı).

## 1. Çerçeve (protokol §18.1)

- LEAD, ana veri setlerindeki (hane, ELD) yaklaşımı gerçek etiketlerle sınayan **ek doğrulamadır**.
- **Birincil metrik: `other` (davranışsal) dilimi val PR-AUC.** Genel ROC-AUC/PR-AUC ikincildir (literatür kıyası).
- Ana senaryo **çevrimiçi (nedensel)**. 3 bina bölmesi seed'i (42, 7, 2026), her birinde 120/40/40 bina. Değerler 3 seed ort. ± std.

## 2. Final val özet tablosu (bina bölmesi; `other` başta)

PR-AUC; satırlar `other` PR-AUC'ye göre sıralı. Taban = dilimin pozitif oranı (rastgele skorun beklenen PR-AUC'si). ROC-AUC, F1 ve olay recall dahil tam tablo: `results/lead/tables/val_final_summary.md` (dilim başına ayrı tablo, genel ROC-AUC önde bir ana tablo ve literatür satırları).

| model | set | other PR-AUC | genel PR-AUC | gap_edge PR-AUC | frozen PR-AUC | paylasilan PR-AUC | tekil PR-AUC | genel ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| L2-FE | behavior | 0.075 ± 0.062 | 0.516 ± 0.086 | 0.181 ± 0.153 | 0.683 ± 0.062 | 0.577 ± 0.108 | 0.240 ± 0.094 | 0.852 ± 0.015 |
| L5-FE | behavior | 0.074 ± 0.057 | 0.583 ± 0.038 | 0.294 ± 0.122 | 0.797 ± 0.092 | 0.650 ± 0.027 | 0.341 ± 0.040 | 0.872 ± 0.015 |
| L1 | behavior | 0.058 ± 0.011 | 0.575 ± 0.017 | 0.292 ± 0.051 | 0.791 ± 0.085 | 0.635 ± 0.027 | 0.361 ± 0.055 | 0.857 ± 0.005 |
| L0-A0 | - | 0.057 ± 0.006 | 0.361 ± 0.011 | 0.105 ± 0.029 | 0.474 ± 0.044 | 0.372 ± 0.016 | 0.218 ± 0.032 | 0.720 ± 0.028 |
| L1-FE | behavior | 0.055 ± 0.039 | 0.572 ± 0.035 | 0.288 ± 0.110 | 0.803 ± 0.104 | 0.636 ± 0.024 | 0.350 ± 0.037 | 0.863 ± 0.016 |
| L4 | - | 0.055 ± 0.022 | 0.140 ± 0.039 | 0.169 ± 0.031 | 0.076 ± 0.031 | 0.122 ± 0.036 | 0.062 ± 0.021 | 0.601 ± 0.013 |
| L5-FE | full | 0.053 ± 0.031 | 0.668 ± 0.101 | 0.485 ± 0.068 | 0.884 ± 0.054 | 0.748 ± 0.070 | 0.390 ± 0.099 | 0.914 ± 0.036 |
| L1-FE | full | 0.053 ± 0.030 | 0.667 ± 0.099 | 0.505 ± 0.089 | 0.882 ± 0.054 | 0.749 ± 0.073 | 0.387 ± 0.094 | 0.912 ± 0.033 |
| L4-semi | - | 0.048 ± 0.015 | 0.431 ± 0.035 | 0.265 ± 0.021 | 0.560 ± 0.043 | 0.481 ± 0.048 | 0.244 ± 0.054 | 0.718 ± 0.023 |
| L2-FE | full | 0.045 ± 0.019 | 0.594 ± 0.086 | 0.183 ± 0.051 | 0.792 ± 0.018 | 0.661 ± 0.050 | 0.260 ± 0.085 | 0.903 ± 0.037 |
| L2 | behavior | 0.032 ± 0.015 | 0.499 ± 0.085 | 0.176 ± 0.091 | 0.693 ± 0.063 | 0.574 ± 0.078 | 0.225 ± 0.123 | 0.830 ± 0.028 |
| L1 | full | 0.029 ± 0.012 | 0.627 ± 0.089 | 0.502 ± 0.049 | 0.859 ± 0.067 | 0.690 ± 0.064 | 0.380 ± 0.067 | 0.898 ± 0.028 |
| L2 | full | 0.028 ± 0.011 | 0.520 ± 0.098 | 0.277 ± 0.165 | 0.697 ± 0.068 | 0.599 ± 0.098 | 0.229 ± 0.120 | 0.838 ± 0.026 |
| IF | behavior | 0.012 ± 0.003 | 0.089 ± 0.051 | 0.006 ± 0.004 | 0.088 ± 0.062 | 0.066 ± 0.051 | 0.033 ± 0.012 | 0.752 ± 0.023 |
| L0-R1 | - | 0.007 ± 0.002 | 0.054 ± 0.009 | 0.291 ± 0.062 | 0.013 ± 0.003 | 0.051 ± 0.012 | 0.014 ± 0.000 | 0.529 ± 0.003 |
| L0-R3 | - | 0.007 ± 0.002 | 0.225 ± 0.154 | 0.073 ± 0.053 | 0.290 ± 0.184 | 0.217 ± 0.141 | 0.084 ± 0.089 | 0.726 ± 0.051 |
| L0-normal | - | 0.007 ± 0.002 | 0.023 ± 0.002 | 0.003 ± 0.001 | 0.013 ± 0.003 | 0.015 ± 0.003 | 0.009 ± 0.000 | 0.500 ± 0.000 |
| L0-R2 | - | 0.007 ± 0.002 | 0.254 ± 0.144 | 0.006 ± 0.002 | 0.404 ± 0.189 | 0.231 ± 0.119 | 0.124 ± 0.104 | 0.749 ± 0.054 |
| taban | | 0.0073 | 0.0230 | 0.0026 | 0.0134 | 0.0145 | 0.0088 | 0,5 |

Notlar:
- L3 (CNN+GRU) eğitilmedi (§17). IF yalnız literatür referansıdır.
- L4-semi ve G7 kullanan FE satırları için §4'teki erken durdurma notu geçerlidir.
- L-FE ablasyonunda seed başına seçilen gruplar (yalnız train CV): 42 → G0+G4+G5; 7 → G0+G4+G5+G6+G7; 2026 → G0+G1+G2+G4+G5+G7.

## 3. Final model önerisi (§18.2)

**Önerilen: L5-FE / full** (`other` PR-AUC 0.053, genel PR-AUC 0.668).

Kuralın adımları (`val_selection_v2.csv`):
1. `other` PR-AUC'de en yüksek ortalama: L2-FE / behavior (0.075). Gürültü düzeyindekiler (C1): 17 / 17 modelin tamamı; L0-normal (taban) bile. **`other` PR-AUC, 3 seed'le (seed başına 40 val binası) modelleri ayırt edemiyor**: en iyinin seed'ler arası std'si büyük olduğu için 2·std/√3 sınırı geniş.
2. C1 içinde genel PR-AUC'de en yüksek: L5-FE / full. Gürültü düzeyindekiler (C2): L1/behavior, L5-FE/full, L1-FE/full.
3. C2'de sinir ağı içeren model: L5-FE/full → seçilir.

**Önemli çekince:** L5-FE'nin L1-FE ağırlığı (w_L1, val genel PR-AUC ile seçildi) — behavior/7: 0.6, behavior/42: 0.8, behavior/2026: 0.7, full/7: 0.7, full/42: 1.0, full/2026: 1.0. full varyantında seed 42 ve 2026'da w = 1,0: L5-FE full bu iki seed'de **L1-FE ile aynıdır** (GRU'nun katkısı 0). "Sinir ağı içeren model" tercihi bu seçimde büyük ölçüde biçimseldir. Eski §12 kuralı (genel PR-AUC, basitlik) **L1-FE / full** seçerdi (tarihçe için).

**Önerinin özü:** çevrimiçi senaryoda en iyi genel sonuç L1-FE/L5-FE full ailesinden geliyor; `other` diliminde hiçbir model 3 seed'le tabandan/birbirinden istatistiksel olarak ayrışmıyor (ortalamalar 0,03–0,07; taban ≈ 0,007).

## 4. L4-semi erken durdurma kontrolü

Kod (`src/pipelines/lead_models.py::train_a3`): val kaybı `target_ok` maskesiyle hesaplanıyor; semi modunda bu maske **val binalarının etiketli anomali saatlerini de kayıptan çıkarıyor**. Yani L4-semi'nin erken durdurması ve LR planlayıcısı **val etiketlerini kullanıyor**. Protokol val'de erken durdurmaya izin veriyor (§9), ama L4-semi "etiketsiz" değil ve val etiketini dolaylı görüyor. G7 (L4-semi artığı) val değeri de bu modelden geldiği için L1-FE/L2-FE/L5-FE'nin G7'li seed'leri aynı nottan etkilenir. Tabloda (`val_final_summary.md`, açıklama sütunu) not düşüldü. Train binaları için G7 OOF değerleri erken durdurmasız (sabit epoch) üretildi; onlarda val kullanılmadı.

## 5. Site bölmesindeki yüksek `other` skorlarının analizi

Siteye göre bölmede (val, aynı modeller, ayarlar bina bölmesinden) skorlar **daha yüksek**:

| model | set | other PR-AUC | genel PR-AUC | paylasilan PR-AUC | tekil PR-AUC | genel ROC-AUC |
|---|---|---|---|---|---|---|
| L5-FE | behavior | 0.132 ± 0.035 | 0.690 ± 0.204 | 0.701 ± 0.183 | 0.517 ± 0.306 | 0.882 ± 0.079 |
| L5-FE | full | 0.131 ± 0.028 | 0.815 ± 0.074 | 0.829 ± 0.059 | 0.593 ± 0.207 | 0.938 ± 0.025 |
| L4-semi | - | 0.121 ± 0.034 | 0.531 ± 0.173 | 0.550 ± 0.147 | 0.283 ± 0.234 | 0.750 ± 0.134 |
| L1-FE | behavior | 0.120 ± 0.025 | 0.683 ± 0.211 | 0.693 ± 0.190 | 0.521 ± 0.311 | 0.871 ± 0.091 |
| L4 | - | 0.119 ± 0.043 | 0.250 ± 0.127 | 0.239 ± 0.114 | 0.083 ± 0.080 | 0.674 ± 0.111 |
| L2-FE | full | 0.108 ± 0.025 | 0.770 ± 0.090 | 0.790 ± 0.073 | 0.480 ± 0.246 | 0.929 ± 0.028 |
| L1-FE | full | 0.098 ± 0.028 | 0.807 ± 0.078 | 0.820 ± 0.064 | 0.594 ± 0.215 | 0.934 ± 0.027 |
| L2-FE | behavior | 0.084 ± 0.035 | 0.655 ± 0.182 | 0.668 ± 0.163 | 0.426 ± 0.278 | 0.874 ± 0.081 |
| L0-A0 | - | 0.044 ± 0.034 | 0.423 ± 0.088 | 0.426 ± 0.059 | 0.251 ± 0.239 | 0.742 ± 0.036 |
| L0-R1 | - | 0.004 ± 0.002 | 0.096 ± 0.021 | 0.098 ± 0.030 | 0.020 ± 0.011 | 0.546 ± 0.018 |
| L0-R3 | - | 0.004 ± 0.002 | 0.477 ± 0.077 | 0.487 ± 0.074 | 0.162 ± 0.028 | 0.789 ± 0.048 |
| L0-normal | - | 0.004 ± 0.002 | 0.025 ± 0.007 | 0.020 ± 0.006 | 0.005 ± 0.001 | 0.500 ± 0.000 |
| L0-R2 | - | 0.004 ± 0.002 | 0.544 ± 0.109 | 0.555 ± 0.124 | 0.331 ± 0.053 | 0.812 ± 0.041 |
| taban | | 0.0041 | 0.0249 | 0.0202 | 0.0048 | 0,5 |

Parça bileşimi (`val_split_composition.csv`; seed başına, train/val):

| bölme | seed | parça | bina | site | anomali oranı | other taban | gap_edge / frozen / other payı | other olayı | paylaşılan other olay payı | other'ın ilk 3 binadaki payı | siteler |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bina | 42 | train | 120 | 12 | 0.0220 | 0.0052 | 0.13 / 0.64 / 0.23 | 1052 | 0.68 | 0.16 | - |
| bina | 42 | val | 40 | 11 | 0.0223 | 0.0092 | 0.12 / 0.48 / 0.41 | 558 | 0.53 | 0.37 | - |
| bina | 7 | train | 120 | 12 | 0.0216 | 0.0058 | 0.12 / 0.61 / 0.27 | 1192 | 0.66 | 0.20 | - |
| bina | 7 | val | 40 | 9 | 0.0256 | 0.0056 | 0.12 / 0.67 / 0.22 | 449 | 0.62 | 0.28 | - |
| bina | 2026 | train | 120 | 12 | 0.0242 | 0.0062 | 0.12 / 0.63 / 0.25 | 1358 | 0.64 | 0.17 | - |
| bina | 2026 | val | 40 | 12 | 0.0213 | 0.0072 | 0.10 / 0.57 / 0.33 | 351 | 0.54 | 0.39 | - |
| site | 42 | train | 68 | 2 | 0.0202 | 0.0034 | 0.18 / 0.65 / 0.17 | 602 | 0.78 | 0.23 | 1,9 |
| site | 42 | val | 70 | 3 | 0.0269 | 0.0067 | 0.13 / 0.63 / 0.24 | 847 | 0.72 | 0.22 | 2,4,14 |
| site | 7 | train | 90 | 11 | 0.0214 | 0.0084 | 0.05 / 0.57 / 0.38 | 1100 | 0.57 | 0.14 | 0,1,2,4,5,7,8,10,12,13,15 |
| site | 7 | val | 52 | 1 | 0.0170 | 0.0025 | 0.26 / 0.60 / 0.14 | 421 | 0.83 | 0.24 | 9 |
| site | 2026 | train | 82 | 9 | 0.0200 | 0.0095 | 0.05 / 0.48 / 0.47 | 979 | 0.53 | 0.19 | 0,2,3,4,5,8,12,13,15 |
| site | 2026 | val | 49 | 2 | 0.0306 | 0.0031 | 0.15 / 0.75 / 0.10 | 450 | 0.64 | 0.26 | 10,14 |

Gözlemler:
1. **Site val parçaları büyük ve az sayıda siteden oluşuyor** (49–70 bina, 1–3 site; seed 7'de yalnız site 9). Sonuçlar bu sitelerin özelliklerine bağlı.
2. **`other` tabanı site val'de daha düşük** (0,0025–0,0067; bina val'de 0,0056–0,0092). Yüksek PR-AUC "kolay taban"dan gelmiyor.
3. **Paylaşılan `other` olaylarının payı site val'de daha yüksek** (0,64–0,83; bina val'de 0,53–0,62). Paylaşılan olaylarda bütün modellerin PR-AUC'si tekil olaylardakinin yaklaşık iki katı (ör. L1-FE full, bina: 0,749'a karşı 0,387 — `val_results.csv`, paylasilan/tekil dilimleri). Site parçası bir sitenin bütün binalarını birlikte içerdiği için, aynı saatte başlayan site çapındaki olaylar val'de "kalabalık" görünür; G5/G7 ve akran benzeri sinyaller bunları daha kolay yakalar.
4. **Bina val'de `other` saatleri birkaç binada yoğun** (ilk 3 bina %28–39; site val'de %22–26): bina bölmesinin `other` sonucu birkaç binanın davranışına çok bağlı, seed'ler arası oynaklığın bir nedeni bu.
5. Sonuç: site bölmesi bu veri setinde "daha zor" bir genelleme sınaması olmadı; farkın en olası açıklaması olay yapısının (paylaşılan olay payı) parçalara farklı dağılması. Bu bir **ilişkidir**, nedensel olarak kanıtlanmadı. Site sonuçları seçimde kullanılmaz (§5, sağlamlık).

## 6. Test kontrol listesi (test YALNIZ kullanıcının açık onayıyla)

- [ ] Seçim kuralı ve önerilen model kullanıcı tarafından onaylandı (§18.2; çekinceler §3).
- [ ] Testte raporlanacak modeller sabit: bütün val satırları (L0, L1, L1-FE, L2, L2-FE, L4, L4-semi, L5-FE, IF; L1-OFF literatür satırı ve I-turu satırları eklenirse onlar da) — testte model eklenmez/çıkarılmaz.
- [ ] Her modelin ayarları, öznitelik grupları (seed başına), eşikleri (val F1-max), L5-FE ağırlıkları ve G7 OOF dosyaları val'de sabitlendi; testte hiçbir şey yeniden seçilmez.
- [ ] Test binaları her bölme seed'i için **bir kez** skorlanır (§13). Test binalarının G7 değeri bütün train binalarıyla eğitilmiş L4-semi'den gelir (test etiketi kullanılmaz; L4-semi'nin erken durdurması val'de kalır).
- [ ] Test komutu ayrı bir aşama olarak yazılacak (`--stage test`); bu kod henüz yok ve kullanıcı onayı olmadan yazılıp çalıştırılmayacak.
- [ ] Raporlama: `other` PR-AUC (birincil), genel ROC-AUC/PR-AUC (literatür), dilimler, olay recall, bina bazında F1 (§11); Kaggle kıyası yalnız koşul farklarıyla (§14).
- [ ] Testten sonra sonuçlara göre hiçbir şey değiştirilmez; değişiklik gerekirse ayrı ve "test sonrası" diye işaretlenir.
- [ ] `pytest` geçiyor; çalışma dalı temiz; `results/lead/` commit'li.

## 7. Açık sorular

1. `other` PR-AUC 3 seed'le modelleri ayırt edemiyor. Birincil metrik böyle kalsın mı, yoksa (test öncesi) daha fazla bölme seed'i mi eklensin?
2. Sinir ağı tercihi L5-FE full'ü seçiyor, ama iki seed'de L5-FE = L1-FE. Öneri L5-FE full olarak mı kalsın, yoksa "gerçekten sinir ağı katkısı olan" bir koşul (ör. w_L1 < 1) mi eklensin?
3. L4-semi'nin val etiketiyle erken durdurması kabul mü, yoksa testten önce val etiketsiz erken durdurmayla (ör. sabit epoch) yeniden mi üretilsin?
4. Literatür satırındaki 0,952 (Tech Science) değerinin künyesi repoda yok; doğrulanmalı.
