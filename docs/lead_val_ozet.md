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

> **Güncelleme (2026-11-26, kullanıcı kararları, protokol §21; test öncesi):** sinir ağı tercihi yalnız sinir ağının gerçekten katkı verdiği modellere uygulanır (L5 toplulukları ancak bütün seed'lerde 1 − w > 0 ise). L5-FE full'de seed 42 ve 2026'da w = 1 olduğu için tercih düşer; §18.2 kuralı aynen uygulanınca C2 = {L1 behavior, L1-I full, L1-FE full, L5-FE full} içinden basitlik sırası (L1 ailesi; aile içinde FE'siz < FE; eşitlikte `other` PR-AUC) **L1 behavior**'ı seçer. **Final model: L1 behavior.** Aşağıdaki ilk öneri (L5-FE full) tarihçe için korunur. Seed sayısı 3 kalır; `other` PR-AUC'nin 3 seed'le modelleri ayırt etmemesi bir **bulgudur**. L4-semi'nin val etiketiyle erken durdurulması kabul edildi: **G7 içeren modellerin (L1-FE, L2-FE, L5-FE, L1-I, L2-I, L1-OFF) val değerleri hafif iyimser olabilir; test etkilenmez.**

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

> **Yanıtlandı (2026-11-26):** 1 → seed 3 kalır, bulgu olarak yazılır. 2 → final L1 behavior (§21.2). 3 → kabul, tablolara not. 4 → künye: "An Integrated Framework of Feature Engineering and Machine Learning for Large-Scale Energy Anomaly Detection", *Energy Engineering* (Tech Science Press), c. 123, s. 3 (yazar, yıl, sayfa doğrulanmalı). 5 → val raporu için L1-I'nin değerleri ayrıca verilir; final kuralla belirlendi.

1. `other` PR-AUC 3 seed'le modelleri ayırt edemiyor. Birincil metrik böyle kalsın mı, yoksa (test öncesi) daha fazla bölme seed'i mi eklensin?
2. Sinir ağı tercihi L5-FE full'ü seçiyor, ama iki seed'de L5-FE = L1-FE. Öneri L5-FE full olarak mı kalsın, yoksa "gerçekten sinir ağı katkısı olan" bir koşul (ör. w_L1 < 1) mi eklensin?
3. L4-semi'nin val etiketiyle erken durdurması kabul mü, yoksa testten önce val etiketsiz erken durdurmayla (ör. sabit epoch) yeniden mi üretilsin?
4. Literatür satırındaki 0,952 (Tech Science) değerinin künyesi repoda yok; doğrulanmalı.
5. (I-turu sonrası) L1-FE/L2-FE/L5-FE val sonuçlarında val erken durdurmasından gelen hafif iyimserlik var (§8.2). Test öncesinde final model, val erken durdurması olmayan L1-I ile mi değiştirilsin, yoksa L5-FE full mü kalsın? (Test sonucu her iki durumda da iyimserlik taşımaz; soru yalnız val raporunun dürüstlüğü ve seçim kuralının uygulanması.)

## 8. I-turu: davranışsal (`other`) odaklı iyileştirme (protokol §20)

Seçim yalnız train binalarında, 3 katlı gruplu CV'nin OOF skorlarıyla, her seed ayrı (ana ölçüt `other` PR-AUC, ikincil genel PR-AUC, gürültü kuralı 2·std/√3). Varyant full. Kaynak: `results/lead/tables/i_cv_steps.csv`, `i_cv_folds.csv`, `i_shap_other.csv`, `val_results.csv` (L1-I, L2-I).

### 8.1 CV adımları (kat ort.; Δ = önceki kabul edilen yapılandırmaya göre)

| seed | hat | adım | other PR-AUC | genel PR-AUC | frozen PR-AUC | gap_edge PR-AUC | other ROC-AUC | genel ROC-AUC | Δ other (2se) | Δ genel (2se) | karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | L1 | temel L1-FE | 0.053 ± 0.041 | 0.712 | 0.862 | 0.490 | 0.816 | 0.940 | — | — | temel |
| 42 | L1 | +P1 (L1-other, max) | 0.054 ± 0.044 | 0.599 | 0.687 | 0.368 | 0.817 | 0.938 | +0.0007 (0.0041) | -0.1134 (0.1547) | ret |
| 42 | L1 | +P4 (akran sapması) | 0.060 ± 0.039 | 0.724 | 0.872 | 0.494 | 0.826 | 0.942 | +0.0063 (0.0109) | +0.0113 (0.0088) | kabul |
| 42 | L1 | +P5 (k=3) | 0.054 ± 0.031 | 0.692 | 0.854 | 0.332 | 0.831 | 0.941 | -0.0054 (0.0096) | -0.0315 (0.0139) | ret |
| 42 | L1 | +P5 (k=6) | 0.049 ± 0.027 | 0.627 | 0.785 | 0.239 | 0.818 | 0.933 | -0.0112 (0.0147) | -0.0969 (0.0260) | ret |
| 42 | L2 | temel L2-FE | 0.105 ± 0.054 | 0.735 | 0.875 | 0.393 | 0.800 | 0.928 | — | — | temel |
| 42 | L2 | +P1 (L2-other, max) | 0.097 ± 0.050 | 0.696 | 0.822 | 0.295 | 0.808 | 0.927 | -0.0087 (0.0074) | -0.0388 (0.0497) | ret |
| 42 | L2 | +P5 (k=3) | 0.069 ± 0.024 | 0.693 | 0.849 | 0.263 | 0.792 | 0.923 | -0.0365 (0.0375) | -0.0419 (0.0070) | ret |
| 42 | L2 | +P5 (k=6) | 0.059 ± 0.017 | 0.630 | 0.784 | 0.185 | 0.777 | 0.913 | -0.0466 (0.0464) | -0.1049 (0.0100) | ret |
| 7 | L1 | temel L1-FE | 0.111 ± 0.036 | 0.772 | 0.953 | 0.601 | 0.812 | 0.940 | — | — | temel |
| 7 | L1 | +P1 (L1-other, max) | 0.115 ± 0.054 | 0.767 | 0.946 | 0.562 | 0.817 | 0.940 | +0.0046 (0.0213) | -0.0059 (0.0014) | ret |
| 7 | L1 | +P4 (akran sapması) | 0.107 ± 0.040 | 0.772 | 0.952 | 0.591 | 0.822 | 0.943 | -0.0037 (0.0105) | -0.0000 (0.0024) | ret |
| 7 | L1 | +P5 (k=3) | 0.083 ± 0.027 | 0.741 | 0.929 | 0.417 | 0.822 | 0.941 | -0.0280 (0.0136) | -0.0318 (0.0061) | ret |
| 7 | L1 | +P5 (k=6) | 0.065 ± 0.018 | 0.675 | 0.866 | 0.284 | 0.813 | 0.935 | -0.0460 (0.0300) | -0.0973 (0.0163) | ret |
| 7 | L2 | temel L2-FE | 0.086 ± 0.034 | 0.709 | 0.887 | 0.296 | 0.811 | 0.928 | — | — | temel |
| 7 | L2 | +P1 (L2-other, max) | 0.097 ± 0.039 | 0.690 | 0.852 | 0.260 | 0.826 | 0.930 | +0.0112 (0.0216) | -0.0190 (0.0189) | ret |
| 7 | L2 | +P5 (k=3) | 0.059 ± 0.025 | 0.658 | 0.848 | 0.211 | 0.802 | 0.922 | -0.0267 (0.0167) | -0.0516 (0.0109) | ret |
| 7 | L2 | +P5 (k=6) | 0.048 ± 0.022 | 0.580 | 0.756 | 0.153 | 0.785 | 0.910 | -0.0379 (0.0216) | -0.1295 (0.0193) | ret |
| 2026 | L1 | temel L1-FE | 0.134 ± 0.088 | 0.779 | 0.943 | 0.500 | 0.815 | 0.944 | — | — | temel |
| 2026 | L1 | +P1 (L1-other, max) | 0.136 ± 0.078 | 0.715 | 0.845 | 0.454 | 0.820 | 0.942 | +0.0023 (0.0165) | -0.0639 (0.1125) | ret |
| 2026 | L1 | +P4 (akran sapması) | 0.130 ± 0.088 | 0.788 | 0.957 | 0.518 | 0.817 | 0.944 | -0.0037 (0.0073) | +0.0092 (0.0190) | ret |
| 2026 | L1 | +P5 (k=3) | 0.110 ± 0.064 | 0.758 | 0.929 | 0.385 | 0.817 | 0.942 | -0.0244 (0.0331) | -0.0212 (0.0058) | ret |
| 2026 | L1 | +P5 (k=6) | 0.078 ± 0.041 | 0.705 | 0.884 | 0.279 | 0.807 | 0.936 | -0.0562 (0.0559) | -0.0738 (0.0129) | ret |
| 2026 | L2 | temel L2-FE | 0.139 ± 0.067 | 0.766 | 0.921 | 0.418 | 0.824 | 0.940 | — | — | temel |
| 2026 | L2 | +P1 (L2-other, max) | 0.140 ± 0.061 | 0.731 | 0.867 | 0.368 | 0.839 | 0.942 | +0.0002 (0.0095) | -0.0347 (0.0584) | ret |
| 2026 | L2 | +P5 (k=3) | 0.089 ± 0.033 | 0.722 | 0.894 | 0.299 | 0.818 | 0.936 | -0.0499 (0.0420) | -0.0439 (0.0032) | ret |
| 2026 | L2 | +P5 (k=6) | 0.064 ± 0.021 | 0.657 | 0.832 | 0.218 | 0.804 | 0.927 | -0.0750 (0.0540) | -0.1088 (0.0067) | ret |

**Kabul edilen adımlar:** seed 42 L1 +P4 (akran sapması). P1 (L1-other / L2-other, kalibre max) hiçbir seed ve hatta kabul edilmedi: `other` PR-AUC değişimi gürültü düzeyinde, genel PR-AUC (özellikle frozen ve gap_edge) belirgin düştü — kalibre max, `other` modelinin alarmlarını veri kalitesi olaylarının önüne taşıyor. P5 (yumuşatma) her yerde ret: anomalilerin çoğu kısa, ortalama alma tepeleri bastırıyor. P2 (sıcaklık) yapılmadı (§8.4).

### 8.2 Son val (bir kez; son modellerde val erken durdurması YOK)

| model | other PR-AUC | other ROC-AUC | genel PR-AUC | genel ROC-AUC | frozen PR-AUC | frozen ROC-AUC | gap_edge PR-AUC | gap_edge ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| L1-FE | 0.053 ± 0.030 | 0.770 ± 0.074 | 0.667 ± 0.099 | 0.912 ± 0.033 | 0.882 ± 0.054 | 0.990 ± 0.005 | 0.505 ± 0.089 | 0.945 ± 0.012 |
| L1-I | 0.049 ± 0.030 | 0.770 ± 0.062 | 0.640 ± 0.123 | 0.911 ± 0.030 | 0.856 ± 0.082 | 0.988 ± 0.007 | 0.426 ± 0.202 | 0.937 ± 0.017 |
| L2-FE | 0.045 ± 0.019 | 0.772 ± 0.090 | 0.594 ± 0.086 | 0.903 ± 0.037 | 0.792 ± 0.018 | 0.980 ± 0.007 | 0.183 ± 0.051 | 0.895 ± 0.006 |
| L2-I | 0.039 ± 0.020 | 0.731 ± 0.108 | 0.546 ± 0.100 | 0.892 ± 0.044 | 0.721 ± 0.043 | 0.977 ± 0.007 | 0.134 ± 0.044 | 0.920 ± 0.011 |

- L1-I − L1-FE, other PR-AUC: -0.0042 (2se 0.0014).
- L1-I − L1-FE, genel PR-AUC: -0.0264 (2se 0.0378).
- L2-I − L2-FE, other PR-AUC: -0.0058 (2se 0.0046).
- L2-I − L2-FE, genel PR-AUC: -0.0475 (2se 0.0176).

**Yorum:** seed 7 ve 2026'da L1-I'nin yapılandırması L1-FE ile aynı (bütün adımlar ret); L2-I üç seed'de de L2-FE ile aynı. Aradaki tek fark son modelin val'de erken durdurma yapmaması (ağaç/epoch = CV ortalaması). Buna rağmen val'de L1-I/L2-I biraz daha düşük: **L1-FE ve L2-FE'nin val sonuçlarında val erken durdurmasından gelen hafif iyimserlik var** (protokol §9 buna izin veriyordu; ama val hem erken durdurmada hem ölçümde kullanıldığı için). Test için önemli: test sonucu bu iyimserliği taşımaz; L1-I/L2-I val sonuçları test için daha dürüst bir beklentidir.

### 8.3 Davranışsal dilimde en önemli 10 öznitelik (SHAP, tutulan katlar; L1 hattının seçilen modeli)

| sıra | seed 42 | seed 7 | seed 2026 |
|---|---|---|---|
| 1 | profile_z | l4semi_score | l4semi_score |
| 2 | log_diff_168h | profile_z | profile_z |
| 3 | log_diff_1h | gap_age_h | gap_age_h |
| 4 | gap_age_h | log_diff_1h | log_diff_168h |
| 5 | log_diff_24h | log_diff_168h | roll_iqr_24h |
| 6 | backward_run_length | primary_use | dow_sin |
| 7 | last_gap_len | last_gap_len | n_distinct_24h |
| 8 | lag_1h | log_diff_24h | absdiff_1h |
| 9 | month_cos | profile_mean | log_diff_1h |
| 10 | profile_mean | backward_run_length | log_diff_24h |

Üç seed'de de ilk sıralarda `profile_z` (aynı hafta-saatine göre sapma), `l4semi_score` (G7, tahmin artığı), `log_diff_168h`, `gap_age_h`: davranışsal sinyal profil sapması ve tahmin artığından geliyor; `gap_age_h` (son boşluktan beri geçen süre) `other` saatlerinin de boşluklara yakın olduğunu gösteriyor.

### 8.4 P2 (sıcaklık) — kullanılmadı

`train_features.csv`'de `air_temperature` var, ama **üretimi belgelenemedi**: sütunda hiç eksik yok (%0 NaN; ASHRAE ham hava verisinde eksik saatler bulunur → Kaggle boşlukları bilinmeyen bir yöntemle doldurmuş; geleceğe bakan doldurma dışlanamıyor). `air_temperature_mean_lag7` geri 7 saat, merkezli 7 saat ya da t−7…t−1 pencerelerinin hiçbirine uymuyor. §20.2 kuralı gereği kullanılmadı.

### 8.5 I-turu sonrası seçim (§18.2)

Seçim değişmedi: **L5-FE / full** (`val_selection_v2.csv`). L1-I ve L2-I yarışa girdi. İkisi de `other`'da gürültü düzeyinde (C1). Genel PR-AUC'de L1-I gürültü düzeyinde (C2'de), L2-I anlamlı biçimde geride (C2 dışı). C2'de sinir ağı içeren tek model L5-FE full olduğu için seçim değişmedi. **Çekince:** L5-FE'nin üyeleri (L1-FE, L2-FE) val'de erken durdurulmuş ve ağırlığı val'de seçilmiştir; §8.2'deki iyimserlik L5-FE için de geçerlidir. Val erken durdurması olmayan karşılığı L1-I'dir.
