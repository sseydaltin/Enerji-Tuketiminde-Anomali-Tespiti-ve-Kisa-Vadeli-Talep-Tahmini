# Literatür Taraması — UCI "Individual Household Electric Power Consumption" (IHEPC) Veri Seti

**YMÜ425 – Uygulamalı Sinir Ağları** · Dilara YILDIRIM · Şeyda ALTIN
**Tarama tarihi:** 2026-11-09 · **Durum:** taslak. Her iki üye okuyup "doğrulanmalı" işaretli maddeleri kontrol etmeli.

> **Okuma kuralları**
> - Başkalarının sayıları makaleden alındığı gibi aktarılır. Her satırda bilginin **nereden** geldiği yazılıdır:
>   **TM** = tam metinden okundu, **Ö** = yalnız özetten, **İ** = ikincil (başka bir makalenin karşılaştırma tablosundan aktarıldı).
> - Bizim sayılarımız yalnız `results/tables/final_forecast_val_vs_test.csv` ve `results/tables/final_anomaly_val_vs_test.csv` dosyalarından alınmıştır.
> - Literatür sayıları farklı çözünürlük, ufuk, bölme ve ölçekte hesaplandığı için **bizim sonuçlarımızla doğrudan kıyaslanamaz**. Bu belgenin ana konusu da bu farklardır (§4).
> - Çalışma bazında ayrıntılı matris `docs/literatur.md`'de, taramanın ham çıktısı `docs/literatur_katalog.csv`'dedir.

---

## 0. Özet: hocaya söylenecek beş cümle

1. Bu veri seti çok kullanılıyor. OpenAlex'te **660** yayın tam metninde veri setinin adını geçiriyor. **141** yayın başlığında ya da özetinde anıyor, **126** yayın veri setinin DOI'sine atıf yapıyor. Yayınların çoğu 2023 sonrasına ait (§1–2).
2. En çok atıf alan çalışmaların büyük kısmı CNN+LSTM/GRU **hibrit** modelleri öneriyor ve çok yüksek başarı bildiriyor (ör. R² = 0,98). Bu sonuçların hemen hepsi bizimkinden **daha kolay bir görevde** elde edilmiş: çoğunlukla 1 adım (1 saat) ileri tahmin, normalize edilmiş ölçekte hata, k-katlı çapraz doğrulama ve naif referans yöntem yokluğu (§4).
3. Aynı veri setinde **bizimkine en yakın protokolü** (gün öncesi tahmin, kronolojik bölme, son yıl test) kullanan Gasparin vd. [14], 11 farklı derin modelde test R² değerini **0,00 ile 0,28** arasında buluyor. Bizim final modelimiz F13-ens'in test R² değeri **0,297**. Çözünürlük ve test dönemi farklı olduğu için bu bir "daha iyiyiz" iddiası değildir. Ama aynı zorlukta bir görevde benzer bir düzeyde olduğumuzu gösterir (§5.1).
4. Tek hanede tahmin hatasının yüksek olması literatürde bilinen bir olgu. Hata, toplulaştırma (aggregation) ile azalıyor [37]–[39], [14]. Bizim hane–şebeke karşılaştırmamız (hane R² 0,297, şebeke R² 0,974) bununla tutarlı (§5.2).
5. Bu veri setinde anomali tespiti çalışması **az** (ayrıntılı incelediğimiz 4 çalışma var) ve etiket gerektirenlerin hepsi **sentetik etiket** kullanıyor. Örneğin her gün tek bir noktanın değerini ikiye katlıyorlar. Eşiği ayrı bir doğrulama kümesi olmadan seçiyorlar ve nokta bazlı metrik bildiriyorlar (F1 0,72–0,99). Bizim protokolümüz daha zor: 4 anomali tipi, σ tabanlı büyüklük, olay F1, val-dev/val-hold/test ayrımı. Bu yüzden sayılar doğrudan kıyaslanamaz. Ayrıca **H3 hedefimiz (F1 ≥ 0,70) karşılanmadı**; literatür bu sonucu değiştirmez, yalnızca bağlamını açıklar (§6).

---

## 1. Tarama yöntemi

### 1.1 Veri tabanları ve sorgular

Arama 2026-11-09'da yapıldı. Yalnız resmî ve açık API'ler kullanıldı (yayıncı sitelerinin bot korumaları aşılmadı).

| Kaynak | Sorgu | Bulunan kayıt |
|---|---|---|
| OpenAlex – başlık+özet | `"household electric power consumption"` | 119 |
| OpenAlex – başlık+özet | `IHEPC` | 39 |
| OpenAlex – tam metin | `"individual household electric power consumption"` | 660 |
| OpenAlex – atıf | Veri seti DOI'sine (10.24432/C58K54) atıf yapanlar | 126 |
| **Birleşim (tekrarsız)** | | **740** |
| Europe PMC (açık erişim) | `"individual household electric power consumption" OR "household electric power consumption dataset" OR IHEPC` | 55 |
| Semantic Scholar, arXiv/ar5iv, yayıncı sayfaları | Tek tek makale doğrulaması | — |

### 1.2 Eleme ve seçim

- **Katalog (`docs/literatur_katalog.csv`, 271 satır):** Başlık/özet eşleşmeleri, veri seti DOI'sine atıflar ve ayrıntılı incelediğimiz tam metin eşleşmelerinden oluşur. **Ham arama çıktısıdır, elle elenmemiştir.** Örneğin 2012 öncesine ait birkaç kayıt yalnız kelime benzerliğiyle eşleşmiştir ve veri setiyle ilgisizdir.
- **Ayrıntılı inceleme (≈ 30 çalışma):** Şu ölçütlerden en az birini sağlayanlar seçildi:
  - (a) en çok atıf alan IHEPC tahmin çalışmaları,
  - (b) bize yöntem olarak en yakın olanlar (çok adımlı tahmin, kronolojik bölme),
  - (c) bu veri setinde anomali/aykırı değer tespiti yapan tüm çalışmalar,
  - (d) değerlendirme yöntemini eleştiren temel makaleler (bunlar başka veri setleri de kullanabilir).
- **Dışlananlar:** Veri setini yalnız kaynakçada ya da "ileride kullanılabilir" diye anan çalışmalar. Örnekler: HousEEC (Pecan Street verisi) ve Le vd. 2020 (Kore binaları).

### 1.3 Sınırlılıklar

- OpenAlex'in tam metin dizini eksik; bazı ücretli makaleler (ör. Kim ve Cho 2019 [5]) tam metin sorgusunda çıkmadı. Bunlar başlık/özet sorgusu ve atıflar üzerinden bulundu.
- Tam metnine yasal yoldan ulaşamadığımız makalelerde (MDPI, IEEE, Elsevier sayfaları) yalnız özet okundu. Sayılar ise ikincil tablolardan aktarıldı (**İ**). Bu satırlar asıl makaleden kontrol edilmeli.
- 660 tam metin eşleşmesinin hepsi okunmadı. Bu bir "sistematik eşleme + seçilmiş çekirdek inceleme"dir, tam bir PRISMA sistematik derlemesi değildir.

---

## 2. Veri setinin literatürdeki genel görünümü

**Yıllara göre dağılım** (başlık/özet eşleşmesi olan 141 kayıt; 2012 öncesindeki 3 kayıt ilgisiz eşleşmedir):

| Dönem | Kayıt sayısı |
|---|---|
| ≤ 2019 | 19 |
| 2020–2022 | 29 |
| 2023–2026 | 93 (yalnız 2026'da 45) |

Bu dağılımdan çıkan gözlemler:

- **Baskın yöntem ailesi CNN + LSTM/GRU hibritleri.** Kim ve Cho'nun CNN-LSTM'i [5] ile başlayan çizgi, Bi-LSTM [7], [8], LSTM-AE [9], GRU [10], [11], [16], [17], dikkat mekanizması [12], [15] ve son yıllarda Transformer/PatchTST [30], [31] ile sürmüş.
- **Yazar yoğunlaşması.** En çok atıf alan IHEPC tahmin makalelerinin önemli bir kısmında aynı araştırma grubu ortak yazar: S. W. Baik ve ekibi [7]–[11], [16], [17], [22]. Bu makaleler birbirlerinin sonuçlarını karşılaştırma tablolarında taşıyor (§4). Bu tek başına bir kusur değildir, ancak protokollerin bağımsız olarak yeniden üretilmediği anlamına gelir.
- **Görev dağılımı çok dengesiz.** Tahmin çalışmaları baskın. Anomali tespiti ise yalnız birkaç çalışmada var [33]–[36]. Bunlara ek olarak, veri setini veri akışı (streaming) ve kümeleme deneylerinde kıyas verisi olarak kullanan makaleler de mevcut.

---

## 3. Tahmin çalışmaları — özet tablo

Ayrıntılı matris (güçlü/zayıf yön, bize katkısı) için: `docs/literatur.md`.

| # | Çalışma | Model | Çözünürlük / ufuk | Bölme | Bildirilen sonuç (makaledeki ölçekte) | Düzey |
|---|---|---|---|---|---|---|
| [3] | Marino vd. 2016 | LSTM, LSTM-S2S | 1 dk ve 1 sa | doğrulanmalı | Standart LSTM 1 dk'da başarısız, 1 sa'te iyi; S2S ikisinde de iyi | Ö |
| [4] | Mocanu vd. 2016 | FCRBM, CRBM | doğrulanmalı | doğrulanmalı | Saatlik RMSE 0,666, R² −0,0925 | İ ([21]); IHEPC kullanımı **doğrulanmalı** |
| [5] | Kim & Cho 2019 | CNN-LSTM | saatlik ve günlük (İ); diğer çözünürlükler doğrulanmalı | doğrulanmalı | Saatlik RMSE 0,595, MAE 0,3317, MAPE %32,83. Doğrusal regresyon: RMSE 0,6570, MAPE %83,74 | Ö + İ ([21]) |
| [6] | Kim & Cho 2019b | Durum açıklanabilir AE | 60 dk geçmiş → 15/30/45/60 dk | doğrulanmalı | MSE 0,384 | Ö |
| [7] | Le vd. 2019 | CNN + Bi-LSTM (EECP-CBL) | gerçek zamanlıdan uzun vadeye | doğrulanmalı | "Literatürdeki yöntemlerden iyi" | Ö |
| [8] | Ullah vd. 2020 | CNN + çok katmanlı Bi-LSTM | saatlik | **10 katlı CV** + hold-out | Saatlik RMSE 0,565, MAE 0,346, MAPE %29,10 | Ö + İ ([21]) |
| [9] | Khan vd. 2020 | CNN + LSTM-AE | saatlik, günlük | %75/%25 (ilk 3 yıl eğitim, son yıl test) | Saatlik MSE 0,19, MAE 0,31, RMSE 0,47 | **TM** |
| [10] | Sajjad vd. 2020 | CNN-GRU | saatlik | doğrulanmalı | RMSE 0,47, MAE 0,33 | Ö + İ ([21]) |
| [12] | Bu & Cho 2020 | Çok başlıklı dikkat + CRNN | doğrulanmalı | doğrulanmalı | Önceki en iyi modele göre hata %31,01 daha az | Ö |
| [14] | **Gasparin vd. 2022** | FNN, LSTM, GRU, S2S, TCN (11 varyant) | **15 dk, 1 gün ileri (96 adım)** | **Kronolojik: son yıl test** | En iyi: GRU (MIMO) RMSE 0,75, MAE 0,52, **R² 0,279** | **TM** |
| [15] | Özcan vd. 2021 | Çift aşamalı dikkat RNN | 1 dk … 1 hafta | **10 katlı CV** | 1 sa: MAE 240,14, RMSE 300,18 (birim belirtilmemiş) | **TM** |
| [20] | Abdel-Basset vd. 2022 | GRU + TCN (STLF-Net) | 1 saat ileri | doğrulanmalı | RMSE 0,4386, MAPE %36,24 | Ö + İ ([21]) |
| [21] | Alsharekh vd. 2022 | Residual CNN + ML-LSTM | saatlik, günlük | tam metinde bulunamadı | Saatlik RMSE 0,0325, MAE 0,0144, **R² 0,9841**, MAPE %1,024 | **TM** |
| [25] | da Silva & Meneses 2023 | LSTM vs Bi-LSTM (veri setlerinden biri Fransa'daki hane; tam metinde IHEPC adı geçiyor) | doğrulanmalı | **Zaman serisi CV** | Bi-LSTM, LSTM'den istatistiksel olarak anlamlı biçimde iyi (Friedman p = 0,0455; 4 veri seti üzerinde) | Ö |
| [29] | Alden vd. 2020 | LSTM | 1/6/24 sa, 1/7 gün | doğrulanmalı | Tek tek 24/6/1 saatlik ufuklarda "tatmin edici değil" | Ö; UCI kullanımı **doğrulanmalı** |
| [30] | Qarni vd. 2026 | MCV-PatchTST | saatlik, günlük, haftalık | doğrulanmalı | PatchTST'ye göre MAE %10–17 daha düşük | Ö |
| [35] | Mathonsi & van Zyl 2025 | LSTM, ESN, CCN, SARIMAX | **28 gün → 7 gün** | **Kronolojik: ilk 3 yıl eğitim, son yıl test; walk-forward ayar** | RMSE: Persistence 465, SARIMAX 377,86, LSTM 337,95 (birim belirtilmemiş) | **TM** (ar5iv) |

**Bizim F13-ens modelimiz** (kaynak `final_forecast_val_vs_test.csv`): saatlik çözünürlük, 168 sa → 24 sa, kronolojik bölme, test (2010-04-01 – 2010-11-26) bir kez kullanıldı. Sonuçlar **kW** cinsinden: MAE 0,406 ± 0,005, RMSE 0,598, R² 0,297. SN168'e göre MAE kazancı %26,0.

---

## 4. Literatürdeki sayılar neden birbirleriyle (ve bizimle) karşılaştırılamaz?

Bu bölüm, hocanın "skorunuz düşük" değerlendirmesine verilecek cevabın temelidir.

### 4.1 Aynı veri seti, aynı tablo, 10–80 kat farklı hatalar

Alsharekh vd. [21] "saatlik IHEPC" başlığı altında önceki çalışmaların sonuçlarını tek tabloda topluyor. Aşağıdaki değerler o tablodan **aynen** aktarılmıştır (**İ**):

| Yöntem (kaynak) | RMSE | MAE | MSE | MAPE (%) |
|---|---|---|---|---|
| Doğrusal regresyon [5] | 0,6570 | 0,5022 | 0,4247 | 83,74 |
| ANN [27] | 1,15 | 1,08 | – | – |
| CNN-LSTM [5] | 0,595 | 0,3317 | 0,3549 | 32,83 |
| CNN-BDLSTM [8] | 0,565 | 0,346 | 0,319 | 29,10 |
| CNN-LSTM-AE [9] | 0,47 | 0,31 | 0,19 | – |
| SE-AE [6] | – | 0,395 | 0,384 | – |
| GRU [17] | 0,41 | 0,19 | 0,17 | 34,48 |
| FCRBM [4] | 0,666 | – | – | – (R² −0,0925) |
| CNN-BiGRU [11] | 0,42 | 0,29 | 0,18 | – |
| CNN-GRU [10] | 0,47 | 0,33 | 0,22 | – |
| STLF-Net [20] | 0,4386 | 0,2674 | 0,1924 | 36,24 |
| Residual GRU [16] | 0,4186 | 0,2635 | 0,1753 | – |
| Residual CNN stacked LSTM [22] | 0,058 | 0,003 | 0,038 | – |
| **R-CNN + ML-LSTM (önerilen) [21]** | **0,0325** | **0,0144** | **0,0011** | **1,024** (R² 0,9841) |

Aynı "saatlik IHEPC" etiketi altında RMSE değerleri 0,03 ile 1,15 arasında değişiyor. Bu tablo dört şeyi gösteriyor:

- **Ölçek belirsiz.** 0,4–0,6 civarındaki değerler kW ölçeğiyle uyumlu. 0,03–0,06 civarındakiler ise büyük olasılıkla min-max ile normalize edilmiş seride hesaplanmış. Alsharekh vd. ön işlemede min-max ölçekleyici kullandığını açıkça yazıyor [21]. Ancak hatanın hangi ölçekte raporlandığını açıkça belirtmiyor (doğrulanmalı). İki farklı ölçekteki sayılar aynı tabloda yan yana sıralanıyor.
- **İç tutarsızlıklar var.**
  - [22] için tabloda verilen değerlerde RMSE² ≠ MSE (0,058² ≈ 0,0034, ama MSE 0,038 yazılmış).
  - Alsharekh vd.'nin kendi metninde saatlik sonuç için 0,0447/0,0132/0,002/0,9759 yazıyor; tabloda ise 0,0325/0,0144/0,0011/0,9841 var.
  - Zhang vd. [33] tahmin RMSE'sini metinde 0,73 (LSTM 0,86), tabloda 0,74 (LSTM 0,91) olarak veriyor.
- **Farklı veri setlerinin sonuçları karşılaştırılıyor.** Özcan vd. [15], IHEPC sonuçlarını "önceki çalışmalarla karşılaştırma" tablosunda başka veri setlerinde (ENTSO-E İtalya, RTE Fransa, AEP) elde edilmiş sonuçlarla yan yana koyuyor. Bu bilgi makalenin kendi ilgili çalışmalar tablosundan alınmıştır (TM).
- **Birim yok.** Özcan vd. MAE = 240,14 bildiriyor [15], Mathonsi ve van Zyl RMSE = 337,95 [35]. Her iki makalede de birim belirtilmemiş.

### 4.2 Ufuk: 1 adım ileri tahmin ≠ 24 saat ileri tahmin

Çalışmaların çoğu **bir sonraki adımı** tahmin ediyor: STLF-Net 1 saat ileri [20], Kim ve Cho 15–60 dk ileri [6]. 1 saat ileri tahminde son gözlem çok güçlü bir ipucudur. Bizim görevimiz ise geçmiş 168 saatten **sonraki 24 saati birden** tahmin etmek. Ufuk uzadıkça hata artar; bunu kendi ufuk bazlı hata grafiğimiz de gösteriyor.

Literatür de bunu destekliyor:
- Yan vd. [13] mevcut çalışmaların çoğunun yalnız tek adımlı tahmin yaptığını ve bunun pratik kullanım için çok kısa olduğunu açıkça eleştiriyor (Ö).
- Alden vd. [29], LSTM'nin tek hanede 24/6/1 saatlik ufuklarda tatmin edici doğruluk vermediğini bildiriyor (Ö; UCI kullanımı doğrulanmalı).

### 4.3 Bölme yöntemi: k-katlı çapraz doğrulama ve doğrulama kümesi yokluğu

- Ullah vd. [8] ve Özcan vd. [15] **10 katlı çapraz doğrulama** kullanıyor. Zaman serisinde rastgele katlar, test pencerelerinin eğitim pencereleriyle komşu, hatta örtüşen olmasına yol açar.
- Bergmeir ve Benítez [42] ile Bergmeir, Hyndman ve Koo [43] bu konuyu inceliyor. Saf otoregresif ve hataları ilişkisiz modellerde k-katlı CV'nin geçerli olabileceğini gösteriyorlar. Ancak dışsal öznitelikler, normalizasyonun tüm veriye uygulanması gibi durumlarda bu varsayım kolayca bozulur. **Bu nüans önemli:** "k-katlı CV her zaman sızıntıdır" demek yanlış olur. Ama gerçek kullanımı (geleceği tahmin etmeyi) temsil etmediği açıktır.
- Hewamalage vd. [41], ML/DL yayınlarında yaygın değerlendirme hatalarını sıralıyor: uygunsuz bölme, normalize ölçekte metrik, naif referans eksikliği, tek bir test dönemi. Bu hataların yöntemleri gerçekte olduğundan daha rekabetçi gösterdiğini belirtiyor.
- Çoğu makalede **ayrı bir doğrulama kümesi** veya "test bir kez kullanıldı" ifadesi yok.
- Bizde durum şöyle:
  - kronolojik train/val/test,
  - early stopping ve model seçimi yalnız val ile,
  - Optuna yalnız train içi zamansal CV ile,
  - test bir kez kullanıldı.

### 4.4 Naif referans yöntem (persistence / seasonal naive) çoğunlukla yok

İncelediğimiz yüksek atıflı tahmin makalelerinin çoğu yalnız başka derin modellerle karşılaştırma yapıyor.

- Naif referans **bildiren** çalışmalar:
  - Mathonsi ve van Zyl [35] persistence modelini raporluyor. LSTM, persistence'a göre RMSE'yi %27,3 düşürüyor (465 → 337,95; hesap bizim).
  - Yan vd. [13] persistence modelini karşılaştırmaya dahil ediyor (Ö).
- Naif referans **bildirmeyen** çalışmalar:
  - Gasparin vd. [14] temel referans olarak yalnızca derin FNN kullanıyor; istatistiksel ya da naif bir referans bildirmiyor.
  - Alsharekh vd. [21] persistence modellerini "birkaç saat ileri tahminde başarısız" diye ilgili çalışmalar kısmında eliyor ve deneylere almıyor.

Naif referans olmadan "R² = 0,98" gibi bir sayının ne kadar zor kazanıldığı anlaşılamaz. Biz her modeli SN168'e göre yüzde kazançla raporluyoruz. Bu, Hewamalage vd.'nin [41] önerdiği pratikle uyumlu.

### 4.5 Ön işleme kararları sonucu şişirebilir

- **Aykırı değerleri silmek.** Alsharekh vd. 3-sigma kuralıyla aykırı değerleri siliyor [21]. Khan vd. "null, tekrarlı ve aykırı değerleri" çıkarıyor [9]. Test kümesinden aykırı değer silmek, tahmini en zor saatleri değerlendirmeden atmak demektir.
  Aykırı değer silme bizim anomali göreviyle de çelişir: anomali tespit etmek istediğimiz şey tam da bu aykırılıklardır. Biz hiçbir değeri silmiyoruz. Yalnızca doldurulmuş (`is_imputed=1`) hedefleri metrikten çıkarıyoruz.
- **Gelecekteki veriyle doldurma.** Gasparin vd. eksik değerleri "farklı yıllarda aynı zaman dilimindeki ortalama" ile dolduruyor [14]. Bu ortalama, sonraki yılların verisini de kullanır (küçük bir gelecek bilgisi). Biz ≤3 saatlik boşluklarda iki yanı kullanan interpolasyon, uzun boşluklarda bir hafta önceki değeri kullanıyoruz. Küme sınırında ise yalnız hafta-önce yöntemi uygulanıyor (README §4.3).
- **Doldurma yöntemleri.** Hanna vd. [32] aynı veri setinde 30 doldurma modelini karşılaştırıyor. Ağaç tabanlı topluluk yöntemlerinin derin ağlarla aynı düzeyde ya da daha iyi olduğunu buluyor (Extra Trees MAE 0,0474 kW). Bizim basit doldurma tercihimiz için bu bir alternatif ve bir tartışma noktası.

---

## 5. Bize en yakın protokoller ve sonuçlarımızın konumu

### 5.1 Gasparin, Luković ve Alippi (2022) — aynı veri, gün öncesi, kronolojik test [14]

Bulduğumuz çalışmalar arasında **protokolü bize en yakın olan** çalışma bu (TM, ar5iv). Kurulumları:

- 15 dakikalık çözünürlük, hedef `Global_active_power`, dışsal öznitelik yok.
- 4 günlük girdi → 1 günlük (96 adım) çıktı.
- **Test: son yıl (kronolojik)**. Eğitim/doğrulama: kalan veriden her beş aydan biri doğrulamaya ayrılmış.
- Her model 10 kez çalıştırılmış; ortalama ± std raporlanmış.

| Model (Gasparin vd. Tablo 5) | RMSE | MAE | R² |
|---|---|---|---|
| GRU (MIMO) | 0,75 ± 0,00 | 0,52 ± 0,00 | 0,279 ± 0,004 |
| LSTM (MIMO) | 0,75 ± 0,00 | 0,53 ± 0,00 | 0,276 ± 0,006 |
| DFNN | 0,75 ± 0,01 | 0,53 ± 0,01 | 0,269 ± 0,007 |
| seq2seq (SG) | 0,76 ± 0,01 | 0,53 ± 0,01 | 0,253 ± 0,03 |
| GRU (özyinelemeli) | 0,89 ± 0,02 | 0,70 ± 0,02 | 0,00 ± 0,04 |

| Bizim modeller (test, saatlik, 24 sa) | MAE (kW) | RMSE (kW) | R² |
|---|---|---|---|
| F5-opt LSTM (3 seed) | 0,413 ± 0,005 | 0,581 | 0,336 |
| F12-L336 XGBoost | 0,416 | 0,568 | 0,364 |
| F13-ens (final, 3 seed) | 0,406 ± 0,005 | 0,598 | 0,297 |

**Bu kıyaslama nasıl okunmalı?**

- **Farklar:**
  - Onlar 15 dakikalık, biz saatlik çözünürlük kullanıyoruz. Saatlik ortalama gürültüyü azaltır, bu yüzden MAE/RMSE'yi doğrudan kıyaslamak bizim lehimize yanlı olur.
  - Test dönemleri farklı: onlarınki son 1 yıl, bizimki Nisan–Kasım 2010.
  - RMSE/MAE birimi makalede açıkça yazılmamış (hedef kW olduğu için büyük olasılıkla kW).
- **Söylenebilecek olan:** Aynı veri setinde, gün öncesi ve kronolojik test protokolünde iyi kurulmuş derin modeller R² ≈ 0,25–0,28 civarında kalıyor. Bizim test R² değerlerimiz (Seq2Seq 0,263 ile XGBoost 0,364 arası) aynı düzeyde. "R² = 0,98" gibi değerler ise farklı (daha kolay) bir görev tanımından geliyor.
- **Ortak bulgu 1:** Gasparin vd. de doğrudan çok çıktılı (MIMO) stratejinin, özyinelemeli (recursive) stratejiden iyi olduğunu buluyor. Bizim tasarım kararımız da aynı yönde: 24 çıktılı doğrudan tahmin.
- **Ortak bulgu 2:** Onlarda da modellerin çoğu birbirine çok yakın. Bizde de F5, F6, F10, F11 ve F12'nin test MAE'leri 0,41–0,43 aralığında.

### 5.2 Hane mi, şebeke mi? — toplulaştırma literatürü

- Gasparin vd. [14] tek sayaç verisinin "çok daha oynak ve müşteri davranışına duyarlı" olduğunu söylüyor. Toplulaştırılmış GEFCom2014 verisinde ise çok daha yüksek doğruluk elde ediyorlar.
- Humeau vd. [37] hane düzeyindeki tüketimin iletim/dağıtım düzeyine göre çok daha düzensiz olduğunu ve mahalle düzeyinde (yüzlerce hane) tahminin farklı yöntemler gerektirdiğini gösteriyor.
- Sevlian ve Rajagopal [38], 180 binden fazla müşteri üzerinde "toplulaştırma arttıkça göreli tahmin hatası belli bir noktaya kadar azalır" şeklinde bir **ölçekleme yasası** öneriyor.
- Haben vd. [39] alçak gerilim (hane, trafo) düzeyindeki tahmin literatürünü derliyor.
- **Bizim bulgumuz bunlarla tutarlı:** Aynı ayarlarla F13-ens, hanede MAPE* %49,4 / R² 0,297; 160 müşterinin toplamında (ELD) MAPE* %3,24 / R² 0,974 (`results/eld/tables/hane_vs_sebeke_forecast.csv`, README).

### 5.3 ELD veri setinin literatürdeki yeri (kısa)

UCI ElectricityLoadDiagrams20112014 [2], LSTNet [55] ile "Electricity" adlı kıyas kümesine dönüştürülmüş: 2012–2014 arasında aktif ~320–321 müşteri, saatlik veri. Bu kıyas kümesi uzun ufuklu tahmin çalışmalarında yaygın olarak kullanılıyor (ör. [47], [56], [57]; kullanılan sürümün ayrıntısı doğrulanmalı).

Bizim kullanımımız farklı: 160 müşterinin **toplamı** üzerinden şebeke düzeyinde tek bir seri. Bu yüzden bu benchmark sonuçlarıyla sayısal kıyas yapılmamalı.

---

## 6. Bu veri setinde anomali tespiti

### 6.1 Bulunan çalışmalar

| # | Çalışma | Yöntem | Etiket nasıl elde edilmiş? | Eşik | Değerlendirme | Bildirilen sonuç | Düzey |
|---|---|---|---|---|---|---|---|
| [33] | Zhang vd. 2021 | Transformer tahmini + K-means; tahmin hatasına eşik | **Sentetik:** her gün rastgele bir değer ×2 (4.800 test saatinde 200 aykırı nokta) | 0,45, "çok sayıda deneme ve ayar sonrası". Ayrı doğrulama kümesi belirtilmemiş | Nokta bazlı Acc/P/R/F1. 1.240 gün eğitim / 200 gün test (kronolojik olup olmadığı belirtilmemiş) | F1 0,72 (P 0,80, R 0,66); LSTM F1 0,66; K-means F1 0,42 | TM |
| [34] | Kea vd. 2023 | Otokodlayıcı + federe öğrenme (FLAE) | **Sentetik:** test değerleri ikiye katlanarak; [33]'e benzer yöntem | POT (uç değer teorisi) | Nokta bazlı. Global model **tüm veri kümesinde (36.000 kayıt)**, yani eğitim verisini de içerecek şekilde değerlendirilmiş | F1 0,9950 (AE 0,9945, USAD 0,9947) | TM |
| [35] | Mathonsi & van Zyl 2025 | Tahmin aralığı (LSTM/ESN/CCN) dışına çıkma | **Sentetik**, toplu (collective) anomali, kirlilik oranı ≈ %1,42. Enjeksiyon ayrıntısı verilmemiş | Tahmin aralığı (α = 0,1/0,05/0,01) | Kronolojik (son yıl test) | F1 değerleri yalnız şekilde verilmiş, sayısal değer yok | TM (ar5iv) |
| [36] | Amalou vd. 2024 | FISVDD (çekirdek tabanlı, artımlı) | Aykırı değer tespiti, tahmin öncesi temizlik amacıyla | – | GRU tahmin MSE'sine etkisi üzerinden ölçülmüş | Uygun çekirdekle en iyi MSE | Ö |

### 6.2 Bu çalışmalardan çıkan dersler

1. **Gerçek etiket yok, herkes sentetik enjeksiyon yapıyor.** Bizim yaklaşımımızla aynı; bu bir sınırlılık olarak kabul ediliyor. Ama enjeksiyonların çoğu çok kolay türden. Tek bir saatin ×2 yapılması, bizim "spike" tipimizin basit bir hâlidir. Seviye kayması, gece desen anomalisi ve kesinti (drop) gibi zor tipler literatürde yok.
2. **Eşik seçimi çoğunlukla test üzerinde ya da belirsiz.** Zhang vd. eşiği "deneme-yanılma ile" seçiyor ve ayrı bir doğrulama kümesinden söz etmiyor [33]. Kea vd. değerlendirmeyi eğitim verisini de içeren tüm kümede yapıyor [34]. Bizde eşik yalnız val-dev'de seçildi. Val-hold bağımsız ölçüm için ayrıldı, test zaman olarak ayrı tutuldu.
3. **Nokta metrikleri ve accuracy yanıltıcı.** Kea vd.'de anomali oranı düşük olduğu için accuracy 0,99 düzeyinde. Bu dengesiz sınıf yapısında bilgi vermez.
4. **Değerlendirme yöntemleri eleştiriliyor.**
   - Kim vd. [45], *point-adjust* yöntemiyle hesaplanan F1'in rastgele skorlara bile yüksek değer verebildiğini gösteriyor. Bu yüzden biz point-adjust uygulamadık; olay F1'i ayrı bir metrik olarak raporladık.
   - Wu ve Keogh [46], yaygın anomali kıyas kümelerindeki birçok anomalinin "önemsiz" olduğunu, yani tek satırlık basit bir kuralla bulunabildiğini ve bunun "ilerleme yanılsaması" yarattığını savunuyor. Tek noktayı ×2 yapan enjeksiyonlar da bu türden.
5. **Bizim sonuçlarımızın konumu** (`final_anomaly_val_vs_test.csv`, test, ana senaryo, 5 seed):
   - A3-s2s-norm: olay F1 0,384 ± 0,170; nokta F1 0,147; PR-AUC 0,241; ROC-AUC 0,864.
   - A0 z-skoru: olay F1 0,398 ± 0,064; ROC-AUC 0,812.

   Literatürdeki F1 0,72–0,99 değerleri daha kolay anomali tiplerinde, nokta bazında ve çoğu zaman eşik testte ayarlanarak elde edildi. Bu yüzden sayılar karşılaştırılamaz. Ancak dürüst olmak gerekirse: **final yöntemimiz testte basit z-skorundan ayırt edilemedi ve H3 hedefi karşılanmadı.** Literatür bu sonucu açıklamaya yardım eder, ama başarılı göstermez.

---

## 7. Literatürdeki boşluklar ve projemizin özgün değeri

Literatür taramasından çıkan ve projemizin doldurduğu boşluklar (öneri formu "Özgün Değer" bölümü için):

| Literatürdeki boşluk | Kanıt | Projemizde ne yapıldı |
|---|---|---|
| Çoğu çalışma 1 adım ileri tahmin yapıyor | [6], [20], [13] eleştirisi | 168 sa → 24 sa doğrudan çok çıktılı tahmin |
| Naif referans nadiren raporlanıyor | §4.4 | Persistence, SN24, SN168 ve her modelin SN168'e göre kazancı |
| k-katlı CV, ayrı doğrulama kümesi yok, test tekrar kullanılıyor | [8], [15], [41] | Kronolojik train/val/test; Optuna train içi zamansal CV; test tek sefer |
| Hata ölçeği belirsiz (normalize mi, kW mı?) | §4.1 | Tüm metrikler kW'ta (ters dönüşüm sonrası); MAPE yalnız y > 0,1 kW |
| Tek seed, belirsizlik yok | çoğu çalışma | 3 seed, ortalama ± std |
| Anomali: kolay enjeksiyon, eşik testte, nokta metrik | [33], [34] | 4 tip, σ tabanlı büyüklük, val-dev'de eşik, olay F1, 5–10 enjeksiyon seed'i, point-adjust yok |
| Tahmin ve anomaliyi aynı hatta bağlayan, artık tabanlı ve OOF kalibreli dedektör nadir | [33] tahmin hatası + sabit eşik | A3: F10-opt artığı / OOF std (saat × gün) |
| Hane ve şebeke düzeyinin aynı ayarlarla karşılaştırılması | [37], [38] başka verilerle | IHEPC (hane) vs ELD-160 (şebeke), aynı model ve ayarlar |

---

## 8. Hocaya sunulabilecek cevap: "Skorunuz düşük"

Önerilen anlatım (dürüst ve kaynaklı):

1. "Literatürde bu veri setinde R² = 0,98 ya da MAPE %1 gibi sonuçlar var. Ancak bu sonuçlar 1 adım ileri, normalize ölçekte ve çoğu zaman k-katlı CV ile hesaplanmış [21], [8], [15]. Aynı tabloda RMSE 0,03 ile 1,15 arasında değişiyor; yani sayılar birbiriyle bile karşılaştırılamıyor (§4.1)."
2. "Bizim görevimiz daha zor: 24 saat ileri, kW ölçeğinde, kronolojik test, test bir kez. Bu protokole en yakın çalışmada [14] en iyi derin modeller R² ≈ 0,28'de kalıyor. Bizim final modelimiz 0,297, en iyi tekil modelimiz XGBoost 0,364."
3. "Tek hane tahmini doğası gereği zor. Toplulaştırınca aynı modelle R² 0,974'e çıkıyoruz; bu, literatürdeki ölçekleme bulgusuyla tutarlı [38]."
4. "Anomali tarafında hedefimizi tutturamadık; bunu raporda açıkça yazıyoruz. Literatürdeki yüksek F1'ler daha kolay enjeksiyon ve eşiğin testte ayarlanmasıyla elde edilmiş [33], [34]."

**Uyarı:** "Biz literatürden daha iyiyiz" demiyoruz. Koşullar farklı olduğu için böyle bir iddia savunulamaz (CLAUDE.md, rapor kuralları).

---

## 9. Literatürden çıkan olası ek adımlar (henüz yapılmadı)

Hiçbiri çalıştırılmadı. Test kümesi zaten bir kez kullanıldığı için yeni deneyler **yalnız val üzerinde** yapılmalı. Testte yeniden ölçüm gerekirse "test sonrası, final değil" diye işaretlenmeli.

1. **Literatür protokolünde ek ölçüm (en ikna edici adım).** Aynı modelleri val üzerinde (a) 1 saat ileri tahminle, (b) min-max normalize ölçekte metrikle hesaplamak. Böylece "protokol değişince sayılar ne kadar değişiyor?" sorusu kendi verimizle cevaplanır.
2. **Zhang vd. tipi enjeksiyon.** Val'de her gün bir saati ×2 yapıp nokta F1 hesaplamak; anomali sonuçlarının literatürle kıyaslanabilir bir versiyonunu elde etmek.
3. **DLinear / PatchTST referansı.** Zeng vd. [47] basit doğrusal modellerin Transformer'larla yarışabildiğini gösteriyor; Qarni vd. [30] IHEPC'de PatchTST kullanıyor.
4. **Olasılıksal tahmin / tahmin aralığı.** Hong ve Fan [48] ile Mathonsi ve van Zyl [35] tahmin aralığı tabanlı bir yaklaşım öneriyor. A3 dedektörüne alternatif olabilir.

---

## 10. Kaynakça (IEEE)

> **Doğrulama notu:** Künyeler 2026-11-09'da OpenAlex/Crossref kayıtlarından (DOI ile) kontrol edildi. "doğrulanmalı" yazan alanlar ya kayıtlarda tutarsız ya da tam metinden teyit edilemedi.

**Veri setleri**
- [1] G. Hébrail and A. Bérard, "Individual household electric power consumption," UCI Machine Learning Repository, 2012. doi: 10.24432/C58K54. *(yıl doğrulanmalı (DataCite 2006, UCI sayfası 2012))*
- [2] A. Trindade, "ElectricityLoadDiagrams20112014," UCI Machine Learning Repository, 2015. doi: 10.24432/C58C86.

**IHEPC üzerinde tahmin**
- [3] D. L. Marino, K. Amarasinghe, and M. Manic, "Building energy load forecasting using deep neural networks," in *Proc. IECON 2016 – 42nd Annu. Conf. IEEE Ind. Electron. Soc.*, 2016, pp. 7046–7051. doi: 10.1109/IECON.2016.7793413.
- [4] E. Mocanu, P. H. Nguyen, M. Gibescu, and W. L. Kling, "Deep learning for estimating building energy consumption," *Sustain. Energy Grids Netw.*, vol. 6, pp. 91–99, 2016. doi: 10.1016/j.segan.2016.02.005. *(IHEPC kullanımı doğrulanmalı)*
- [5] T.-Y. Kim and S.-B. Cho, "Predicting residential energy consumption using CNN-LSTM neural networks," *Energy*, vol. 182, pp. 72–81, 2019. doi: 10.1016/j.energy.2019.05.230.
- [6] J.-Y. Kim and S.-B. Cho, "Electric energy consumption prediction by deep learning with state explainable autoencoder," *Energies*, vol. 12, no. 4, Art. no. 739, 2019. doi: 10.3390/en12040739.
- [7] T. Le, M. T. Vo, B. Vo, E. Hwang, S. Rho, and S. W. Baik, "Improving electric energy consumption prediction using CNN and Bi-LSTM," *Appl. Sci.*, vol. 9, no. 20, Art. no. 4237, 2019. doi: 10.3390/app9204237.
- [8] F. U. M. Ullah, A. Ullah, I. U. Haq, S. Rho, and S. W. Baik, "Short-term prediction of residential power energy consumption via CNN and multi-layer bi-directional LSTM networks," *IEEE Access*, vol. 8, pp. 123369–123380, 2020. doi: 10.1109/ACCESS.2019.2963045.
- [9] Z. A. Khan, T. Hussain, A. Ullah, S. Rho, M. Lee, and S. W. Baik, "Towards efficient electricity forecasting in residential and commercial buildings: A novel hybrid CNN with a LSTM-AE based framework," *Sensors*, vol. 20, no. 5, Art. no. 1399, 2020. doi: 10.3390/s20051399.
- [10] M. Sajjad, Z. A. Khan, A. Ullah, T. Hussain, W. Ullah, M. Y. Lee, and S. W. Baik, "A novel CNN-GRU-based hybrid approach for short-term residential load forecasting," *IEEE Access*, vol. 8, pp. 143759–143768, 2020. doi: 10.1109/ACCESS.2020.3009537.
- [11] Z. A. Khan, A. Ullah, W. Ullah, S. Rho, M. Lee, and S. W. Baik, "Electrical energy prediction in residential buildings for short-term horizons using hybrid deep learning strategy," *Appl. Sci.*, vol. 10, no. 23, Art. no. 8634, 2020. doi: 10.3390/app10238634.
- [12] S.-J. Bu and S.-B. Cho, "Time series forecasting with multi-headed attention-based deep learning for residential energy consumption," *Energies*, vol. 13, no. 18, Art. no. 4722, 2020. doi: 10.3390/en13184722.
- [13] K. Yan, X. Wang, Y. Du, N. Jin, H. Huang, and H. Zhou, "Multi-step short-term power consumption forecasting with a hybrid deep learning strategy," *Energies*, vol. 11, no. 11, Art. no. 3089, 2018. doi: 10.3390/en11113089. *(beş veri setinden birinin IHEPC olduğu doğrulanmalı)*
- [14] A. Gasparin, S. Luković, and C. Alippi, "Deep learning for time series forecasting: The electric load case," *CAAI Trans. Intell. Technol.*, vol. 7, no. 1, pp. 1–25, 2022. doi: 10.1049/cit2.12060. *(OpenAlex yayın yılı 2021; ön baskı arXiv:1907.09207)*
- [15] A. Özcan, C. Catal, and A. Kasif, "Energy load forecasting using a dual-stage attention-based recurrent neural network," *Sensors*, vol. 21, no. 21, Art. no. 7115, 2021. doi: 10.3390/s21217115.
- [16] S. U. Khan, I. U. Haq, Z. A. Khan, N. Khan, M. Y. Lee, and S. W. Baik, "Atrous convolutions and residual GRU based architecture for matching power demand with supply," *Sensors*, vol. 21, no. 21, Art. no. 7191, 2021. doi: 10.3390/s21217191.
- [17] T. Han, K. Muhammad, T. Hussain, J. Lloret, and S. W. Baik, "An efficient deep learning framework for intelligent energy management in IoT networks," *IEEE Internet Things J.*, vol. 8, no. 5, pp. 3170–3179, 2021. doi: 10.1109/JIOT.2020.3013306.
- [18] S. K. Hora, R. Poongodan, R. P. de Prado, M. Woźniak, and P. B. Divakarachari, "Long short-term memory network-based metaheuristic for effective electric energy consumption prediction," *Appl. Sci.*, vol. 11, no. 23, Art. no. 11263, 2021. doi: 10.3390/app112311263.
- [19] D. Syed, H. Abu-Rub, A. Ghrayeb, and S. S. Refaat, "Household-level energy forecasting in smart buildings using a novel hybrid deep learning model," *IEEE Access*, vol. 9, pp. 33498–33511, 2021. doi: 10.1109/ACCESS.2021.3061370. *(iki veri setinden birinin IHEPC olduğu doğrulanmalı)*
- [20] M. Abdel-Basset, H. Hawash, K. M. Sallam, S. S. Askar, and M. Abouhawwash, "STLF-Net: Two-stream deep network for short-term load forecasting in residential buildings," *J. King Saud Univ. – Comput. Inf. Sci.*, vol. 34, no. 7, pp. 4296–4311, 2022. doi: 10.1016/j.jksuci.2022.04.016.
- [21] M. F. Alsharekh, S. Habib, D. A. Dewi, W. Albattah, M. Islam, and S. Albahli, "Improving the efficiency of multistep short-term electricity load forecasting via R-CNN with ML-LSTM," *Sensors*, vol. 22, no. 18, Art. no. 6913, 2022. doi: 10.3390/s22186913.
- [22] Z. A. Khan, A. Ullah, I. U. Haq, M. Hamdy, G. M. Mauro, K. Muhammad, M. Hijji, and S. W. Baik, "Efficient short-term electricity load forecasting for effective energy management," *Sustain. Energy Technol. Assess.*, vol. 53, Art. no. 102337, 2022. doi: 10.1016/j.seta.2022.102337.
- [23] O. Jogunola *et al.*, "CBLSTM-AE: A hybrid deep learning framework for predicting energy consumption," *Energies*, vol. 15, no. 3, Art. no. 810, 2022. doi: 10.3390/en15030810.
- [24] L. Cascone, S. Sadiq, S. Ullah, S. Mirjalili, H. U. R. Siddiqui, and M. Umer, "Predicting household electric power consumption using multi-step time series with convolutional LSTM," *Big Data Res.*, vol. 31, Art. no. 100360, 2023. doi: 10.1016/j.bdr.2022.100360. *(OpenAlex yayın yılı 2022; yıl doğrulanmalı)*
- [25] D. G. da Silva and A. A. de M. Meneses, "Comparing long short-term memory (LSTM) and bidirectional LSTM deep neural networks for power consumption prediction," *Energy Rep.*, vol. 10, pp. 3315–3334, 2023. doi: 10.1016/j.egyr.2023.09.175.
- [26] R. Gonçalves, V. M. Ribeiro, and F. L. Pereira, "Variable split convolutional attention: A novel deep learning model applied to the household electric power consumption," *Energy*, vol. 274, Art. no. 127321, 2023. doi: 10.1016/j.energy.2023.127321.
- [27] R. Rajabi and A. Estebsari, "Deep learning based forecasting of individual residential loads using recurrence plots," in *Proc. IEEE Milan PowerTech*, 2019, pp. 1–5. doi: 10.1109/PTC.2019.8810899.
- [28] A. Estebsari and R. Rajabi, "Single residential load forecasting using deep learning and image encoding techniques," *Electronics*, vol. 9, no. 1, Art. no. 68, 2020. doi: 10.3390/electronics9010068. *(IHEPC kullanımı doğrulanmalı)*
- [29] R. E. Alden, H. Gong, C. Ababei, and D. M. Ionel, "LSTM forecasts for smart home electricity usage," in *Proc. 9th Int. Conf. Renewable Energy Res. Appl. (ICRERA)*, 2020, pp. 434–438. doi: 10.1109/ICRERA49962.2020.9242804. *(UCI kullanımı doğrulanmalı)*
- [30] A. M. Qarni, I. U. Haq, A. Iqbal, and S. M. Elkhider, "MCV-PatchTST: multi-scale cross-variate patch transformers for household short-term load forecasting," *Sci. Rep.*, 2026. doi: 10.1038/s41598-026-67495-w. *(cilt/makale no doğrulanmalı)*
- [31] E. A. Abbas, R. Alubady, A. Sahi, and M. Diykh, "A hybrid CNN–Transformer deep learning model for household electric energy consumption forecasting," *Int. J. Energy Res.*, vol. 2026, 2026. doi: 10.1155/er/3999444. *(makale no doğrulanmalı)*
- [32] B. S. H. Hanna, G. Xu, X. Wang, and M. J. Hossain, "A systematic comparison of machine learning models for missing value imputation in household electricity consumption data," *Sci. Rep.*, vol. 16, 2026. doi: 10.1038/s41598-026-51159-w.

**IHEPC üzerinde anomali / aykırı değer tespiti**
- [33] J. Zhang, H. Zhang, S. Ding, and X. Zhang, "Power consumption predicting and anomaly detection based on Transformer and K-means," *Front. Energy Res.*, vol. 9, Art. no. 779587, 2021. doi: 10.3389/fenrg.2021.779587.
- [34] K. Kea, Y. Han, and T.-K. Kim, "Enhancing anomaly detection in distributed power systems using autoencoder-based federated learning," *PLoS ONE*, vol. 18, no. 8, Art. no. e0290337, 2023. doi: 10.1371/journal.pone.0290337.
- [35] T. Mathonsi and T. L. van Zyl, "Multivariate anomaly detection based on prediction intervals constructed using deep learning," *Neural Comput. Appl.*, vol. 37, no. 2, pp. 707–721, 2025. doi: 10.1007/s00521-021-06697-x. *(OpenAlex çevrimiçi yılı 2022; ön baskı arXiv:2110.03393)*
- [36] I. Amalou, N. Mouhni, and A. Abdali, "Kernel function selection for FISVDD applied to outliers detection application for energy time series datasets," *Electr. Power Compon. Syst.*, vol. 53, no. 19, pp. 2460–2475. doi: 10.1080/15325008.2024.2350058. *(yıl doğrulanmalı: OpenAlex 2024, cilt 53)*

**Toplulaştırma ve hane düzeyi tahmin**
- [37] S. Humeau, T. K. Wijaya, M. Vasirani, and K. Aberer, "Electricity load forecasting for residential customers: Exploiting aggregation and correlation between households," in *Proc. Sustain. Internet ICT Sustain. (SustainIT)*, 2013, pp. 1–6. doi: 10.1109/SustainIT.2013.6685208.
- [38] R. Sevlian and R. Rajagopal, "A scaling law for short term load forecasting on varying levels of aggregation," *Int. J. Electr. Power Energy Syst.*, vol. 98, pp. 350–361, 2018. doi: 10.1016/j.ijepes.2017.10.032.
- [39] S. Haben, S. Arora, G. Giasemidis, M. Voss, and D. V. Greetham, "Review of low voltage load forecasting: Methods, applications, and recommendations," *Appl. Energy*, vol. 304, Art. no. 117798, 2021. doi: 10.1016/j.apenergy.2021.117798.
- [40] W. Kong, Z. Y. Dong, Y. Jia, D. J. Hill, Y. Xu, and Y. Zhang, "Short-term residential load forecasting based on LSTM recurrent neural network," *IEEE Trans. Smart Grid*, vol. 10, no. 1, pp. 841–851, 2019. doi: 10.1109/TSG.2017.2753802. *(farklı veri seti; hane düzeyinde LSTM için genel kaynak)*

**Değerlendirme yöntemi**
- [41] H. Hewamalage, K. Ackermann, and C. Bergmeir, "Forecast evaluation for data scientists: Common pitfalls and best practices," *Data Min. Knowl. Discov.*, vol. 37, no. 2, pp. 788–832, 2023. doi: 10.1007/s10618-022-00894-5.
- [42] C. Bergmeir and J. M. Benítez, "On the use of cross-validation for time series predictor evaluation," *Inf. Sci.*, vol. 191, pp. 192–213, 2012. doi: 10.1016/j.ins.2011.12.028.
- [43] C. Bergmeir, R. J. Hyndman, and B. Koo, "A note on the validity of cross-validation for evaluating autoregressive time series prediction," *Comput. Stat. Data Anal.*, vol. 120, pp. 70–83, 2018. doi: 10.1016/j.csda.2017.11.003.
- [44] R. J. Hyndman and A. B. Koehler, "Another look at measures of forecast accuracy," *Int. J. Forecast.*, vol. 22, no. 4, pp. 679–688, 2006. doi: 10.1016/j.ijforecast.2006.03.001.
- [45] S. Kim, K. Choi, H.-S. Choi, B. Lee, and S. Yoon, "Towards a rigorous evaluation of time-series anomaly detection," in *Proc. AAAI Conf. Artif. Intell.*, vol. 36, no. 7, 2022, pp. 7194–7201. doi: 10.1609/aaai.v36i7.20680.
- [46] R. Wu and E. Keogh, "Current time series anomaly detection benchmarks are flawed and are creating the illusion of progress," *IEEE Trans. Knowl. Data Eng.*, 2021 (erken erişim). doi: 10.1109/TKDE.2021.3112126. *(nihai cilt/sayfa doğrulanmalı)*
- [47] A. Zeng, M. Chen, L. Zhang, and Q. Xu, "Are transformers effective for time series forecasting?," in *Proc. AAAI Conf. Artif. Intell.*, vol. 37, no. 9, 2023, pp. 11121–11128. doi: 10.1609/aaai.v37i9.26317.
- [48] T. Hong and S. Fan, "Probabilistic electric load forecasting: A tutorial review," *Int. J. Forecast.*, vol. 32, no. 3, pp. 914–938, 2016. doi: 10.1016/j.ijforecast.2015.11.011.

**Temel modeller ve anomali yöntemleri**
- [49] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Comput.*, vol. 9, no. 8, pp. 1735–1780, 1997. doi: 10.1162/neco.1997.9.8.1735.
- [50] K. Cho *et al.*, "Learning phrase representations using RNN encoder–decoder for statistical machine translation," in *Proc. EMNLP*, 2014, pp. 1724–1734. doi: 10.3115/v1/D14-1179.
- [51] V. Chandola, A. Banerjee, and V. Kumar, "Anomaly detection: A survey," *ACM Comput. Surv.*, vol. 41, no. 3, pp. 1–58, 2009. doi: 10.1145/1541880.1541882.
- [52] F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation forest," in *Proc. IEEE Int. Conf. Data Mining (ICDM)*, 2008, pp. 413–422. doi: 10.1109/ICDM.2008.17.
- [53] K. Hundman, V. Constantinou, C. Laporte, I. Colwell, and T. Soderstrom, "Detecting spacecraft anomalies using LSTMs and nonparametric dynamic thresholding," in *Proc. ACM SIGKDD*, 2018, pp. 387–395. doi: 10.1145/3219819.3219845.
- [54] Y. Himeur, K. Ghanem, A. Alsalemi, F. Bensaali, and A. Amira, "Artificial intelligence based anomaly detection of energy consumption in buildings: A review, current trends and new perspectives," *Appl. Energy*, vol. 287, Art. no. 116601, 2021. doi: 10.1016/j.apenergy.2021.116601.

**ELD / uzun ufuk kıyas çalışmaları**
- [55] G. Lai, W.-C. Chang, Y. Yang, and H. Liu, "Modeling long- and short-term temporal patterns with deep neural networks," in *Proc. 41st Int. ACM SIGIR Conf.*, 2018, pp. 95–104. doi: 10.1145/3209978.3210006.
- [56] H. Zhou *et al.*, "Informer: Beyond efficient transformer for long sequence time-series forecasting," in *Proc. AAAI Conf. Artif. Intell.*, vol. 35, no. 12, 2021, pp. 11106–11115. doi: 10.1609/aaai.v35i12.17325.
- [57] Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, "A time series is worth 64 words: Long-term forecasting with transformers," in *Proc. ICLR*, 2023. arXiv:2211.14730. *(ICLR künyesi doğrulanmalı)*

> README §11'deki eski havuzla fark: Malhotra vd. (ESANN 2015; arXiv:1607.00148) bu taramada yeniden doğrulanmadı; rapora girecekse ayrıca kontrol edilmeli. Kong vd. [40] künyesi doğrulandı. Marino vd. [3] için sayfa 7046–7051 doğrulandı.
