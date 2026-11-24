# Literatür Matrisi

> Kurallar (README §11, CLAUDE.md): Yalnızca **okunan ve Google Scholar / DOI ile doğrulanan** kaynaklar eklenir. Emin olunmayan bilgi "doğrulanmalı" diye işaretlenir. Sayısal sonuçlar makaleden birebir alınır; kendi sonuçlarımızla kıyaslarken veri seti, çözünürlük, ufuk, bölme ve metrik farkları belirtilir.

- Anlatı, karşılaştırma ve kaynakça: `docs/literatur_taramasi.md`. Köşeli parantezdeki numaralar ([n]) oradaki kaynakçayla aynıdır.
- Taramanın ham çıktısı (271 kayıt, elle elenmemiş): `docs/literatur_katalog.csv`.
- **Doğrulandı mı?** sütunundaki kısaltmalar:
  - **DOI** = künye DOI kaydıyla doğrulandı,
  - **TM** = tam metin okundu,
  - **Ö** = yalnız özet okundu,
  - **İ** = sayılar başka bir makalenin tablosundan aktarıldı.
- Tarama tarihi: 2026-11-09.

## A. IHEPC üzerinde tahmin çalışmaları

| # | Kaynak (IEEE) | Doğrulandı mı? | Veri seti | Görev | Model | Bölme yöntemi | Metrik ve sonuç | Güçlü yön | Zayıf yön | Bize katkısı |
|---|---|---|---|---|---|---|---|---|---|---|
| [3] | Marino, Amarasinghe, Manic, IECON 2016 | DOI, Ö | Tek konut müşterisi (tam metin dizininde IHEPC adı geçiyor) | 1 dk ve 1 sa tahmin | LSTM, LSTM-S2S | doğrulanmalı | Standart LSTM 1 dk'da başarısız, 1 sa'te iyi; S2S ikisinde de iyi | Standart LSTM ile S2S'i aynı veride iki çözünürlükte karşılaştırıyor | Ayrıntılar özetten çıkarılamadı | F10 seq2seq tercihimiz için öncül kaynak |
| [5] | Kim & Cho, *Energy* 2019 | DOI, Ö, İ | IHEPC | Çok değişkenli tüketim tahmini | CNN-LSTM | doğrulanmalı | Saatlik: RMSE 0,595, MAE 0,3317, MAPE %32,83. Doğrusal regresyon: RMSE 0,657, MAPE %83,74 (İ: [21]) | En çok atıf alan IHEPC hibrit modeli; alt sayaçların etkisini inceliyor | Tam metin ücretli; ölçek ve ufuk doğrulanmalı | F11 CNN-LSTM mimarimizin kaynağı |
| [6] | Kim & Cho, *Energies* 2019 | DOI, Ö | IHEPC | 60 dk geçmiş → 15/30/45/60 dk | Durum açıklanabilir otokodlayıcı | doğrulanmalı | MSE 0,384 | t-SNE ile açıklanabilirlik | Çok kısa ufuk | Kısa ufuk ile 24 sa ufkun farkını anlatırken örnek |
| [7] | Le vd., *Appl. Sci.* 2019 | DOI, Ö | IHEPC | Gerçek zamanlıdan uzun vadeye | CNN + Bi-LSTM (EECP-CBL) | doğrulanmalı | "Literatürdeki yöntemlerden iyi" | Birden çok zaman ölçeği | Protokol ve ölçek özetten çıkarılamadı | Savunma notu: Bi-LSTM'nin geriye giden yönü yalnız girdi penceresinin içinde çalışır, hedef saatleri görmez; bu yüzden tek başına sızıntı değildir |
| [8] | Ullah vd., *IEEE Access* 2020 | DOI, Ö, İ | IHEPC | Kısa vadeli tahmin | CNN + çok katmanlı Bi-LSTM | **10 katlı CV** + hold-out | RMSE 0,565, MAE 0,346, MAPE %29,10 (İ: [21]) | Ön işleme adımları ayrıntılı | Zaman serisinde k-katlı CV | §4.3 bölme eleştirisi örneği |
| [9] | Khan vd., *Sensors* 2020 | DOI, **TM** | IHEPC (+ Kore ticari bina) | Saatlik ve günlük tahmin | CNN + LSTM-AE | %75/%25: ilk 3 yıl eğitim, son yıl test; eğitimde 0,2 doğrulama oranı | Saatlik MSE 0,19, MAE 0,31, RMSE 0,47 | Kronolojik test; çoklu normalizasyon karşılaştırması | Aykırı değerler veriden çıkarılıyor; ölçek belirtilmemiş | Aykırı değer silmenin anomali göreviyle çelişkisi (§4.5) |
| [10] | Sajjad vd., *IEEE Access* 2020 | DOI, Ö, İ | IHEPC + AEP | Kısa vadeli tahmin | CNN-GRU | doğrulanmalı | RMSE 0,47, MAE 0,33, MSE 0,22 (İ: [21]) | En çok atıf alan IHEPC makalesi (557, OpenAlex) | Protokol ayrıntıları özetten çıkarılamadı | GRU (F6) karşılaştırması için |
| [12] | Bu & Cho, *Energies* 2020 | DOI, Ö | IHEPC | Çok değişkenli tahmin | Çok başlıklı dikkat + CRNN | doğrulanmalı | Önceki en iyi modele göre hata %31,01 daha az | Dikkat mekanizmasının katkısı ölçülmüş | Göreli kazanç; mutlak ölçek belirsiz | Dikkat tabanlı alternatifler |
| [13] | Yan vd., *Energies* 2018 | DOI, Ö | 5 hane veri seti (IHEPC olduğu doğrulanmalı) | **Çok adımlı** tahmin | CNN-LSTM | doğrulanmalı | ARIMA, persistence, SVR, LSTM'den iyi | Tek adım tahmini açıkça eleştiriyor; persistence modelini dahil ediyor | Veri seti listesi doğrulanmalı | 24 sa ufuk ve naif referans gerekçesi |
| [14] | **Gasparin, Luković, Alippi, *CAAI TIT* 2022** | DOI, **TM** (arXiv/ar5iv) | IHEPC (15 dk) + GEFCom2014 + diğerleri | **Gün öncesi (96 adım)** | FNN, ERNN, LSTM, GRU, S2S, TCN (MIMO ve özyinelemeli) | **Kronolojik: son yıl test**; her 5 aydan biri doğrulama | En iyi GRU (MIMO): RMSE 0,75, MAE 0,52, **R² 0,279**; 10 tekrar ort ± std | Bize en yakın protokol; çoklu tekrar; MIMO ile özyinelemeli karşılaştırması | Eksikler sonraki yılların ortalamasıyla dolduruluyor; naif referans yok | **Ana kıyas kaynağı** (§5.1): R² ≈ 0,28 tavanı |
| [15] | Özcan vd., *Sensors* 2021 | DOI, **TM** | IHEPC (+ HEDG) | 1 dk … 1 hafta | Çift aşamalı dikkat RNN | **10 katlı CV** | 1 sa: MAE 240,14, RMSE 300,18 (birim yok) | Çok sayıda zaman ölçeği | Başka veri setlerindeki sonuçlarla aynı tabloda kıyas; min-max ölçeği; birim yok | §4.1 kıyaslanamazlık örneği |
| [16] | S. U. Khan vd., *Sensors* 2021 | DOI, İ | IHEPC (diğer veri setleri doğrulanmalı) | doğrulanmalı | Atrous CNN + residual GRU | doğrulanmalı | RMSE 0,4186, MAE 0,2635 (İ: [21]) | — | — | Alsharekh tablosundaki karşılaştırma |
| [20] | Abdel-Basset vd., *JKSU-CIS* 2022 | DOI, Ö, İ | IHEPC + AEP | **1 saat ileri** | GRU + TCN (STLF-Net) | doğrulanmalı | RMSE 0,4386, MAPE %36,24 (İ: [21]) | İki akışlı mimari | Yalnız 1 sa ufuk | §4.2 ufuk farkı |
| [21] | Alsharekh vd., *Sensors* 2022 | DOI, **TM** | IHEPC + PJM | Saatlik, günlük | Residual CNN + çok katmanlı LSTM | Tam metinde bulunamadı | Saatlik RMSE 0,0325, MAE 0,0144, R² 0,9841, MAPE %1,024 | **16 önceki çalışmanın sonuçlarını tek tabloda topluyor** | 3-sigma ile aykırı değer silme; min-max; metin ile tablo arasında tutarsızlık; persistence elenmiş | §4.1'deki karşılaştırma tablosunun kaynağı |
| [25] | da Silva & Meneses, *Energy Rep.* 2023 | DOI, Ö | 4 veri seti (Fransa hane, üniversite binası, Tétouan, Singapur) | Tek değişkenli kısa vadeli | LSTM vs Bi-LSTM | **Zaman serisi CV** | Bi-LSTM > LSTM (Friedman p = 0,0455) | İstatistiksel test kullanıyor | — | Modeller arası farkın anlamlılığını test etme örneği |
| [26] | Gonçalves vd., *Energy* 2023 | DOI, Ö | IHEPC | Çok değişkenli tahmin | ConvLSTM + değişken bölmeli konvolüsyonel dikkat | doğrulanmalı | "En düşük hata" | Konvolüsyonlarda nedensel dolgu (causal padding) kullanıyor (özetten) | Sayılar özette yok | Nedensel dolgu konusunu bizim sızıntı kurallarımızla birlikte tartışmak için |
| [29] | Alden vd., ICRERA 2020 | DOI, Ö | 4 akıllı ev veri seti (UCI olduğu doğrulanmalı) | 1/6/24 sa, 1/7 gün | LSTM | doğrulanmalı | 24/6/1 sa ufukta tatmin edici değil | Birden çok veri seti | — | Tek hanede 24 sa tahminin zorluğu |
| [30] | Qarni vd., *Sci. Rep.* 2026 | DOI, Ö | IHEPC | Saatlik, günlük, haftalık | MCV-PatchTST | doğrulanmalı | PatchTST'ye göre MAE %10–17 daha düşük | Güncel Transformer çizgisi | — | PatchTST referansı fikri (§9) |
| [32] | Hanna vd., *Sci. Rep.* 2026 | DOI, Ö + TM (kısmi) | IHEPC | Eksik değer doldurma | 30 ML modeli | doğrulanmalı | Extra Trees MAE 0,0474 kW; ağaç toplulukları ≥ derin ağlar | Kapsamlı doldurma karşılaştırması | — | Bizim basit doldurma kararımızın alternatifi |

## B. IHEPC üzerinde anomali / aykırı değer tespiti

| # | Kaynak (IEEE) | Doğrulandı mı? | Veri seti | Görev | Model | Bölme yöntemi | Metrik ve sonuç | Güçlü yön | Zayıf yön | Bize katkısı |
|---|---|---|---|---|---|---|---|---|---|---|
| [33] | Zhang vd., *Front. Energy Res.* 2021 | DOI, **TM** | IHEPC (saatlik) | 1 sa ileri tahmin + anomali | Transformer + K-means; tahmin hatasına eşik | 1.240 gün eğitim / 200 gün test (kronolojik olup olmadığı belirtilmemiş) | Nokta F1 0,72 (P 0,80, R 0,66); LSTM 0,66 | Tahmin ve anomaliyi aynı hatta bağlıyor (A3'ümüze benzer) | Her gün tek nokta ×2 enjeksiyonu; eşik "deneme-yanılma" ile; metin ile tablo tutarsız | En yakın anomali çalışması; kolay protokol örneği (§6) |
| [34] | Kea vd., *PLoS ONE* 2023 | DOI, **TM** | IHEPC (saatlik, 6 istemciye bölünmüş) | Anomali tespiti (federe) | Otokodlayıcı + federe öğrenme; POT eşiği | Global model tüm veride (36.000 kayıt) değerlendirilmiş | F1 0,9950; accuracy 0,9968 | Kod ve veri bağlantısı açık; POT ile otomatik eşik | Test değerleri ×2 ile etiketleniyor; değerlendirme eğitim verisini içeriyor | Accuracy'nin dengesiz veride yanıltıcı olması örneği |
| [35] | Mathonsi & van Zyl, *Neural Comput. Appl.* 2025 | DOI, **TM** (ar5iv) | IHEPC (+ 4 veri seti) | Tahmin aralığı tabanlı anomali | LSTM, ESN, CCN, SARIMAX | **Kronolojik: ilk 3 yıl / son yıl; walk-forward ayar** | RMSE: persistence 465 → LSTM 337,95 (birim yok); F1 yalnız şekilde | Kronolojik bölme + persistence referansı | Enjeksiyon ayrıntısı verilmemiş | Tahmin aralığı fikri (§9); naif referans örneği |
| [36] | Amalou vd., *Electr. Power Compon. Syst.* | DOI, Ö | IHEPC + şebeke kararlılığı (simülasyon) | Aykırı değer temizliği | FISVDD (çekirdek seçimi) | doğrulanmalı | GRU tahmin MSE'sine etkisi | Artımlı (online) yöntem | Etiketli değerlendirme yok | Aykırı değerleri silme / tutma tartışması |

## C. Yöntem ve değerlendirme kaynakları (veri setinden bağımsız)

| # | Kaynak | Doğrulandı mı? | Bize katkısı |
|---|---|---|---|
| [37]–[39] | Humeau vd. 2013; Sevlian & Rajagopal 2018; Haben vd. 2021 | DOI, Ö | Toplulaştırma arttıkça göreli hata azalır → hane ve şebeke karşılaştırmamızın literatür dayanağı |
| [41] | Hewamalage, Ackermann, Bergmeir 2023 | DOI, Ö | Yaygın değerlendirme hataları (bölme, ölçek, naif referans) |
| [42], [43] | Bergmeir & Benítez 2012; Bergmeir, Hyndman, Koo 2018 | DOI | k-katlı CV'nin zaman serisinde ne zaman geçerli olduğu (nüanslı) |
| [44] | Hyndman & Koehler 2006 | DOI | MAPE'nin sıfıra yakın değerlerde bozulması → MAPE yalnız y > 0,1 kW |
| [45] | Kim vd., AAAI 2022 | DOI | Point-adjust eleştirisi → olay F1'i ayrı raporladık |
| [46] | Wu & Keogh, IEEE TKDE | DOI (erken erişim) | "Önemsiz" anomali kıyas kümeleri eleştirisi |
| [47] | Zeng vd., AAAI 2023 | DOI | Basit doğrusal modeller Transformer'larla yarışabiliyor → Ridge/XGBoost'un güçlü çıkması şaşırtıcı değil |
