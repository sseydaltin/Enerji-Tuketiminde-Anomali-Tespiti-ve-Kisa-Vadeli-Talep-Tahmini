# Literatür Matrisi

> Kurallar (README §11, CLAUDE.md): Yalnızca **okunan ve Google Scholar / DOI ile doğrulanan** kaynaklar eklenir. Emin olunmayan bilgi "doğrulanmalı" diye işaretlenir. Sayısal sonuçlar makaleden birebir alınır; kendi sonuçlarımızla kıyaslarken veri seti, çözünürlük, ufuk, bölme ve metrik farkları belirtilir.

## Doğrulama yöntemi (2026-11-06)
- Künyeler kayıt kurumlarının API'lerinden alındı: **Crossref** (DOI'li makaleler), **arXiv API** (ön baskılar), **DataCite** (UCI veri setleri).
- Özetler Crossref, arXiv veya Semantic Scholar API'sinden alındı. Özet alınamayan kaynaklarda içerik alanları **boş** bırakıldı.
- Bu doğrulama yalnızca **künye** doğrulamasıdır. İçerik alanları özetten çıkarıldı; makaleler henüz **okunmadı**. Her satır, rapora girmeden önce okunarak doğrulanmalıdır.
- "DOĞRULANAMADI": kayıt kurumunda bulunamayan ya da README ile çelişen bilgi. Düzeltme uydurulmadı.

## Künye doğrulama tablosu

| # | Künye (IEEE, kayıt kurumundan) | DOI / bağlantı | Künye durumu | README §11'e göre fark |
|---|---|---|---|---|
| 1 | S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997. | 10.1162/neco.1997.9.8.1735 | Doğrulandı (Crossref) | Yok |
| 2 | K. Cho, B. van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, "Learning phrase representations using RNN encoder–decoder for statistical machine translation," in *Proc. 2014 Conf. Empirical Methods in Natural Language Processing (EMNLP)*, 2014, pp. 1724–1734. | 10.3115/v1/D14-1179 | Doğrulandı (Crossref) | README'de sayfa yoktu |
| 3 | J. Chung, C. Gulcehre, K. Cho, and Y. Bengio, "Empirical evaluation of gated recurrent neural networks on sequence modeling," arXiv:1412.3555, 2014. | arxiv.org/abs/1412.3555 | Doğrulandı (arXiv) | arXiv notu: "Presented in NIPS 2014 Deep Learning and Representation Learning Workshop" |
| 4 | D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. 3rd Int. Conf. Learning Representations (ICLR)*, San Diego, CA, USA, 2015. arXiv:1412.6980. | arxiv.org/abs/1412.6980 | Doğrulandı (arXiv; konferans bilgisi arXiv notundan) | Yok |
| 5 | W. Kong, Z. Y. Dong, Y. Jia, D. J. Hill, Y. Xu, and Y. Zhang, "Short-term residential load forecasting based on LSTM recurrent neural network," *IEEE Trans. Smart Grid*, vol. 10, no. 1, pp. 841–851, 2019. | 10.1109/TSG.2017.2753802 | Doğrulandı (Crossref) | Yok |
| 6 | D. L. Marino, K. Amarasinghe, and M. Manic, "Building energy load forecasting using deep neural networks," in *Proc. IECON 2016 – 42nd Annu. Conf. IEEE Industrial Electronics Society*, 2016, pp. 7046–7051. | 10.1109/IECON.2016.7793413 | Doğrulandı (Crossref) | README'de sayfa yoktu |
| 7 | T.-Y. Kim and S.-B. Cho, "Predicting residential energy consumption using CNN-LSTM neural networks," *Energy*, vol. 182, pp. 72–81, 2019. | 10.1016/j.energy.2019.05.230 | Doğrulandı (Crossref) | Yok |
| 8 | T. Hong and S. Fan, "Probabilistic electric load forecasting: A tutorial review," *Int. J. Forecasting*, vol. 32, no. 3, pp. 914–938, 2016. | 10.1016/j.ijforecast.2015.11.011 | Doğrulandı (Crossref) | Yok |
| 9 | R. J. Hyndman and A. B. Koehler, "Another look at measures of forecast accuracy," *Int. J. Forecasting*, vol. 22, no. 4, pp. 679–688, 2006. | 10.1016/j.ijforecast.2006.03.001 | Doğrulandı (Crossref) | Yok |
| 10 | V. Chandola, A. Banerjee, and V. Kumar, "Anomaly detection: A survey," *ACM Computing Surveys*, vol. 41, no. 3, pp. 1–58, 2009. | 10.1145/1541880.1541882 | Doğrulandı (Crossref; Crossref'teki başlık yalnızca "Anomaly detection") | README'de sayfa yoktu |
| 11 | P. Malhotra, L. Vig, G. Shroff, and P. Agarwal, "Long short term memory networks for anomaly detection in time series," in *Proc. ESANN*, 2015. | — | **DOĞRULANAMADI.** DOI kaydı yok; ESANN sayfası kullanım limiti nedeniyle açılamadı. Sayfa ve ISBN eksik. | — |
| 12 | P. Malhotra, A. Ramakrishnan, G. Anand, L. Vig, P. Agarwal, and G. Shroff, "LSTM-based encoder-decoder for multi-sensor anomaly detection," arXiv:1607.00148, 2016. | arxiv.org/abs/1607.00148 | Doğrulandı (arXiv) | arXiv notu: "Accepted at ICML 2016 Anomaly Detection Workshop" |
| 13 | K. Hundman, V. Constantinou, C. Laporte, I. Colwell, and T. Soderstrom, "Detecting spacecraft anomalies using LSTMs and nonparametric dynamic thresholding," in *Proc. 24th ACM SIGKDD Int. Conf. Knowledge Discovery & Data Mining*, 2018, pp. 387–395. | 10.1145/3219819.3219845 | Doğrulandı (Crossref) | README'de sayfa yoktu |
| 14 | F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation forest," in *Proc. 8th IEEE Int. Conf. Data Mining (ICDM)*, 2008, pp. 413–422. | 10.1109/ICDM.2008.17 | Doğrulandı (Crossref) | README'de sayfa yoktu |
| 15 | Y. Himeur, K. Ghanem, A. Alsalemi, F. Bensaali, and A. Amira, "Artificial intelligence based anomaly detection of energy consumption in buildings: A review, current trends and new perspectives," *Applied Energy*, vol. 287, Art. no. 116601, 2021. | 10.1016/j.apenergy.2021.116601 | Doğrulandı (Crossref) | README'de "et al." ve makale no. yoktu |
| 16 | G. Hebrail and A. Berard, "Individual household electric power consumption," UCI Machine Learning Repository, [yıl DOĞRULANAMADI]. doi: 10.24432/C58K54. | 10.24432/C58K54 | Yazar, başlık ve DOI doğrulandı (DataCite). **Yıl DOĞRULANAMADI.** | DataCite yayın yılı **2006**, README'de **2012**. UCI sayfası kullanım limiti nedeniyle açılamadı. |
| 17 | A. Trindade, "ElectricityLoadDiagrams20112014," UCI Machine Learning Repository, 2015. doi: 10.24432/C58C86. | 10.24432/C58C86 | Doğrulandı (DataCite) | Yok |

## Aynı veri setlerini kullanan çalışmalar
- **UCI Household:** [7] Kim ve Cho (2019). Özetine göre bu çalışma "individual household power consumption" veri setini kullanıyor. [6] Marino vd. (2016) özetinde "one residential customer" benchmark verisinden söz ediyor; aynı veri seti olduğu özette yazmıyor, okunarak doğrulanmalı.
- **ElectricityLoadDiagrams ve daha güncel (2022+) çalışmalar:** **Henüz bulunamadı.** Arama kullanım limiti nedeniyle yarıda kaldı. Aday olarak görülen M.-P. Wu ve F. Wu, "Predicting residential electricity consumption using CNN-BiLSTM-SA neural networks," *IEEE Access*, vol. 12, pp. 71555–71565, 2024, doi: 10.1109/ACCESS.2024.3400972 için künye Crossref'te var; ancak hangi veri setini kullandığı **DOĞRULANAMADI**, bu yüzden matrise eklenmedi.

## İçerik matrisi
> Her satır: **özetten çıkarıldı, okunarak doğrulanmalı.** Sayısal sonuç yalnız özette açıkça yazıyorsa girildi. "Özet alınamadı" yazan satırlar boştur.

| # | Veri seti | Model / yöntem | Metrik | Raporlanan sonuç (özetten) | Güçlü yön | Zayıf yön | Bu projeye katkısı | Not |
|---|---|---|---|---|---|---|---|---|
| 1 | Özet alınamadı | LSTM | — | — | — | — | LSTM mimarisinin temel kaynağı | Özet alınamadı; okunarak doğrulanmalı |
| 2 | Özet alınamadı | RNN kodlayıcı–kod çözücü; GRU birimi | — | — | — | — | GRU'nun temel kaynağı | Özet alınamadı; okunarak doğrulanmalı |
| 3 | Polifonik müzik ve konuşma sinyali modelleme | LSTM, GRU, tanh RNN | Özette belirtilmemiş | Kapılı birimler tanh birimlerinden iyi; GRU, LSTM ile "comparable" | Kapılı birimlerin doğrudan karşılaştırması | Enerji verisi değil | AS2/H2 (LSTM–GRU kıyası) gerekçesi | Özetten çıkarıldı, okunarak doğrulanmalı |
| 4 | Özette belirtilmemiş | Adam optimizasyonu | — | Özette sayısal sonuç yok | Az hiperparametre ayarı gerektirir (özet) | — | Eğitimde kullanılan optimizer | Özetten çıkarıldı, okunarak doğrulanmalı |
| 5 | Kamuya açık gerçek konut akıllı sayaç verisi (adı özette yok) | LSTM | Özette belirtilmemiş | LSTM, listelenen rakip algoritmalardan iyi | Tek hane düzeyinde tahminin zorluğunu (yüksek oynaklık) vurgular | Veri seti adı ve metrikler özette yok | Hane düzeyi tahminin zorluğu; LSTM seçimi | Özetten çıkarıldı, okunarak doğrulanmalı |
| 6 | Tek konut müşterisinin elektrik tüketimi; 1 saat ve 1 dakika çözünürlük | Standart LSTM, LSTM tabanlı S2S | Özette belirtilmemiş | Standart LSTM 1 dakikada başarısız, 1 saatte iyi; S2S ikisinde de iyi | Çözünürlük etkisini gösterir | Tek müşteri | Saatlik çözünürlük seçimi için destek | Özetten çıkarıldı, okunarak doğrulanmalı |
| 7 | Individual household power consumption | CNN-LSTM | RMSE | Kıyaslanan yöntemler içinde en küçük RMSE (sayı özette yok) | Aynı veri seti | Bölme yöntemi özette yok | Aynı veri setinde literatür kıyası (koşullar karşılaştırılarak) | Özetten çıkarıldı, okunarak doğrulanmalı |
| 8 | Özet alınamadı | Olasılıksal yük tahmini (derleme) | — | — | — | — | Gelecek çalışmalar: olasılıksal tahmin | Özet alınamadı; okunarak doğrulanmalı |
| 9 | M ve M3 yarışmaları | Doğruluk ölçütleri | MASE önerisi | Birçok ölçüt yetersiz ve sık görülen durumlarda dejenere; MASE önerilir | Metrik seçimini gerekçelendirir | — | MAPE'nin sıfıra yakın değerlerde sorunu; sMAPE/MAE seçimi | Özetten çıkarıldı, okunarak doğrulanmalı |
| 10 | Özet alınamadı | Anomali tespiti (derleme) | — | — | — | — | Anomali tanımları ve türleri | Özet alınamadı; okunarak doğrulanmalı |
| 11 | DOĞRULANAMADI | — | — | — | — | — | — | Künye doğrulanmadan kullanılmamalı |
| 12 | Power demand, space shuttle, ECG ve iki motor veri seti | LSTM kodlayıcı–kod çözücü (EncDec-AD), yeniden yapılandırma hatası | Özette belirtilmemiş | Öngörülebilir ve öngörülemez serilerde sağlam; uzunluk 30–500 seriler | Yeniden yapılandırma tabanlı tespit; "power demand" verisi | Sayısal sonuç özette yok | LSTM-AE yaklaşımının temeli | Özetten çıkarıldı, okunarak doğrulanmalı |
| 13 | SMAP uydusu ve MSL (Curiosity) telemetrisi, uzman etiketli | LSTM tahmini + parametrik olmayan dinamik eşik | Özette belirtilmemiş | Özette sayısal sonuç yok | Tahmin artığı tabanlı tespit; yanlış alarm azaltma | Enerji verisi değil | GRU artık dedektörü ve eşik seçimi | Özetten çıkarıldı, okunarak doğrulanmalı |
| 14 | Özet alınamadı | Isolation Forest | — | — | — | — | Anomali referans yöntemi | Özet alınamadı; okunarak doğrulanmalı |
| 15 | Özet alınamadı | Binalarda YZ tabanlı anomali tespiti (derleme) | — | — | — | — | Enerjide anomali problemi ve motivasyon | Özet alınamadı; okunarak doğrulanmalı |
| 16 | Veri seti kaydı | — | — | — | — | — | Ana veri seti | Kaynak özelliklerini UCI sayfasından doğrulayın |
| 17 | Veri seti kaydı | — | — | — | — | — | Şebeke düzeyi ek deney | Kaynak özelliklerini UCI sayfasından doğrulayın |
