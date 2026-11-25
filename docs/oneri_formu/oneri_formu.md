# Proje Öneri Formu – içerik kaynağı

> Bu dosya `Proje_Oneri_Formu.docx` dosyasının içerik kaynağıdır. Word dosyası `python docs/oneri_formu/build_docx.py` ile bu dosyadan üretilir (şablon: `sablon.docx`, Arial 9 pt).
> Sayıların kaynakları: `sayi_kaynaklari.md`. Henüz okunmamış kaynaklara yapılan atıflar: `okunacak_atiflar.md`.
> Biçim kuralı: `## ` bölüm başlığı, `### ` alt başlık, `| ... |` tablo, `- ` madde. `<!-- -->` yorumları Word'e aktarılmaz.

## Proje Başlığı

Derin Öğrenme ile Hane ve Şebeke Düzeyinde 24 Saatlik Elektrik Talep Tahmini ve Tahmin Artığı Tabanlı Tüketim Anomalisi Tespiti

## Proje Ekibi

| Öğrenci Adı/Soyadı | Öğrenci Numarası | Projedeki Görevi |
|---|---|---|
| Dilara YILDIRIM | 230290018 | Veri hattı (yükleme, temizleme, saatliğe indirgeme, bölme, pencereleme), talep tahmini modelleri ve referansları, demo uygulaması |
| Şeyda ALTIN | 230290046 | Keşifsel veri analizi, literatür taraması, anomali tespiti (enjeksiyon protokolü, LSTM-Autoencoder, artık tabanlı dedektör, referanslar), şebeke düzeyi deney, görselleştirme |

## Projenin Özgün Değeri

### Konunun önemi

Elektrik sistemlerinde arz ile talebin her an dengelenmesi gerektiğinden, kısa vadeli yük tahmini planlama ve işletme kararlarının temel girdilerinden biridir [1]. Akıllı sayaçların yaygınlaşmasıyla tahmin, trafo merkezi veya şebeke toplamından tek tek hanelere kadar inmiştir. Ancak tek bir hanenin tüketimi, cihazların açılıp kapanması ve bireysel alışkanlıklar nedeniyle çok oynaktır ve bu oynaklık hane düzeyinde tahmini toplulaştırılmış yüke göre belirgin biçimde zorlaştırır [2]. Aynı veri, olağandışı tüketimin (unutulan bir cihaz, arıza, kaçak kullanım, beklenmedik kesinti) erken fark edilmesi için de kullanılabilir; binalarda yapay zekâ tabanlı tüketim anomalisi tespiti, enerji verimliliği ve maliyet açısından giderek önem kazanan bir alandır [3].

### Literatür ve eleştirel boşluk

Kapı mekanizmalı tekrarlayan sinir ağları, yani uzun kısa süreli bellek (long short-term memory, LSTM) [4] ve kapılı tekrarlayan birim (gated recurrent unit, GRU) [5], uzun zamansal bağımlılıkları öğrenebildikleri için yük tahmininde sıkça kullanılmaktadır. Konut düzeyinde LSTM tabanlı tahminin rakip yöntemlerden daha iyi sonuç verdiği [2], LSTM tabanlı diziden diziye (sequence-to-sequence, seq2seq) yapının saatlik ve dakikalık çözünürlükte çalışabildiği [6] ve bu projede kullanılan UCI hane veri setinde evrişimli ve tekrarlayan katmanları birleştiren bir modelin kıyaslanan yöntemlerden daha düşük hata verdiği [7] raporlanmıştır. Anomali tespitinde ise istatistiksel yöntemlerden [8] ağaç tabanlı Isolation Forest'a [9], LSTM kodlayıcı–kod çözücü ile yeniden yapılandırma hatasına dayalı yaklaşımlardan [10] tahmin hatasına ve dinamik eşiğe dayalı yaklaşımlara [11] kadar geniş bir yöntem yelpazesi bulunmaktadır.

İncelenen çalışmalarda üç eksiklik dikkat çekmektedir. Birincisi, tahmin [2], [6], [7] ve anomali tespiti [10], [11] çoğunlukla ayrı problemler olarak ele alınmakta; iyi bir tahmin modelinin artıklarının hane tüketiminde anomali sinyali olarak ne kadar işe yaradığı aynı veri hattında sınanmamaktadır. İkincisi, hane tüketim verilerinde gerçek anomali etiketi çoğunlukla yoktur; kullandığımız UCI veri setinde de böyle bir etiket bulunmamaktadır [12]. Etiket olmadığında yöntemler ancak nitel olarak incelenebilir ya da tipi ve büyüklüğü bilinen, kontrollü sentetik anomalilerle nicel olarak karşılaştırılabilir. Üçüncüsü, hane düzeyinde elde edilen sonuçların toplulaştırılmış şebeke yüküne ne ölçüde taşındığı, incelediğimiz çalışmalarda aynı model ve ayarlarla doğrudan karşılaştırılmamaktadır.

### Projenin katkısı

Bu proje, hane düzeyinde kısa vadeli talep tahmini ve tüketim anomalisi tespiti problemlerini tek bir sızıntısız veri hattında birleştiren ve sonuçlarını iki toplulaşma düzeyinde sınayan bir yaklaşım sunmaktadır. Katkılar şunlardır:

- (i) Aynı saatlik veri hattında 24 saatlik talep tahmini ile anomali tespiti birlikte ele alınmakta; final tahmin modelinin bir adım ilerideki hatası, saat ve gün bazında ölçeklenerek anomali skoruna dönüştürülmektedir.
- (ii) Tahmin artığı tabanlı tespit, yeniden yapılandırma tabanlı LSTM-Autoencoder, hareketli z-skoru ve Isolation Forest ile aynı koşullarda karşılaştırılmaktadır.
- (iii) Gerçek etiket eksikliği, tipleri (ani tepe, düşüş, seviye kayması, desen) ve büyüklükleri tanımlı, tekrarlanabilir sentetik anomali enjeksiyonu ile giderilmekte; yöntemler iki zorluk senaryosunda, çok sayıda enjeksiyon tohumu (seed) üzerinden ve olay düzeyindeki metriklerle ölçülmektedir.
- (iv) Kronolojik bölme, yalnızca eğitim kümesine uydurulan ölçekleyiciler, doğrulama kümesinde yapılan tüm seçimler ve bir kez kullanılan test kümesi ile veri sızıntısına kapalı bir değerlendirme protokolü uygulanmaktadır.
- (v) Hane düzeyinde seçilen yöntemler, hiçbir yeni ayar yapılmadan ElectricityLoadDiagrams20112014 veri setindeki 160 müşterinin toplam yüküne [13] uygulanmakta ve toplulaştırmanın tahmin ile anomali tespitine etkisi ölçülmektedir.

### Araştırma soruları ve hipotezler

- **AS1:** Kapı mekanizmalı tekrarlayan ağlar (LSTM, GRU), hane düzeyi saatlik tüketimin 24 saatlik tahmininde mevsimsel naif ve klasik regresyon yöntemlerinden daha düşük hata üretir mi? **Hp1:** LSTM ve GRU, test RMSE'sinde haftalık mevsimsel naif yöntemden (168 saat) ve Ridge regresyondan daha düşük hata verir.
- **AS2:** LSTM ile GRU arasında doğruluk ve eğitim maliyeti açısından anlamlı bir fark var mıdır? **Hp2:** GRU, LSTM'ye benzer doğruluğa daha az parametre ve daha kısa eğitim süresiyle ulaşır.
- **AS3:** Derin öğrenme tabanlı anomali yöntemleri, istatistiksel ve klasik makine öğrenmesi yöntemlerinden daha başarılı mıdır? **Hp3:** LSTM-Autoencoder ve tahmin artığı dedektörü, özellikle seviye kayması ve desen anomalilerinde hareketli z-skoru ve Isolation Forest'tan daha yüksek recall sağlar.
- **AS4:** Hane düzeyinde seçilen tahmin ve anomali yöntemleri, toplulaştırılmış şebeke yükünde nasıl davranır? **Hp4:** Toplulaştırma bireysel dalgalanmaları azalttığı için aynı yöntemler şebeke düzeyinde, ölçekten bağımsız metriklerde (MAPE, R², olay F1) hane düzeyine göre daha yüksek başarı gösterir.
