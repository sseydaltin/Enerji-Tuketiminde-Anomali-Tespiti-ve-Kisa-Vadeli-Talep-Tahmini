# Savunmada sorulabilecekler — derlenmiş liste

Bu dosya, proje boyunca (2026-10-28 – 2026-11-26) her görevin sonunda Claude Code'un verdiği "savunmada sorulabilecekler" listelerinin **oturum kayıtlarından olduğu gibi** derlenmiş hâlidir. Maddeler değiştirilmedi; yalnız tekrarlar çıkarıldı ve konu başlıklarına (anahtar kelimelerle) ayrıldı. Tarih, listenin verildiği gündür. Maddelerdeki sayılar o günün sonuçlarıdır; final sayılar için `docs/final_sonuclar.md`, `docs/hane_vs_sebeke.md`, `docs/lead_sonuclar.md` esas alınır (ör. LEAD'de final model sonradan L1 behavior oldu).

## Hane

### 2026-10-30
- sMAPE neden %89 çıkıyordu? Metrik z-skoru üzerinden hesaplanıyordu. sMAPE ölçeğe bağımlı bir metrik: sıfıra yakın ve negatif değerlerde paydası küçülüp şişiyor.
- Bu hata early stopping'in seçtiği epoch'u değiştirdi mi? Değiştirmedi. z cinsinden MAE kW cinsinden MAE ile orantılı, bu yüzden aynı epoch seçilirdi. Ama raporlanan sayıların birimi yanlıştı ve eğitim hedefi çift ölçeklenmişti; bu yüzden yeniden eğitim yaptık.
- Hatayı yakalayacak bir test neden yoktu? Mevcut test yalnızca şekilleri ve scaler'ın nereye fit edildiğini kontrol ediyordu, `y`'nin hangi ölçekte olduğunu kontrol etmiyordu. Yeni test `y_val`'i ham değerlerle karşılaştırıyor.
- Saat bazında neden "hedef saat"e göre grupladınız? 24 ufuk adımının her biri günün farklı bir saatine denk geliyor. Hatanın günün hangi saatinde yoğunlaştığını görmek için ilk hedef saatine ufuk adımını ekledik (mod 24).
- R² neden hatadan etkilenmedi? R² afin dönüşüm altında değişmez. Bu yüzden eski R² değerleri (0,44–0,45) yeni değerlere yakın çıktı.

### 2026-10-30
- **Tüm modellerin aynı koşullarda değerlendirildiğini nasıl kanıtladınız?** Üç ayrı pencere üretim hattını ve kayıtlı tüm tahmin dosyalarını aynı referansla karşılaştırdık: hedef zamanları ve kW değerleri birebir aynı. Bunu gerçek veride `verify_forecast_eval`, sentetik veride bir pytest gösteriyor.
- **Ridge'de alpha neden işe yaramadı?** Yaklaşık 24 bin örnek ve 26 standartlaştırılmış öznitelik var; örnek sayısı öznitelik sayısının çok üstünde. Bu durumda 0,1–100 aralığındaki ceza katsayısı ağırlıkları neredeyse hiç küçültmüyor. Ridge'in zayıflığı takvim bilgisi olmamasından geliyor.
- **MLP neden LSTM'den kötü, sklearn MLP'den neden iyi?** PyTorch MLP aynı pencereyi görüyor ama zaman sırasını modellemiyor; LSTM ile arasındaki fark yaklaşık 0,015 kW. sklearn MLP'ye göre iyileşmenin kaynakları: B öznitelikleri, val ile early stopping, scheduler ve yakınsamaya yetecek epoch.
- **Modeller neden tepe tüketimi düşük tahmin ediyor?** MSE kaybı belirsizlik altında koşullu ortalamaya yöneltir. Seyrek ve ani tepeler ortalamaya çekiliyor; sonuç düşük düzeyde fazla, yüksek düzeyde az tahmin.
- **Ridge'in hatası neden h=1'de düşük, ortada yüksek?** h=1'de son gözlem çok bilgi taşıyor. Ufuk uzadıkça Ridge'in günün saatini bilmesinin tek yolu gecikmeler; takvim özniteliği yok. LSTM/GRU'da takvim öznitelikleri olduğu için hata ufukla daha yavaş artıyor.

### 2026-10-30
- **Shuffle genelde iyi bir uygulama; burada neden kötüleştirdi?** Karıştırma her batch'i tüm döneme yayıyor ve model genel örüntüyü çok hızlı öğreniyor (en iyi epoch 2–5). Sonrasında val'e göre aşırı uyuyor. Sıralı batch'ler ise örtük bir düzenlileştirme gibi davranmış olabilir. Tek seed ile sonuç kesin değil, ama üç modelde de yön aynı.
- **Artık hedef neden işe yaramadı; hata olmadığını nasıl gösterdiniz?** Tanılama yaptık: tahminler SN168'e yapışık, eğitim kaybı düşmüyor. MLP aynı kodla öğreniyor; taban değer çıktıya verilince LSTM/GRU da öğreniyor. Sorun kodda değil, son gizli durumun bir hafta önceki tek tek değerleri taşıyamamasında.
- **"Bilinen gelecek öznitelikleri" sızıntı değil mi?** Hayır. Takvim önceden bilinir. `y[t+h−168]` için h ≤ 24 < 168 olduğundan her değer tahmin başlangıcından en az 144 saat önce gözlenmiştir. Bunu her pencere ve her ufuk için bir test kontrol ediyor.
- **Ridge + takvim karşılaştırması neden adil?** Ridge'e son gözlenen saatin takvimini verdik; LSTM'in B setinde gördüğü bilgiyle aynı. Hedef saatlerin takvimini verseydik Ridge'e LSTM'de olmayan açık bir bilgi vermiş olurduk.
- **Bu kadar konfigürasyon aynı val kümesinde denendi; val skoru güvenilir mi?** Seçilen modelin val skoru iyimser olabilir. Bu yüzden denenen her konfigürasyonu `experiments.csv`'de tutuyoruz. Aşama 4'te seçimi train içi çapraz doğrulamaya taşıyoruz, test kümesi ise tek bir final değerlendirme için duruyor.

### 2026-10-31
- **Optuna neden val'de değil de train içinde arama yaptı?** Aynı val kümesinde 40 deneme yapılsaydı seçilen ayarın val skoru şişerdi. Train'in son 12 ayını 4'er aylık üç ardışık kata böldük; her katta yalnızca o katın öncesindeki veriyle eğittik. Val'i yalnızca seçilen ayarı bir kez doğrulamak için kullandık.
- **Neden 3 seed? Tek seed'deki farklara neden güvenmiyorsunuz?** Seq2seq'in seed std'si 0,006 kW. Bundan küçük farklar (lookback, LSTM–GRU, C seti) gürültüden ayırt edilemiyor. Yalnızca std'nin birkaç katı olan farkları gerçek sayıyoruz.
- **Topluluk gerçekten daha iyi mi?** Ortalamada yaklaşık 0,006 kW iyi ve daha kararlı. Ama ağırlıklar val'de seçildiği için bu iyimser; yansız ölçüm ancak test'te yapılabilir. XGBoost'un ağırlığı 0 çıktı.
- **En basit anomali yöntemi neden kazandı?** A0 her saati önceki 7 günün aynı saatiyle kıyaslıyor; bu, günlük desene dayalı anomaliler (örneğin gece tüketiminin gündüz seviyesine çıkması, recall 0,60) için doğru karşılaştırma. LSTM-AE ise oynak hane verisinde normal saatleri de kötü geri kuruyor, bu yüzden anomaliler öne çıkmıyor. Gizil boyut tanılaması "AE girdiyi kopyalıyor" açıklamasını dışladı.
- **Eşiği neden test'te seçmiyorsunuz? Yüzdelik eşik neden kötü?** Eşik bir hiperparametredir; test'te seçmek sızıntı olur, bu yüzden val'de seçilip test'e aynen uygulanıyor. Yüzdelik eşik etiket kullanmaz; anomali oranı (%2,7) 95. yüzdeliğin işaretlediği %5 ile örtüşmediği için precision ve recall dengesizleşiyor.

### 2026-10-31
- **Eşiği neden her seed'de ayrı değil, birleşik veride seçtiniz?** Her yöntemin test'e tek bir eşiği gidecek. Kopya başına seçmek, o kopyaya göre ayarlanmış iyimser bir skor verirdi; birleşik seçim tek eşiğin farklı olay yerleşimlerindeki performansını ölçüyor.
- **Olay bazlı recall ile point-adjust arasındaki fark ne?** Biz "en az bir saati işaretlenen olay yakalandı" kuralını yalnız olay sayımında kullanıyoruz. Point-adjust ise yakalanan olayın bütün saatlerini doğru sayıp nokta F1'ini şişiriyor; rastgele skorlarla bile yüksek F1 verebildiği için nokta metriklerine uygulamadık.
- **A1'in PR-AUC'si daha yüksekken neden A0'ı daha iyi buluyorsunuz?** PR-AUC farkı standart sapmanın içinde (0,123 ± 0,092'ye karşı 0,110 ± 0,036), yani kesin bir sıralama yok. Kullanılacak eşikte ise A0'ın F1'i ve olay recall'ü belirgin şekilde yüksek.
- **Tip bazlı sonuçlara ne kadar güvenilir?** Az güvenilir: seed başına tür başına 2–8 olay var. Bu yüzden ortalama ± std veriyor ve olay sayılarını açıkça yazıyoruz.
- **Seed 1'deki A0 sonucu neden düştü?** Tek bir enjeksiyon kopyası şansa bağlı: seed 1'de A0'ın iyi yakaladığı desen olayları fazlaydı. 5 kopya bu şansı ortalayarak daha dürüst bir tahmin veriyor.

### 2026-10-31
- **Normalize skor neden bu kadar fark yarattı?** Tahmin hatası saate göre çok değişiyor: train'de std 0,17 kW (gece) ile 0,93 kW (akşam) arasında. Ham hata akşamları sürekli yüksek kaldığı için yanlış alarm veriyordu. Bölme, her saati kendi olağan hatasıyla kıyaslıyor; std yalnız train artıklarından hesaplanıyor.
- **Anomaliyi girdi penceresinde bırakmak sonucu nasıl etkiliyor?** Gerçek kullanımda da olan budur. Bedeli seviye kaymasında görülüyor: model kaymış düzeyi "normal" saymaya başlıyor, bu yüzden olayın başını yakalıyor ama devamını kaçırıyor.
- **Yumuşatma gelecek bilgisi kullanıyor mu?** Hayır. Ortalama yalnız geçmişe bakıyor (t−w+1'den t'ye); bunu bir test kontrol ediyor. Bedeli, alarmın birkaç saat gecikebilmesi. Eşitlikte küçük pencereyi seçmemizin nedeni de bu.
- **A3, A0'dan gerçekten iyi mi?** Aynı seed'ler üzerinden eşleştirilmiş farkta 5 seed'in 4'ünde önde, ama fark küçük ve kesin değil. Eşikten bağımsız ROC-AUC'de A0 hâlâ önde (0,755'e karşı 0,712). Kazanç precision'dan geliyor.
- **1 adım tahmin için neden F13-ens değil de F10-opt ana model?** Kullanıcı kararıyla seq2seq ana model, topluluk ikinci varyant oldu. Sonuçlar ikisinin arasında anlamlı fark olmadığını gösterdi: LSTM'in 0,2 ağırlığı 1 adım hatasını pek değiştirmiyor.

### 2026-10-31
- **Çok ufuklu skor seviye kaymasında neden beklenen etkiyi vermedi?** Her ufuk kendi train std'siyle normalize ediliyor ve uzun ufukta std büyük. ×1,5'lik kayma bu yüzden normalize skorda küçük kalıyor. Olay recall'ü arttı, ama saat bazlı recall artmadı. Bu açıklama bir hipotez; ayrıca test etmedim.
- **Birleşimde neden rank ve z ayrı denendi?** A0 (|z|) ile A3 (std katı) farklı ölçekte. z ölçeği büyüklüğü korur, rank aykırı değerlere dayanıklıdır ama 1'de doyar. Doyma, rank ortalamasının spike'ları kaçırmasının nedeni. Bu yüzden en yüksek F1 tek başına seçim için yeterli değil, tip bazlı tablo da gerekli.
- **Ölçek referansı neden enjeksiyonsuz val?** Referans etiket ve enjeksiyon içermiyor ve yalnız val'den geliyor. Test'te aynı referans aynen uygulanıyor, yani test'te hiçbir şey fit edilmiyor.
- **"Gürültü düzeyi" ölçütü ne kadar güvenilir?** 5 seed'de kaba bir ölçüt; istatistiksel güç düşük. Daha sağlam bir alternatif, daha fazla seed ya da bootstrap güven aralığı olurdu. Bu yüzden belirsizlikte basit yöntemi seçtik.
- **Final yöntem tahmin modeliyle neden aynı değil (F13-ens yerine F10-opt)?** A3, 1 adım ileri tahmine dayanıyor. Topluluk tek modele göre anlamlı fark yaratmadı (−0,006 F1), ama ikinci bir modelin bakımını gerektiriyor. Tahmin tarafındaki final model F13-ens olarak kalıyor.

### 2026-10-31
- **OOF std neden in-sample'dan yalnız %2–3 büyük çıktı?** F10-opt az epoch eğitildi (en iyi epoch 3–8) ve dropout yüksek; train'i ezberlemiyor. Bu yüzden in-sample artıkları zaten gerçekçi. Asıl iyimserlik, seçimin aynı val'de yapılmasından geliyordu.
- **Val-hold neden gerekliydi?** Val-dev'de A0 ile A3 arasındaki fark küçük görünüyordu (+0,04 ile +0,11). Val-hold'da fark +0,17 ile +0,27'ye çıktı ve A0'ın eşiğinin başka bir döneme taşınmadığı ortaya çıktı. Seçim ile ölçüm aynı veride yapılsaydı bunu göremezdik.
- **σ tabanlı senaryo neden "ana" senaryo?** Anomaliyi "olağan oynaklığın dışında" diye tanımlar. Zor senaryodaki ×1,5'lik kayma ya da gece yarısındaki ×2'lik spike, bu hanenin normal değişkenliği içinde kalabiliyor. Ana senaryoda bile drop'ların yakalanamaması, tanımın drop için hâlâ zayıf olduğunu gösteriyor.
- **Eşik neden olay F1'e göre seçildi?** Operatör için soru "kaç olayı kaç alarmla yakaladık?" sorusudur. Bunun bedeli, olay precision tanımının cömert olması: tek bir saatlik örtüşme yetiyor.
- **Gürültü kuralı neden tartışmalı?** Std tabanlı ölçüt seed sayısı arttıkça sıkılaşıyor (yaklaşık t ≤ √n). Standart hata tabanlı bir ölçüt (std/√n) daha doğru olurdu. Kural sonuçlar görülmeden konduğu için değiştirmedim.

### 2026-10-31
- **Test MAE neden val'den düşük, buna "iyileşme" denebilir mi?** Hayır. Test yaz dönemi; tüketim ve oynaklık daha düşük, bu yüzden mutlak hatalar küçülüyor. Doğru karşılaştırma aynı dönemdeki referansa göre: SN168'e göre kazanç val'de +25,6 %, testte +26,0 %.
- **Final anomali yöntemi testte A0'ı neden geçemedi?** Val'de seçilen eşik ve yıllık std yazın kalibrasyonunu kaybetti. Sıralama gücü korunuyor (PR-AUC 5/5 seed üstün), kayıp eşikte. Mevsime duyarlı normalizasyon bir sonraki adım olurdu.
- **Testten sonra neden hiçbir şeyi değiştirmediniz?** Değiştirseydik test de bir seçim kümesine dönüşür ve bağımsız ölçüm kalmazdı. Kodun testten önce sabitlendiği commit geçmişinde görülüyor.
- **Val-hold neden "tam bağımsız değil"?** Gürültü kuralı val-hold sonuçları görüldükten sonra revize edildi. Bu yüzden bağımsız hüküm testte veriliyor.
- **"En iyi tahmin modeli" hangisi?** Metriğe bağlı. MAE'de F13-ens; RMSE, R² ve günlük toplamda XGBoost. Kullanım amacına göre seçim yapılmalı: saatlik planlamada MAE, tepe yük için RMSE.

### 2026-10-31
- **Yaz saati düzeltmesini neden toplam yükte, 15 dakikalık düzeyde yaptık?** Sorun dört çeyrekte; saatlik toplama hangi zaman damgası kuralıyla yapılırsa yapılsın, önce kaynakta düzeltmek kuraldan bağımsız sonuç verir. Ayrıca beklenen desen görülmezse kod hata veriyor, yani düzeltme sessizce yapılmıyor.
- **2012 öncesi aktif müşteri kuralı neden gerekli?** Sonradan başlayan müşteriler başlangıç öncesinde sıfırla doldurulmuş. Onları toplama katmak toplam yükte gerçek olmayan bir artış eğilimi yaratır ve model bu eğilimi öğrenir.
- **Hiperparametreleri neden ELD'de yeniden aramadık?** Karşılaştırmanın sorusu "aynı yöntem toplulaştırılmış veride ne yapıyor?". Yeniden arama bu soruyu bozar. Bedeli F5-opt'un 60 epoch'ta yakınsamaması.
- **Hane ile şebekeyi neden MAE ile değil MAPE, R² ve SN168 kazancıyla karşılaştırıyoruz?** MAE kW cinsinden ve iki veri arasında ölçek ~100.000 kat farklı. Oranlar ve aynı dönemdeki naif referansa göre kazanç ölçekten bağımsız.

### 2026-10-31
- **H3 neden karşılanmadı, bunu neden gizlemediniz?** Eşik ve normalizasyon kışa denk gelen val döneminde ayarlandı, test ise yaz döneminde. Testten sonra bir şey değiştirmek veri sızıntısı olurdu. Eşikten bağımsız PR-AUC'de A3 yine A0'dan iyi (0,241'e karşı 0,148).
- **Final model neden XGBoost değil?** Seçim test öncesinde val MAE ile yapıldı ve F13-ens orada birinciydi. XGBoost'un testte RMSE'de öne geçmesi seçim kuralını değiştirmez; bu yüzden "en iyi model" iddiası metriğe bağlı.
- **Hane ve şebeke karşılaştırmasında neden MAE yok?** İki serinin ölçeği yaklaşık 100.000 kat farklı. Bu yüzden ölçekten bağımsız ölçütler kullanıldı: MAPE*, R² ve SN168'e göre kazanç.
- **README'deki bir sayının kaynağı nerede?** Her sayı `results/tables/` veya `results/eld/tables/` altındaki CSV dosyalarından geliyor. Bölümün başında bu dosyalar ve ayrıntılı belgeler gösteriliyor.

### 2026-11-03
- **Demo neden eğitim yapmıyor?** Kayıtlı ağırlıkları ve sonuç tablolarını yüklüyor. Sunumda eğitim gerekmiyor, gösterilen sayılar da `results/` altındaki dosyalardan geliyor.
- **Demo test kümesini kullanıyor, bu kuralı bozmuyor mu?** Eşikler ve model seçimi testten yapılmıyor; kayıtlı doğrulama eşikleri okunuyor. Yani demo testi yalnızca görüntülüyor, hiçbir seçimi ona göre yapmıyor.
- **Anomali enjeksiyonu orijinal veriyi değiştiriyor mu?** Hayır, kopya üzerinde çalışıyor. Bu dal `inject_anomalies` fonksiyonuna `anomaly_types` parametresini ekliyor; bu parametre verilmezse tüm tipler kullanılıyor.
- **Neden `streamlit run` yerine `python -m streamlit run`?** `app/streamlit_app.py` içindeki `from app.pages import ...` satırı proje kökünün Python yolunda olmasını gerektiriyor, `python -m` de çalıştırılan klasörü yola ekliyor. Alternatifi paketi kurmak veya `PYTHONPATH` ayarlamak olurdu.
- **Household ve ELD'nin MAE değerleri neden doğrudan kıyaslanmıyor?** Ölçekleri çok farklı: hane yaklaşık 1 kW, ELD yaklaşık 106 MW. Bu yüzden MAPE*, R² ve her veri kümesinin kendi SN168 baseline'ına göre MAE kazancı kullanılıyor.

### 2026-11-03
- **Kolay/zor gün test hatasına göre seçiliyor, bu sızıntı değil mi?** Hayır. Bu sıralama yalnızca hangi günün gösterileceğini belirliyor; model ve eşik seçimi doğrulama kümesinde yapılıp donduruldu.
- **Neden bir yöntem testte daha yüksek F1 alsa da final değil?** Birleşik skorun hane F1 değeri (0,463) final yöntemden (0,384) yüksek. Ama final seçim doğrulamada yapıldı; testten sonra değiştirmek test kümesini seçim aracına çevirirdi.
- **Hane MAPE'si neden %49, Şebeke'ninki neden %3?** Tek evin tüketimi düzensiz ve küçük değerlerde küçük hatalar büyük yüzdelere dönüşür. 160 müşterinin toplamında bireysel dalgalanmalar birbirini dengeler.
- **"Yanlış alarm" gerçekten yanlış mı?** Kesin değil. Bu sayı yalnızca enjekte edilen saatlerle kesişmeyen alarm bloklarını sayar; test verisinde gerçek bir anormallik de olabilir, bu yüzden üst sınır gibi okunmalı.
- **`.copy()` neden gerekliydi?** `test.loc[...]` bir dilim döndürür ve pandas bu dilime yazmanın asıl tabloyu değiştirip değiştirmeyeceğini garanti etmez. `.copy()` bu belirsizliği kaldırır; `runs` ayrı bir kopya olur, sonuç değişmez.

### 2026-11-03
- **A6 neden testte denenmedi?** Fikir test sonucunu gördükten sonra doğdu. Testte seçilir veya ölçülürse test bağımsızlığını yitirir.
- **Taban kuralı veri sızıntısı yapıyor mu?** Hayır. Eşik yalnız train'den geliyor ve kural yalnız geçmişe bakıyor, alarm k. saatte başlıyor. Bunu test de doğruluyor.
- **Yanlış alarm neden hiç artmadı?** Bu kış ağırlıklı dönemlerde tüketim, enjeksiyon dışında 2 saat boyunca tabanın altında kalmamış olabilir. Yazın ev boş kalırsa bu değişebilir; sonuç genellenemez.
- **Günlük toplam hata neden tek başına yetmez?** Saatlik hatalar toplamda birbirini götürebilir, bu yüzden MAE ile birlikte okunmalı.
- **Val-dev'de neden A6-dual A3'ten iyi değil de val-hold'da iyi?** Fark seed oynaklığıyla aynı büyüklükte, yani gerçek bir iyileşme kanıtı sayılmaz.

### 2026-11-03
- **Neden iki seçenek var, en iyisi gösterilmiyor?** Tek varyant iki sorunu birden çözmüyor; biri kısa sıçramaları, öteki düşüşleri ve olay F1'i iyileştiriyor. İkisini göstermek bu ödünleşimi saklamıyor.
- **Yeni seçenek eklemek yeni deney sayılmaz mı?** Hayır. Ayar seçilmedi, dosya yazılmadı; demo val-dev'de seçilmiş kayıtlı değerleri okuyor.
- **Demonun deneydeki kuralla aynı olduğunu nereden biliyoruz?** Test, karışık enjeksiyonla her seed'de kayıtlı olay F1 değerini `1e-6` toleransla üretiyor.
- **İki seçenekte T2 neden farklı?** T2 her varyant için ayrı seçildi. Taban kuralı düşüşleri zaten yakaladığı için en iyi F1 daha yüksek T2'de çıktı, bu yüzden spike kazancı neredeyse kayboldu.
- **`experiments.csv` değiştiği halde final sonuçlar neden değişmiş sayılmıyor?** Yalnız iki satır eklendi ve eski içerik bayt bayt korundu; bu satırların test sütunları da boş.

### 2026-11-24
- **Çakışmayı neden `git merge-tree` ile denedin?** Hiçbir dala ya da çalışma klasörüne yazmadan, birleştirmenin sonucunu ve çakışan dosyaları bellekte hesaplıyor. Geçici bir dalda deneme birleştirmesi de yapılabilirdi, ama sonradan geri alma riski taşıyordu.
- **A7 sonuçları neden "final" değil?** Deney, test sonucu görüldükten sonra tasarlandı. Bu yüzden yalnız val'de değerlendirildi ve final yöntemi (A3-s2s-norm) değiştirmiyor. Testte çalıştırılsaydı test kümesi seçime karışmış olurdu.
- **A7 − A2 farkı neden anlamlı sayılmıyor?** Eşleştirilmiş farkın ortalaması (+0,122), gürültü sınırının (2·std/√10 ≈ 0,150) altında kalıyor.
- **#8 neden hâlâ gerekli?** Sonraki tüm Dilara dalları onun commit'lerinin üstüne kurulu. Kapatılırsa bu zincir, `main`'e tek ve çok büyük bir PR ile (165 dosya) girmek zorunda kalır.
- **Literatürdeki R² = 0,98 ile bizim 0,297'miz neden doğrudan kıyaslanamaz?** Ufuk farklı (1 adım ve 24 saat), ölçek farklı (normalize ve kW), bölme yöntemi farklı (k-katlı CV ve kronolojik), ve o çalışmalarda naif referans genellikle yok.

## Şebeke (ELD)

### 2026-10-31
- **Hane ve şebeke neden MAE ile karşılaştırılmıyor?** İki seri arasında ölçek farkı ~100.000 kat. MAPE, R² ve aynı dönemdeki SN168'e göre kazanç ölçekten bağımsız ölçütler.
- **Şebekede neden XGBoost kazandı?** XGBoost 336 saatlik gecikmeyle iki haftalık deseni doğrudan görüyor ve ayar gerektirmeden bu veriye uyuyor. Derin modeller ise hane verisinde ayarlandı; LSTM'in yakınsamaması da bunu gösteriyor. Bu, "derin modeller şebekede kötü" demek değil, "aktarılan ayarlar kötü" demek.
- **Noel neden bu kadar büyük hata veriyor?** `is_holiday` yalnız 0/1 bilgisi taşıyor; Noel'in sıradan bir tatilden çok daha derin bir düşüş yarattığını söylemiyor. Train'de de yalnız iki Noel var. Tatil türünü ayırt eden bir öznitelik ya da "geçen yılın aynı tatili" gecikmesi bunu azaltabilir; denemedim.
- **Şebekede anomali sonuçları neden bu kadar yüksek?** Düzgün bir seride σ tabanlı sapmalar çok belirgin kalıyor. Mutlak tanımlı drop ise şebeke ölçeğinde gerçekçi değil ve skoru şişiriyor. Daha adil bir karşılaştırma için drop'u da σ'ya göre tanımlamak gerekirdi.

## LEAD

### 2026-11-25
- *L4-semi neden L4'ten bu kadar iyi?* Etiketli anomali saatleri eğitimde kalınca tahmin modeli donmuş serileri de tahmin etmeyi öğreniyor; tahmin hatası küçülüyor ve anomali skoru düşüyor. Çıkarınca bu olmuyor, ama bu sefer etiket kullanılmış oluyor.
- *Etiketsiz yöntem neden `other` diliminde de iyileşmedi?* `other` anomalileri tahmin hatasıyla zaten zor ayrılıyor. İyileşme neredeyse tamamen frozen ve gap_edge dilimlerinden geliyor.
- *GRU neden LightGBM'den kötü?* Seçili öznitelikler (gecikmeler, sabit kalma süresi) ağaç modeline sinyali hazır veriyor; GRU aynı şeyi 48 saatlik pencereden kendisi öğrenmek zorunda. Optuna denemesi bütçe yüzünden 20'den 6'ya indi; bu da GRU'nun aleyhine olabilir.
- *Seed'ler arasındaki büyük fark (0,41–0,61) ne anlama geliyor?* Val kümesinde yalnız 40 bina var; hangi binaların düştüğü sonucu çok değiştiriyor. Bu yüzden seçim kuralı ortalamayı 2·std/√n gürültü payıyla karşılaştırıyor.
- *Erken durdurma neden bazen 3. epoch'ta oldu?* Val PR-AUC dalgalı; patience 8 olduğu için model en iyi epoch'tan sonra 8 epoch daha eğitildi, sonunda en iyi ağırlıklar geri yüklendi.

### 2026-11-25
- *Neden her seed kendi setini seçiyor?* Seed 42'nin train binalarının bir kısmı, seed 7'nin val binaları. Ortak bir seçim, o binaların etiketlerini dolaylı olarak val değerlendirmesine taşırdı.
- *G7 için neden OOF gerekiyor?* L4-semi bir binanın kendi serisiyle eğitilirse o binadaki tahmin hatası yapay olarak küçük çıkar. LightGBM eğitimde bu küçük hataları görür, val'de ise gerçek (daha büyük) hatalarla karşılaşır. Yani dağılım kayar ve eğitimde görülen sonuç olduğundan iyi görünür.
- *Nedensellik testi ne kanıtlıyor?* t'den sonraki okumalar değiştirilip yeni eksik saatler eklendiğinde, ≤ t'deki bütün özniteliklerin bire bir aynı kalması gerekiyor. Merkezli pencere, `bfill` ya da genişleyen istatistikte bir kayma hatası bu testi bozar.
- *Gürültü kuralı neden 2·std/√k?* Bu, kat başına eşleştirilmiş farkın ortalamasının yaklaşık %95 güven sınırı. k = 3 katta bile aynı katlar karşılaştırıldığı için, katlar arasındaki zorluk farkı sonuçtan düşülmüş oluyor.
- *G4 profili neden "genişleyen"?* Hareketli pencere kullanılsaydı ilk haftalarda profil hiç olmazdı. Genişleyen ortalama, o ana kadarki bütün geçmiş haftaları kullanıyor ve hiçbir zaman gelecek haftaları görmüyor.

### 2026-11-25
- *Full'de val PR-AUC +0,039 arttı; bu, davranışın daha iyi öğrenildiği anlamına mı geliyor?* Hayır, büyük kısmı değil. Kazanç esas olarak G5 ve frozen dilimindeki iyileşmeden geliyor. `other` dilimi yalnız 0,03'ten 0,05'e çıktı, behavior varyantında ise hiç kazanç yok.
- *G7 seed 42'de neden çıktı?* Ortalama katkı +0,034 ama 3 kat arasındaki fark çok büyük (2·std/√3 = 0,048). Kural "katkı gürültüden ayırt edilemiyor" diyor. Daha fazla kat bu kararı değiştirebilirdi; bu, 3 kat kararının bir sonucu.
- *Seed'ler farklı setler seçti; final setiniz hangisi?* Her seed'in val sonucu yalnız kendi train binalarıyla seçilmiş setten geliyor; böylece sızıntı yok. Raporlanan şey tek bir öznitelik listesi değil, "seçim prosedürü"nün performansı.
- *SHAP'in `other` dilimindeki tablosu ne söylüyor?* `profile_z` ve `l4semi_score`, yani "bu binanın bu hafta-saatinde olağandan sapma" ve "tahmin hatası", davranışsal anomalinin en iyi sinyalleri. Bu, ana bilimsel sorumuz için olumlu bir bulgu.
- *CV skorları neden val'den yüksek?* CV'de erken durdurma, değerlendirilen katın kendisinde yapılıyor; bu iyimser gösteriyor. Ayrıca val binaları farklı binalar ve binalar arası fark çok büyük. Bu iyimserlik bütün grupları eşit etkilediği için grup karşılaştırmasını bozmuyor, ama mutlak sayılar raporda val'den verilir.

### 2026-11-25
- *Neden "en iyi" model istatistiksel olarak seçilemiyor?* Seed başına 40 val binası var ve `other` saatleri birkaç binada yoğunlaşıyor. Bu yüzden seed'ler arası fark büyük, 2·std/√3 sınırı da geniş.
- *L1-I neden L1-FE'den biraz düşük?* Yapılandırmaları aynı; tek fark val'de erken durdurma. Val hem durdurmada hem ölçümde kullanılınca sonuç iyimser çıkıyor.
- *Kalibre max (P1) neden genel skoru düşürdü?* `other` modelinin alarmları, veri kalitesi olaylarının (frozen, gap_edge) alarmlarının önüne geçiyor. `other`'da ise kazanç gürültü düzeyinde kalıyor.
- *Offline satır ana sonuç olabilir mi?* Hayır. Geleceğe bakan öznitelikler kullanıyor; yalnız nedensellik kısıtı olmayan Kaggle çözümleriyle kıyas için var ve seçime girmiyor.
- *Notebook'un hiçbir şeyi değiştirmediğini nasıl kanıtlıyorsunuz?* Başta ve sonda `results/` altındaki 529 dosyanın boyutu ve değişiklik zamanı karşılaştırılıyor. Ayrıca bütün yeniden üretilen tablolar kayıtlı tablolarla anahtar sütunlar üzerinden birleştirilip kontrol ediliyor.

### 2026-11-26
- *Final model testte daha iyi bir modelin gerisinde; neden onu seçmediniz?* Seçim yalnız val'le ve önceden yazılmış kuralla yapıldı. Val'de iki model ayırt edilemiyordu; kural basit olanı seçti. Test sonucuna göre seçimi değiştirmek, testi bir doğrulama kümesine çevirmek olurdu.
- *Test skorları neden val'den yüksek?* Kurallar dahil bütün modellerde aynı yönde olduğu için parça bileşimi en olası neden. Test parçalarında kolay yakalanan frozen olayların payı daha yüksek.
- *Modeller kaydedilmemişken test skorları nasıl güvenilir?* Her model aynı seed, ayar ve veriyle yeniden eğitildi ve önce val skorları yeniden üretildi. 39 kontrolün 38'i birebir; kalanı da ihmal edilebilir farkta.
- *0,972 ROC-AUC'yi Kaggle'ın 0,9866'sıyla kıyaslayabilir miyiz?* Yalnız koşul farkları yazılarak. L1-OFF geleceğe bakan öznitelikler kullanıyor; Kaggle'da test 206 ayrı bina, bizde 200 etiketli binanın 40'ı. Ana sonuç nedensel olan L1 behavior.
- *Birincil metrikte (`other` PR-AUC) neden bu kadar düşük değerler var?* Taban 0,005; final model bunun 10 katı, en iyi modeller 24 katı. Ama davranışsal anomaliler az ve birkaç binada yoğun, bu yüzden mutlak değer düşük kalıyor.

### 2026-11-26
- **Notebook'taki 10⁻⁴'lük AUC farkı bir hata mı?** Hayır. Skorlar diske float32 olarak yazıldı; yuvarlama bazı skorları eşit hâle getiriyor ve PR-AUC'nin bağ işlemesi değişiyor. Eşiğe bağlı metrikler birebir aynı.
- **Demo neden eşiği testte yeniden seçmiyor?** Testte hiçbir şey seçilmez. Gösterilen eşik, val'de F1-max ile seçilen ve testte aynen uygulanan eşik; bunu bir pytest testi de denetliyor.
- **L4'ün skoru neden ayrı panelde?** Ölçekleri farklı: LightGBM olasılık (0–1) veriyor, L4 normalize tahmin hatası veriyor. Aynı eksende biri diğerini ezerdi.
- **Birleştirmenin hane ve şebeke sonuçlarını bozmadığını nasıl biliyorsunuz?** Birleştirme öncesi ve sonrası, kayıtlı ağırlıklarla yeniden üretilen 124 dizi ve dosya özetleri karşılaştırıldı; en büyük fark 0.
- **L1-FE full testte daha iyiyken final neden L1 behavior?** Seçim val'de, önceden yazılmış kuralla yapıldı. Testten sonra seçimi değiştirmek testi doğrulama kümesine çevirirdi; bu yüzden durum bir bulgu ve ders olarak raporlandı.

## Genel (kod, yapı, protokol, sızıntı)

### 2026-10-28
- **"`.gitignore`'a neden `data/raw/` ve `.pt` koydunuz?"** Ham veri büyük ve UCI'dan yeniden indirilebiliyor. Ağırlık dosyaları da kodla yeniden üretilebiliyor. Repoda yalnızca kaynak kod tutuluyor.
- **"Tek config dosyasının faydası ne?"** Bütün deneyler aynı yerden ayarlanıyor. Hiçbir sayı koda gömülü değil, böylece her deney tekrarlanabilir.
- **"Seed'i neden sabitlediniz?"** Aynı kod aynı sonucu versin diye. Ayrıca 3 farklı seed ile çalıştırıp sonucun şansa bağlı olmadığını gösteriyoruz (F9).
- **"Neden kronolojik bölme?"** Rastgele bölmede model gelecekteki saatleri eğitimde görmüş olur. Bu veri sızıntısıdır ve sonuçları olduğundan iyi gösterir.

### 2026-10-28
- **"`cudnn.deterministic = True` neden?"** cuDNN varsayılan olarak bazı işlemlerde en hızlı ama her seferinde aynı sonucu garanti etmeyen algoritmayı seçer. Biz biraz hızdan vazgeçip tekrarlanabilirliği tercih ettik.
- **"Config'te neden `yaml.safe_load` kullandınız?"** `safe_load` YAML dosyasının içinden kod çalıştırılmasına izin vermez. Bize yalnızca sayı ve metin lazım.
- **"`Global_intensity` neden öznitelik setinde yok?"** Akım ve aktif güç neredeyse doğrusal ilişkili (P ≈ V·I). Ek bilgi taşımıyor, sadece gereksiz tekrar olur. Bunu EDA'daki korelasyon matrisiyle göstereceğiz.
- **"Split tarihlerini neden test ettiniz?"** Kümeler kesişirse model test döneminden bir şeyi eğitimde görmüş olur. Bu sızıntıyı bir testle otomatik yakalamak, gözle kontrol etmekten daha güvenilir.

### 2026-10-28
- **"`setup_console()` neden her `main()`'de ayrı ayrı çağrılıyor?"** Kodlama ayarı yalnızca çalışan süreç için geçerli. `python -m` ile hangi betik başlatılırsa ayarı o yapmalı. Modül import edilirken çağırsaydık, testler gibi başka kodların çıktısını da habersizce değiştirirdik.
- **"Neden `hasattr(stream, "reconfigure")` kontrolü var?"** Notebook gibi bazı ortamlar stdout'u `reconfigure` metodu olmayan bir nesneyle değiştirir. Kontrol olmasa fonksiyon o ortamlarda kendisi hata verirdi.
- **"numpy'ı neden sabitlediniz?"** numpy 2.5 ile pandas 2.3.3 birlikte kullanıldığında zaman farkı hesaplarında uyarı çıkıyordu ve bu uyarı ileride hataya dönüşecek. Sürümü sabitlemek, kodun her bilgisayarda aynı davranmasını sağlıyor; tekrarlanabilirliğin bir parçası.
- **"Bunu bir testle neden doğruladınız?"** Hatanın önce var olduğunu, sonra düzeldiğini aynı testte gösteriyoruz. Yeni bir betik yazılıp çağrı unutulursa test bunu yakalar.

### 2026-10-28
- **"EDA'yı neden yalnızca eğitim kümesinde yaptınız?"** Lookback ve log1p gibi kararlar bu analizlerden çıktı. Test verisine bakarak karar verseydik, test sonucumuz olduğundan iyi görünebilirdi; bu da bir tür veri sızıntısı.
- **"ACF'de neden doldurulmuş saatleri de kullandınız?"** ACF kesintisiz bir seri ister. Eğitimde doldurulan saatlerin oranı yalnızca %0,56, etkisi çok küçük.
- **"Uç günleri bulurken neden yalnızca tam gözlenen günleri aldınız?"** Doldurulmuş saatler o günün toplamını yapay olarak değiştirirdi. Bu yüzden 13 günü sıralamaya almadık.
- **"Alt sayaç paylarının toplamı neden tam %100?"** `other_energy`, toplam enerjiden alt sayaçlar çıkarılarak türetildi. Toplamın %100 çıkması bu tanımın tutarlı uygulandığını doğrulayan bir kontrol.

### 2026-10-30
- Deney günlüğündeki çakışmayı neden iki tarafı da koruyarak çözdünüz? Günlük, sadece satır eklenen bir kayıt. İki dal aynı tabloya farklı satırlar eklediği için ikisi de geçerli; birini silmek, bir kararın gerekçesini kaybetmek olurdu.
- Birleştirmeden önce neyi kontrol ettiniz? Testlerin geçtiğini ve test kümesine dokunulmadığını (CLAUDE.md kural 5).
- Neden `main`'i EDA dalına birleştirdiniz? EDA PR'ı çakışmasız ve güncel `main` üzerinde test edilmiş olarak açılsın, çakışma `main`'de çözülmek zorunda kalmasın diye. Alternatif `rebase` olurdu, ama zaten push'lanmış bir dalın geçmişini değiştirirdi.
- GRU, Seasonal Naive 168'i yaklaşık %15 geçiyor. Bu fark anlamlı mı? Tek seed ile bilinemez; F9'daki 3 seed'lik tekrar bunu göstermek için yapılacak.

### 2026-10-30
- **Seq2seq neden en iyi çıktı? Takvim zaten girdideydi.** Düz LSTM'de 24 çıktının hepsi tek bir gizli durumdan doğrusal katmanla üretiliyor. Decoder ise her hedef saat için ayrı bir durum taşıyor ve o saatin takvimini girdi olarak alıyor. Saate özgü deseni (akşam tepesi gibi) uzun ufukta korumak bu yüzden kolaylaşıyor; kazancın h≥6'da gelmesi bunu destekliyor.
- **Decoder'a hedef saatlerin takvimini vermek sızıntı değil mi?** Hayır. Takvim (saat, gün, ay, tatil) önceden bilinir; decoder'a hedef değeri ya da önceki tahmini verilmiyor. Bunu bir test kontrol ediyor. Teacher forcing kullanmadığımız için eğitim ve tahmin aynı şekilde çalışıyor.
- **Lookback 48 ile 336 neden neredeyse aynı sonucu veriyor?** Haftalık örüntü büyük ölçüde takvim öznitelikleriyle (gün sin/cos, hafta sonu) taşınıyor. Son 48 saat ise güncel tüketim düzeyini veriyor. Daha uzun geçmiş ek bilgi taşısa da tek seed ile 0,005 kW'tan küçük fark ayırt edilemiyor.
- **XGBoost neden derin modellerin gerisinde, ama h=1'de en iyi?** h=1'de son gözlemler çok bilgi taşıyor ve ağaçlar bunu iyi kullanıyor. Uzun ufukta ise yalnızca son gözlenen saatin takvimini görüyor; diziyi modelleyemediği için saate özgü deseni taşıyamıyor.
- **Bu kadar konfigürasyon aynı val'de denendi; seq2seq'in üstünlüğü gerçek mi?** Fark (0,4747'ye karşı 0,4838–0,4844) tek seed'e dayanıyor. Aşama 5'teki 3 seed'lik tekrar ve Aşama 4'teki train içi çapraz doğrulama bunu sınayacak. Test kümesine tek bir final değerlendirme için dokunulmadı.

### 2026-10-31
- **Veri setinin yerel saatte olduğunu nasıl kanıtladınız?** Ham veride ilkbahar geçişlerindeki kopya saatlerle ve geçiş öncesi/sonrası günlük profilin kaydırma 0'da en iyi örtüşmesiyle. Sabit ofsetli bir kayıtta günlük rutin bir saat kayardı.
- **Daha fazla bilgi veren W2 neden daha kötü?** Ek girdi, ayarlanmamış bir modelde ezberleme imkânı da veriyor; train kaybı düşerken val kaybı yükseliyor. Oracle'ın üst sınır olması ancak model bu bilgiyi genelleştirebildiğinde anlamlı.
- **W2'yi neden model seçiminde kullanmıyorsunuz?** Tahmin anında hedef saatlerin gerçek havası bilinemez; olsa olsa hatası olan bir hava tahmini bilinir. W2 bu yüzden ulaşılabilir bir performans değil, bir tavan ölçüsü.
- **ERA5 "geçmiş hava" için uygun mu?** Gerçekçi ama kusursuz değil. ERA5 bir yeniden analiz: yaklaşık 25 km'lik hücre ortalaması ve kısa süre sonraki gözlemleri de kullanmış olabilir. Bunu sınırlılık olarak yazdık.
- **Sonuç "hava işe yaramaz" mı?** Hayır. Sonuç yalnızca "B için ayarlanmış hiperparametrelerle, son 48 saatin tüketimine ek bir katkı ölçülemedi". Düzenlileştirme ve öznitelik azaltma denenmedi.

### 2026-10-31
- **Normalize skor neden saat × gün std'sine bölünüyor da günlük ortalamaya bölünmüyor?** Hata büyüklüğü akşam saatlerinde 4–5 kat artıyor (0,17'den 0,93 kW'a). Std, "bu saatte olağan hata ne kadar?" sorusunu doğrudan yanıtlıyor.
- **Std tablosunu val artıklarından hesaplasak ne olurdu?** Kalibrasyonu daha iyi olurdu, ama val'i iki kez kullanmış olurduk (hem ölçek hem eşik). Tablo yalnız train'den hesaplanınca eşik dışında val'den bir şey öğrenilmiyor.
- **F1-max eşiği neden 5 seed'in birleşiminde seçiliyor?** Tek kopyada yaklaşık 10 olay var, eşik çok oynar. Birleşik seçim yöntem başına tek ve kararlı bir eşik veriyor; test'e de aynen uygulanıyor.
- **Yumuşatmanın bedeli ne?** Alarm 1–2 saat gecikebiliyor ve 1 saatlik spike 3'e bölünerek sönüyor. Spike recall'ünün A0'dan düşük olmasının bir nedeni bu.
- **Model checkpoint'inin val ile seçilmesi anomali sonucunu geçersiz kılar mı?** Hayır, ama val'i biraz iyimser yapar. Bu yüzden asıl hüküm test kümesinde verilecek; test ne model seçiminde ne eşik seçiminde kullanıldı.

### 2026-11-03
- **In-sample ve OOF std farkı neden önemli?** Model eğitim verisine uyum sağladığı için in-sample artıklar farklı dağılır. Eşik OOF ölçeğiyle seçildiyse skor da aynı ölçekle hesaplanmalı, yoksa eşik anlamını yitirir; burada alarm sayısı 8'den 3'e düşüyordu.
- **Enjeksiyon neden tüm test kopyasına yapılıyor?** Protokol böyle ve oran sınırı küme uzunluğuna bağlı. Kısa aralıkta 12–48 saatlik olaya yer kalmıyor. Ayrıca olaylar artık tarih seçimine göre yer değiştirmiyor.
- **Yumuşatma neden ani yükselmeyi kaçırıyor?** Skor son 3 saatin ortalaması olduğu için 1 saatlik sıçrama üçe bölünür ve alarm olay bittikten sonra gelebilir. Uzun olaylarda ise gürültüyü bastırır; bu bir ödünleşimdir.
- **Test geneli ile seçili gün neden farklı?** Test geneli 3 seed ortalaması ve tüm saatlik pencereler; seçili gün tek seed ve tek pencere.
- **Demo test kümesini kullanıyor, sızıntı var mı?** Hayır. Kalibrasyon yalnızca eğitim ve val-dev'den geliyor; test yalnızca gösterim için. Kolay/zor gün ve tip tablosu hiçbir seçime bağlanmıyor.

### 2026-11-08
- **Yaz-dev'de neden F10-opt kullanılmadı?** Optuna'nın CV'si kat 3'ü görmüştü. O modelle yaz-dev'i ölçmek, seçimde kullanılan veriyle değerlendirme yapmak olurdu.
- **Orta senaryo neden eski ana senaryoyla aynı?** Önceki sonuçlarla kıyas korunsun diye. Aynı seed'lerle aynı sonucu verdiğini de doğruladık.
- **Eşik neden iki mevsimin birleşiminde seçildi?** Tek mevsimde seçilen eşik öbür mevsime taşınmıyor; final testteki düşüşün olası nedeni buydu.
- **Bu sayılar neden "test sonucu" değil?** Seçim de bu kümelerde yapıldı. Bağımsız ölçüm ancak ELD'nin kullanılmamış 210 müşterisiyle, tek seferlik yapılacak.

### 2026-11-24
- **Birleştirmeden önce neden etiket attın?** `v1.5-temizlik-oncesi` geri dönülebilir bir nokta. Bir şey ters giderse `main` tek komutla eski haline alınabilir; dalları silmekten veya force-push'tan çok daha güvenli.
- **Neden `--no-ff`?** Her PR'ın `main`'de kendi birleştirme commit'i oluyor. Hangi değişikliğin hangi PR'dan geldiği geçmişte görünüyor, gerekirse tek bir PR geri alınabiliyor.
- **v2'nin test sonucu neden "final" sayılmaz?** Taban kuralı, test sonucu görüldükten sonra tasarlandı. Aynı test kümesinde yeniden ölçmek bağımsız bir sınama değil, ve kazanç zaten sentetik drop değerinin tabanın altında olmasından geliyor.
- **pandas 3 hatası neden önemliydi?** `asi8`'in birimi pandas 3'te nanosaniye olmak zorunda değil, `Timedelta.value` ise her zaman nanosaniye. Bu yüzden ±3 saatlik sınır koruması sessizce ±125 güne çıkıyordu. Düzeltme, birimden bağımsız Timedelta karşılaştırması kullanıyor.

### 2026-11-25
- Test kümesi görüldükten sonra tasarlanan bir kural neden "bağımsız test sonucu" sayılmaz? Cevap: tasarım test verisine bakılarak yapıldığı için sonuç iyimser olur, yani seçim yanlılığı.
- v2'nin kazancı neden yalnızca drop'tan geliyor? Cevap: enjekte edilen drop değeri (0,05 kW) taban eşiğinin (0,155 kW) altında, bu yüzden kural bu olayları neredeyse tanım gereği yakalıyor.
- pandas 3 düzeltmesinin sonuçları değiştirmediği nasıl gösterildi? Cevap: kayıtlı tablolar ve doldurulan saat sayıları düzeltmeden önce ve sonra karşılaştırıldı.

### 2026-11-25
- v2 neden alınmadı? Çünkü test kümesi görüldükten sonra tasarlanan bir yöntemi testte ölçmek seçim yanlılığı yaratır. Aynı taban kuralının val-hold sonucu zaten raporda var.
- Çakışmalar nasıl çözüldü? İki taraf da yalnızca satır eklemişti, bu yüzden hiçbir satır silinmeden tarih sırasıyla birleştirildi.
- Final sonuçların değişmediği nasıl gösterildi? Birleştirme öncesi etikete (`v1.5`) karşı `git diff` alındı ve fark çıkmadı.

### 2026-11-25
- Dosyaları taşıdığınız hâlde sonuçların değişmediğini nasıl kanıtladınız? Taşımadan önce ve sonra aynı ağırlıklarla 265 çıktı üretildi ve birebir karşılaştırıldı; aynı kodla iki koşunun da aynı sonucu verdiği ayrıca gösterildi.
- Test kümesini burada kullanmak sızıntı değil mi? Hayır; hiçbir şey seçilmedi, kayıtlı seçimler aynen okundu. Bu bir regresyon kontrolü, gerekçesi günlükte.
- `parents[2]` yerine işaret dosyası araması neden daha iyi? Dosya bir klasör derine taşınınca `parents[2]` sessizce yanlış klasörü gösterir; arama ise konumdan bağımsızdır ve kök bulunamazsa açık hata verir.
- XGBoost neden iş parçacığına bağlı? Paralel histogram toplamlarında kayan noktalı sayıların toplanma sırası değişir; bu küçük farklar ağaçtaki bölünme kararlarını etkileyebilir.

### 2026-11-25
- Çakışan bina-saat kayıtları neden tamamen silindi de biri seçilmedi? Birini seçmek, etikete ya da tüketime dair kanıtsız bir karar olurdu.
- "Sabit okuma" nasıl tanımlandı ve sonuç bu tanıma ne kadar bağlı? Aynı değerin en az 3 ardışık saat tekrar etmesi; eşik 2 ile 6 saat arasında değişince oran pek değişmiyor.
- Anomalilerin üçte ikisinin sabit okuma olması modeli nasıl etkiler? Basit bir kural (ardışık aynı değer) yüksek başarı verebilir; bu nedenle literatürdeki yüksek ROC-AUC değerleri dikkatle yorumlanmalı.

### 2026-11-25
- Çakışan bina-saat kayıtları neden tamamen silindi? Birini seçmek kanıtsız bir karar olurdu; bu veride zaten hiç çakışma çıkmadı.
- "Sabit okuma" tanımı keyfi değil mi? 3 saatlik eşik bir seçim; ama 2 ile 6 saat arasında sonuç pek değişmiyor ve bu duyarlılık tabloyla gösterildi.
- Boşluk kenarı bulgusu modeli nasıl etkiler? Eksik veriyi işaretleyen basit bir öznitelik bile etiketlerin büyük kısmını yakalayabilir. Bu yüzden derin modelin gerçekten ne kattığını ancak bu kurallarla kıyaslayarak gösterebiliriz.
- Etiketler yanlış mı? Hayır: veri kalitesi sorunu da bir anomali türü. Ama hane hattındaki "tüketim anomalisi" tanımıyla aynı şey değil; sonuçlar yorumlanırken bu fark belirtilmeli.
