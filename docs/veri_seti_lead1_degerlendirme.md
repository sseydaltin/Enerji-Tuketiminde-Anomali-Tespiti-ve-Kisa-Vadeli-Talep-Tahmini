# LEAD1.0 Veri Seti Değerlendirmesi (Yönerge §5 kriterlerine göre)

**Tarih:** 2026-11-10
**Durum:** Masa başı inceleme. Veri **indirilmedi**. Buradaki sayılar asıl makaleden, Kaggle yarışma sayfasından ve GitHub deposundan alındı. "Bizim hesabımız" diye işaretli sayılar, kaynaktaki sayılardan yapılan basit aritmetiktir. Veri indirilince her sayı tekrar kontrol edilmelidir (§7).

Kaynak düzeyi işaretleri:
- **TM**: tam metin okundu.
- **Ö**: yalnız özet okundu.
- **S**: resmî sayfa okundu (Kaggle veya GitHub).

---

## 0. Kısa sonuç

1. **LEAD1.0 yönergenin kriterlerinin çoğunu karşılıyor.**
   - Gerçek, elle etiketlenmiş anomalileri olan açık bir veri seti.
   - Büyük: 1.413 sayaç ve yaklaşık 12 milyon saatlik okuma.
   - Lisansı CC BY 4.0, akademik kullanım açıkça serbest.
   - Dizi modelleri (LSTM, GRU, AE) için uygun.
   - Mevcut donanımla çalışılabilir.
2. **Bu, bizim projemizin en zayıf noktasını kapatıyor.** IHEPC'de anomali etiketi hiç yok. Bu yüzden anomali sonuçlarımız sentetik enjeksiyona dayanıyor. Yönerge ise "yetersiz etiketlenmiş" veri setinden kaçınmayı ve etiket doğruluğunu açıklamayı istiyor. LEAD bu kriteri doğrudan karşılıyor.
3. **Ancak LEAD tek başına IHEPC'nin yerini almaya uygun değil.** Sebepleri:
   - **Alan farklı:** Veri ticari binalardan geliyor, hane değil. Proje adı "enerji tüketimi" olduğu için kapsam dışı değil, ama bütün anlatı değişir.
   - **Süre kısa:** Bina başına yalnızca 1 yıl (2016) var. 24 saatlik tahmin görevi için mevsimsellik öğrenilemez.
   - **Test etiketleri gizli:** Kaggle test kümesinin etiketleri yayımlanmadı.
   - **Tam sürüm yok:** "Tam sürüm (1.000+ bina) sonra yayımlanacak" denmiş, fakat GitHub deposu Haziran 2022'den beri güncellenmemiş.
   - **Etiket kalitesi belirsiz:** Hakkında soru işaretleri var (§4).
4. **Önerimiz:** IHEPC tahmin ve hane anomali hattı olarak kalsın. LEAD, **"gerçek etiketli ikinci anomali veri seti"** olarak eklensin; ELD'nin şebeke düzeyinde eklendiği gibi. Bu karar sizin (§5).

---

## 1. LEAD1.0 nedir?

| Özellik | Değer | Kaynak |
|---|---|---|
| Tam adı | LEAD1.0: A Large-scale Annotated Dataset for Energy Anomaly Detection in Commercial Buildings | [1] TM |
| Yayın | Gulati & Arjunan, ACM e-Energy '22, s. 485–488. arXiv:2203.17256 | [1], Crossref |
| Kaynak veri | ASHRAE Great Energy Predictor III (Kaggle 2019) [6]. Bu veri Building Data Genome 2'nin bir parçası [7]. GEPIII'te 16 sahadan 1.636 ticari bina var | [1] TM |
| Kapsam | 1.413 elektrik sayacı (yalnız elektrik; soğutma suyu, buhar ve sıcak su sayaçları yok), 16 bina türü | [1] TM |
| Dönem ve çözünürlük | 2016 takvim yılı, **saatlik**; sayaç başına 8.784 örnek (366 × 24) | [1] TM |
| Toplam nokta | 12.411.792 beklenen; 12.060.910 etiketli (eksikler hariç) | [1] TM |
| Eksik nokta | 350.882 (≈ %2,83). *Bizim hesabımız.* Makale bu farkı açıklamıyor | [1] + hesap |
| Anomali sayısı | 199.640, yani etiketli noktaların ≈ **%1,66**'sı. *Oran bizim hesabımız* | [1] TM |
| Etiket türü | Saat başına ikili etiket (0 normal, 1 anomali). Makale "nokta" ve "ardışık/toplu" anomaliyi tanımlıyor | [1] TM, [3] S |
| Etiketleme | Yazarlar ve iki katkıcı, R/Shiny tabanlı açık kaynak bir araçla elle etiketlemiş. Her 24 saatlik pencere 3–5 dakika sürmüş, toplam yaklaşık 100 kişi-saat. Bir "doğrulama süreci" uygulandığı yazıyor ama ayrıntısı ve etiketleyiciler arası uyum verilmiyor | [1] TM |
| Kaggle yarışması (2022) | **train.csv**: 200 bina, sütunlar `building_id, timestamp, meter_reading (kWh), anomaly`. **test.csv**: başka 206 bina, **etiketsiz**. Ölçüt ROC-AUC. Hâlâ "Late Submission" açık | [3] S |
| Ek dosyalar | `train_features.csv`, `test_features.csv`: GEPIII kazananlarının kullandığı bina ve hava öznitelikleri | [3] S |
| Lisans | Kaggle kuralları §7A: veri "ticari veya ticari olmayan her amaçla … akademik araştırma ve eğitim dahil" kullanılabilir; ek olarak **CC BY 4.0** | [3] S |
| Erişim koşulu | Kaggle hesabıyla giriş ve yarışma kurallarının kabulü gerekiyor. Kurallar §7B: veri, yarışmaya katılmamış kişilere **yeniden dağıtılamaz** | [3] S |
| GitHub | `samy101/lead-dataset`. İçinde yalnızca `data/lead1.0-small.zip` (yaklaşık 10 MB; içeriği incelenmedi) ve etiketleme aracı var. Son güncelleme 2022-06-28; LICENSE dosyası 2022-03-29'da silinmiş | [4] S |

---

## 2. Yönerge §5 kriterlerine göre kontrol

| Yönerge kriteri | LEAD1.0 | Durum | Karşılaştırma: bizim IHEPC |
|---|---|---|---|
| Kaynak: açık, akademik, Kaggle vb. | Hakemli bir konferans makalesi (e-Energy '22) ve Kaggle yarışması | Uygun | UCI deposu, DOI'li |
| Kaynak, kapsam ve özelliklerin açıkça belirtilmesi | Makale ve Kaggle veri sayfası sütunları, dönemi ve çözünürlüğü tanımlıyor | Uygun | Uygun |
| Lisans ve kullanım koşulları | CC BY 4.0 ve akademik kullanım serbest. **Ancak** veri Kaggle girişiyle alınır ve yeniden dağıtılamaz: repoya konamaz, her ekip üyesi kendisi indirmeli. GitHub deposunda lisans dosyası yok | Uygun, koşullu | CC BY 4.0, serbest indirme |
| Konuya uygunluk | Anomali tespiti için doğrudan uygun: veri seti bu amaçla etiketlenmiş. Kısa vadeli tahmin için zayıf: bina başına 1 yıl | Anomali: uygun. Tahmin: kısmen | Tahmin: uygun (4 yıl). Anomali: etiket yok |
| Veri miktarı | ~12 M saatlik nokta (tam LEAD1.0). Kaggle eğitim kümesi 200 bina × ~8.760–8.784 saat ≈ 1,75 M satır | Uygun | ~34,6 bin saatlik nokta, tek hane |
| Veri kalitesi, eksik ve hatalı veri | ≈ %2,8 eksik (hesap). Etiketler elle konmuş ama tutarlılık ve tamlık belirsiz (§4.1) | Kısmen | ~%1,25 eksik (dakikalık); belgelenmiş imputasyon |
| Sınıf dağılımı ve dengesizlik | ≈ %1,7 anomali, yani güçlü dengesizlik. Kaggle eğitim alt kümesindeki oran doğrulanmalı | Yönetilebilir; PR-AUC ve F1 raporlanmalı | Gerçek oran bilinmiyor; enjeksiyonla biz belirliyoruz |
| Etiketli mi, etiketler doğru mu? | **Etiketli**, saat düzeyinde, elle. Doğruluk kanıtı zayıf: uyum ölçüsü yok, kurallar binaya göre değişiyor, etiketleme süresi tutarsız (§4.1) | Kısmen | Anomali etiketi **yok**; sentetik |
| Train/val/test'e ayrılabilirlik | Bina bazlı bölme doğal: Kaggle böyle yapmış. Zaman içinde bölme mümkün ama 1 yıl kısa. Kaggle test etiketleri gizli olduğu için 200 etiketli binayı kendimiz bölmeliyiz | Uygun, protokol kararı gerekir (§6) | Kronolojik bölme, 4 yıl |
| Mimariye uygunluk | Saatlik tek değişkenli diziler. LSTM/GRU-AE, artık tabanlı dedektör ve **denetimli** LSTM/GRU sınıflandırıcı kurulabilir | Uygun | Uygun |
| Donanımla yapılabilirlik | Birkaç on MB'lık CSV; CPU'da yapılabilir | Uygun | Uygun |
| "Anlamlı deney ve güvenilir değerlendirme" | Gerçek etiket, yayımlanmış referans sonuçlar ([1], [2] ve §3) ve kıyaslanabilir ROC-AUC ölçütü var | Uygun | Anomalide değerlendirme sentetik etikete bağlı |

**Özet:** LEAD, yönergenin "etiket" ve "güvenilir değerlendirme" kriterlerinde IHEPC'den **daha güçlü**. "Tahmin için veri miktarı ve süre" kriterinde ise **daha zayıf**.

---

## 3. LEAD'i kullanan çalışmalar

OpenAlex'te LEAD1.0'a 32 atıf var (2026-11-10). Aşağıdakiler veriyi gerçekten kullananlardır.

| Çalışma | Görev ve protokol | Bildirilen sonuç | Düzey |
|---|---|---|---|
| Gulati & Arjunan 2022 [1] | 8 denetimsiz PyOD yöntemi; **24 saatlik günün tamamı** anomali mi? Normal günlerin %80/10/10'u ve anomali günlerin %10/20/70'i train/val/test'e, **rastgele** dağıtılmış. Eşik val'de seçilmiş | En iyi F1 0,431 (KNN), en düşük 0,397 (HBOS); kesinlik ~0,90, duyarlılık ~0,28 | TM |
| Fu, Arjunan & Miller 2022 [2] | Kaggle **kazanan** çözüm: ağaç tabanlı, **denetimli** sınıflandırıcı; nokta düzeyi; 206 görülmemiş bina | Private LB ROC-AUC **0,9866**. En büyük etkiyi "değer değişimi" öznitelikleri yapmış | Ö |
| Abiodun & Olukanmi 2025 [8] | Denetimli LR, RF, XGBoost, kNN, Extra Trees | XGBoost ROC-AUC 0,99, duyarlılık 0,83. Bölme yöntemi özette yok | Ö |
| Cansever & Konyar 2026 [9] | K-means ile bina kümeleme, ardından küme başına IF, LOF, XGBoost, RF ve LightGBM | Global model F1 0,572; küme bazlı en iyi F1 0,903 | Ö |
| Almadani vd. 2026 [10] | Denetimli, bağlam kapılı Transformer; elektrik + hava + bina bağlamı | ROC-AUC 0,958 ± 0,003, AUPRC 0,802 ± 0,011 | Ö |
| Orvati Nia vd. 2026 [11] | Yalnız normal pencerelerle eğitilen GAN-LSTM; 60 saatlik pencere; 200 bina train, 206 bina test | Pencere düzeyi F1 0,89, ROC-AUC 0,83. **LSTM-AE**: F1 0,60, ROC-AUC 0,70. Kaggle test etiketleri gizli olduğu halde 206 test binasında sonuç veriyor; etiketlerin kaynağı **doğrulanmalı** | TM |
| Hela, Handigol & Arjunan 2025 [5] | Denetimsiz; TimeGPT ve MOMENT temel modelleri, IQR, değiştirilmiş z-skoru, IF vb. ile kıyaslanmış | Sayılar özetten okunamadı (doğrulanmalı) | Ö |
| Islavath vd. 2026 [12] | Eğitimsiz LLM ve TSFM tabanlı anomali tespiti; 406 bina | Sayı özetten okunamadı | Ö |
| Haddad, Jerbi & Smaali 2024 [13] | 15 hazır yöntem; 3 etiketli veri seti | Bagging/boosting yöntemleri sinir ağlarından iyi | Ö |

**Bu tablodan çıkan dersler:**

- **Skorlar yine protokole bağlı.** Literatürdeki sonuçlar şu farklı koşullardan geliyor:
  - Denetimsiz, gün düzeyi F1 ~0,40–0,43 [1].
  - Denetimli, nokta düzeyi ROC-AUC ~0,96–0,99 [2], [8], [10].

  Anomali oranı ~%1,7 olduğunda ROC-AUC kolayca yüksek çıkar; F1 ve PR-AUC çok daha düşük olabilir. IHEPC literatür taramasındaki uyarımız burada da geçerli.
- **Denetimli ağaç modelleri çok güçlü** [2], [8], [13]. Bizim LSTM-AE veya GRU artık dedektörümüz denetimsiz olduğu için, LEAD'de bu skorlara yaklaşması beklenmemeli. Bunu baştan söylemek gerekir.
- **[11], LSTM-AE için denetimsiz bir referans sayı veriyor:** pencere F1 0,60. Bizim yöntemimizle kıyaslanabilecek en yakın sayı bu. Ancak pencere tanımı ve bölme farklı.

---

## 4. Riskler ve sınırlılıklar

### 4.1 Etiket kalitesi
- **Etiketleme süresi tutarsız.** Makale iki şey söylüyor: her 24 saatlik pencere 3–5 dakika sürdü ve toplam yaklaşık 100 kişi-saat harcandı [1]. Oysa 1.413 sayaç × 366 gün ≈ 517 bin gün-penceresi; bunun her biri 3 dakika sürseydi toplam yaklaşık 25.900 saat olurdu (*bizim hesabımız*).
  - Yani her günün tek tek incelenmemiş olması muhtemel.
  - Sonuç olarak **kaçırılmış anomaliler "normal" etiketli kalmış olabilir** (etiket gürültüsü).
  - Bu bir çıkarım, kanıt değil. Raporda "makaledeki ifadelerden çıkan soru" olarak yazılmalı.
- **Kural tek tip değil.** Makale, anomali tanımının binadan binaya değiştiğini açıkça söylüyor [1]. Etiketleyiciler arası uyum (örneğin Cohen κ) raporlanmamış.
- **Etiketler veri kalitesi sorunlarını da içerebilir.** Örneğin sıfır okumalar, sabit kalan sayaç, sensör arızası. Bu durumda "anomali" yalnızca tüketim davranışını değil, sensör hatalarını da kapsar.
  - Kazanan çözümde "değer değişimi" özniteliklerinin en etkili olması [2] buna işaret ediyor olabilir.
  - Bu da doğrulanmalı: veri indirilince etiketli noktaların ne kadarının sıfır veya sabit değer olduğu sayılmalı.

### 4.2 Erişim ve tekrar üretilebilirlik
- **Kaggle test etiketleri gizli.** Kaggle'a geç gönderim yalnızca ROC-AUC verir; F1 ve olay F1 hesaplanamaz. Bu yüzden 200 etiketli binayı kendimiz train/val/test'e bölmemiz gerekir. Bu durumda sonuçlar Kaggle liderlik tablosuyla doğrudan kıyaslanamaz.
- **"Tam LEAD1.0" sürümü yayımlanmamış görünüyor.** Kaggle ve GitHub'da "1.000+ binalı tam sürüm yarışmadan sonra yayımlanacak" yazıyor. Ama GitHub deposu yarışma bitmeden (2022-06-28) son kez güncellenmiş ve içinde yalnızca `lead1.0-small.zip` var. Kullanılabilir etiketli veri pratikte Kaggle'daki 200 bina (veya small.zip, içeriği doğrulanmalı).
- **Yeniden dağıtım yasak.** Bu açıdan `data/raw/` zaten git'e girmiyor, sorun yok. README'ye indirme adımları yazılmalı; her ekip üyesinin Kaggle hesabı olmalı.

### 4.3 Projeye uyum
- **Alan farkı:** Ticari binalar mesai ve hafta sonu düzenine sahip ve bina yönetim sistemiyle çalışıyor. Hane tüketimi çok daha düzensiz. LEAD'den elde edilen sonuç "hane anomali tespiti" diye sunulamaz.
- **Bina başına 1 yıl:** Kronolojik bölmede eğitim yaz aylarını, test ise yalnızca kış aylarını görür. Yıllık mevsimsellik öğrenilemez. Bu nedenle tahmin görevini LEAD'e taşımak anlamlı değil.
- **Mevcut kural farkı:** CLAUDE.md ve README "yalnızca tarih bazlı kronolojik bölme" diyor. LEAD'de doğal bölme **bina bazlıdır**. Bu bir kural değişikliği demek ve sizin onayınızı gerektirir.

---

## 5. Projemiz için seçenekler

| Seçenek | Artı | Eksi |
|---|---|---|
| **A. IHEPC ile devam, LEAD yok** | İş yükü yok; mevcut sonuçlar tutarlı | Yönergenin "etiket" kriterinde zayıf kalırız; anomali değerlendirmesi sentetik kalır |
| **B. IHEPC + LEAD (ikinci anomali veri seti)** *(önerimiz)* | Gerçek etiketle değerlendirme yapılır. Sentetik enjeksiyonla bulunan sonuçların gerçek veride tutup tutmadığı test edilir. Literatürle (özellikle [1] ve [11]) kıyaslanacak referans sayılar olur | Yeni veri yükleme, bina bazlı bölme ve pencereleme kodu gerekir; protokol kararı gerekir; denetimsiz yöntemimiz denetimli literatür skorlarının altında kalacaktır (önceden söylenmeli) |
| **C. Tamamen LEAD'e geçiş** | Etiket sorunu tamamen çözülür | Tahmin görevi zayıflar (1 yıl); hane odaklı bütün analiz, demo ve rapor yeniden yazılır; zamana sığması zor |

---

## 6. Seçenek B seçilirse protokol taslağı

Henüz uygulanmadı. Aşağıdaki maddelerin her biri karar gerektirir.

1. **Veri:** Yalnızca Kaggle `train.csv` (200 etiketli bina). Test etiketleri gizli olduğundan `test.csv` kullanılmaz.
2. **Bölme (karar gerekli):** Binalar sabit bir seed ile üç gruba ayrılır, örneğin 140 train / 30 val / 30 test. Bina listesi `configs/config.yaml`'a yazılır. Her bina kendi içinde zaman sırasını korur; pencereler bina sınırını geçmez.
3. **Ölçekleme ve sızıntı:** `log1p` ve ölçekleyici yalnızca train binalarına fit edilir.
   - Bina başına normalizasyon istenirse, test binasının istatistikleri tüm yılından hesaplanamaz; bu geleceğe bakmak olur.
   - Bunun yerine yalnızca geçmiş pencerelerden hesaplanan hareketli istatistik kullanılır.
4. **Eğitim:** LSTM-AE ve artık dedektörü yalnızca train binalarının **normal** etiketli pencereleriyle eğitilir. Ek olarak denetimli bir referans (XGBoost, zaten repoda) ve isteğe bağlı bir denetimli GRU sınıflandırıcı eklenebilir.
5. **Eşik:** Yalnızca val binalarında seçilir. Test binaları tek seferlik final değerlendirmede kullanılır.
6. **Metrikler:**
   - Nokta düzeyinde ROC-AUC (Kaggle ve literatürle kıyas için), PR-AUC ve val eşiğinde F1.
   - Mevcut protokolle tutarlılık için olay F1.
   - [1] ile kıyas için gün düzeyi F1.
7. **Testler:** Bina grupları kesişmiyor; ölçekleyici yalnızca train binalarından fit ediliyor; pencere bina sınırını aşmıyor.

---

## 7. Doğrulanması gerekenler

- [ ] Kaggle `train.csv`: satır sayısı, bina sayısı, anomali oranı, `meter_reading` eksik oranı.
- [ ] Etiketli anomalilerin ne kadarının sıfır, sabit veya eksik okuma olduğu (§4.1).
- [ ] `lead1.0-small.zip` içeriği: kaç bina var, Kaggle train ile aynı mı?
- [ ] [5], [11] ve [12]'te test etiketlerinin nasıl elde edildiği.
- [ ] Özetten (Ö) alınan sayılar ([2], [8], [9], [10]) tam metinden kontrol edilmeli.
- [ ] Hela vd. 2025 [5] sonuç sayıları.

---

## Kaynaklar (IEEE)

[1] M. Gulati and P. Arjunan, "LEAD1.0: A large-scale annotated dataset for energy anomaly detection in commercial buildings," in *Proc. 13th ACM Int. Conf. Future Energy Systems (e-Energy '22)*, 2022, pp. 485–488, doi: 10.1145/3538637.3539761. (Ön baskı: arXiv:2203.17256)

[2] C. Fu, P. Arjunan, and C. Miller, "Trimming outliers using trees: Winning solution of the large-scale energy anomaly detection (LEAD) competition," in *Proc. 9th ACM Int. Conf. Systems for Energy-Efficient Buildings, Cities, and Transportation (BuildSys '22)*, 2022, pp. 456–461, doi: 10.1145/3563357.3566147.

[3] M. Gulati and P. Arjunan, "Large-scale Energy Anomaly Detection (LEAD)," Kaggle Community Prediction Competition, 2022. [Online]. Available: https://www.kaggle.com/competitions/energy-anomaly-detection (Erişim: 10 Kas. 2026)

[4] samy101, "lead-dataset," GitHub repository. [Online]. Available: https://github.com/samy101/lead-dataset (Erişim: 10 Kas. 2026)

[5] B. Hela, P. P. Handigol, and P. Arjunan, "Are time series foundation models good for energy anomaly detection?," in *Proc. 16th ACM Int. Conf. Future Energy Systems (e-Energy '25)*, 2025, pp. 656–665, doi: 10.1145/3679240.3734633. (Konferans numarası doğrulanmalı)

[6] C. Miller *et al.*, "The ASHRAE Great Energy Predictor III competition: Overview and results," *Sci. Technol. Built Environ.*, vol. 26, no. 10, pp. 1427–1447, 2020, doi: 10.1080/23744731.2020.1795514.

[7] C. Miller *et al.*, "The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition," *Sci. Data*, vol. 7, no. 1, Art. no. 368, 2020, doi: 10.1038/s41597-020-00712-x.

[8] T. P. Abiodun and P. Olukanmi, "Performance evaluation of machine learning models for anomaly detection in energy usage data," in *Proc. Southern African Universities Power Eng. Conf. (SAUPEC)*, 2025, pp. 1–6, doi: 10.1109/SAUPEC65723.2025.10944339.

[9] B. N. Cansever and M. Z. Konyar, "A clustering-based supervised approach for anomaly detection in building energy consumption," *E3S Web Conf.*, vol. 727, Art. no. 02012, 2026, doi: 10.1051/e3sconf/202672702012.

[10] M. Almadani, S. Atalla, Y. Himeur, H. A. Alkhazaleh, W. Mansoor, and A. Amira, "Context-aware transformer for supervised anomaly detection in smart building energy consumption," in *Proc. IEEE Int. Conf. Consumer Electronics (ICCE)*, 2026, pp. 1–5, doi: 10.1109/ICCE67443.2026.11449641.

[11] F. Orvati Nia, S. Salehi, and J. Peeples, "Evaluating GAN-LSTM for smart meter anomaly detection in power systems," arXiv:2601.09701, 2026.

[12] G. Islavath, K. Kumar G, A. Kumar, and P. Arjunan, "FM4EAD: Towards training-free energy anomaly detection with large language models and time-series foundation models," in *Proc. ACM Sustainability Week*, 2026, pp. 433–437, doi: 10.1145/3765611.3815485.

[13] H. Haddad, F. Jerbi, and I. Smaali, "Comparative analysis of off-the-shelf methods for energy consumption anomaly detection," in *Proc. Int. Conf. Automation, Robotics and Applications (ICARA)*, 2024, pp. 392–396, doi: 10.1109/ICARA60736.2024.10553212.
