# Final Rapor ve Sunum Rehberi — Dilara için

**Proje:** Enerji Tüketiminde Anomali Tespiti ve Kısa Vadeli Talep Tahmini (YMÜ425)
**Durum:** Kodlama, deneyler, testler, notebook'lar ve demo tamamlandı (`main` dalı, etiket `v3.0-final`).
**Senin görevin:** Final raporu ve PowerPoint sunumunu **Claude Code'a** hazırlatmak. Kurulum dahil her şeyi Claude Code yapacak. Sen her adımda çıktıyı okuyup onaylayacaksın.

**Nasıl kullanılır:** Aşağıdaki promptları **sırayla** Claude Code'a yapıştır. Her promptun sonunda Claude Code durup sana sonucu gösterecek. "Senin kontrolün" kısmını yapmadan sonrakine geçme.

---

## 0. Başlamadan önce bilmen gerekenler (5 dk oku)

Projede üç veri seti var:

| Veri | Ne için | Etiket | Ana sonuç |
|---|---|---|---|
| **Hane** (UCI IHEPC) | 24 saatlik talep tahmini + anomali | Gerçek etiket yok, sentetik anomali | Tahmin: haftalık naif yönteme göre MAE %26 daha iyi. Anomali olay F1 0,384 (hedef 0,70 karşılanmadı) |
| **Şebeke** (UCI ElectricityLoadDiagrams, ELD) | Toplam yük tahmini + anomali | Sentetik | MAPE %3,24, R² 0,974, olay F1 0,927 |
| **LEAD** (ASHRAE GEP III'ten türetilmiş, 200 bina) | **Gerçek etiketli** anomali tespiti | Gerçek | Final model L1 behavior (LightGBM), test ROC-AUC 0,899. Testte en yüksek nedensel değer L5-FE full ROC-AUC 0,956 (test sonrası gözlem) |

**LEAD neden eklendi?** Hane verisinde gerçek anomali etiketi yok ve sentetik etiketlerle anomali skoru yükseltilemedi. Hoca gerçek etiketli bir veri seti eklemeyi önerdi.

**Önemli kavramlar** (raporda doğru kullanılmalı):
- **Kronolojik bölme ve bina bölmesi:** Model test verisini ya da test binalarını hiç görmez.
- **Test tek sefer:** Her veri setinde test yalnızca bir kez çalıştırıldı ve seçimler val'de yapıldı. Testte daha iyi çıkan bir model (L1-FE full) olmasına rağmen final model değiştirilmedi. Bu bir **bulgu** olarak yazılacak, hata olarak değil.
- **LEAD alt türleri:** `gap_edge` (veri boşluğu kenarı), `frozen` (donmuş sayaç), `other` (davranışsal anomali). İlk ikisi kolay, `other` zor.
- **ROC-AUC ve PR-AUC farkı:** Nadir olaylarda PR-AUC çok daha sert bir ölçüttür. Rastgele tahmin PR-AUC'de anomali oranı kadar alır (`other` için ~0,005).

**Kesin kurallar** (Claude Code'a da verilecek):
1. Hiçbir sayı elle ya da tahminle yazılmaz. Her sayı repodaki bir dosyadan gelir.
2. Hiçbir model yeniden eğitilmez, test yeniden çalıştırılmaz.
3. Hiçbir kaynak uydurulmaz. DOI'si doğrulanamayan kaynak kullanılmaz.
4. **v2 taban kanalı** ders dışından bir öğretim üyesinin önerisidir. Raporda bu açıkça yazılır.
5. Hocaya sunulan ilk formda planlanan dört aday yöntem (Transformer–K-means, POT eşikleme, nedensel LSTM-AE+POT, tahmin aralığı) **uygulanmadı.** Neden uygulanmadığı (gerçek etiketli veriye öncelik verildi) raporda açıkça yazılır.
6. `docs/README.md`'de "Rapora kaynak olarak kullanma" başlığı altında listelenen dosyalar, `docs/archive/` klasörü ve v2 içeriği rapora kaynak olarak alınmaz. Eski rapor taslağı da arşive taşındı. Rapor sıfırdan yazılacak.

---

## 1. Senin yapman gereken tek elle kurulum (10 dk)

Repo zaten bilgisayarında var ve collaborator'sün. Ama sendeki kopya eski; son haftalarda yapılan her şey (LEAD, testler, notebook'lar, yeni demo) sende yok. Prompt 0 seni güncel hale getirecek.

Senin yapacakların:
1. Bilgisayarındaki proje klasörünün yerini bil (ör. `Belgeler/Enerji-Tuketiminde-...`). Claude Code'u **o klasörde** aç.
2. **GitHub girişi:** Gerekirse Claude Code `gh auth login` çalıştıracak ve tarayıcıda giriş yapmanı isteyecek. Girişi sen yap. Şifreni Claude Code'a asla yazma.
3. Şeyda'dan hocaya sunulan ilk formu (`Enerji_Proje_Oneri_Formu...docx`) al ve masaüstüne koy.

---

## 2. Prompt 0 — Repoyu güncelleme ve kurulum (Claude Code yapar)

Claude Code'u **mevcut proje klasöründe** aç ve şunu ver:

> Bu klasör bir ders projesinin reposu, ama bendeki kopya eski. Final raporunu ve sunumunu hazırlayacağız. Git ve terminal konusunda deneyimli değilim; her adımı sen yap ve ne yaptığını bana kısaca söyle. **Hiçbir şeyi silme ve zorla üzerine yazma (`reset --hard`, `push --force`, `clean` kullanma).**
> 1. Git, Python (3.11+) ve GitHub CLI (`gh`) kurulu mu kontrol et. Eksik varsa nasıl kurulacağını bana adım adım söyle ve kurulumu bekle.
> 2. `gh auth status` ile GitHub girişimi kontrol et. Giriş yoksa `gh auth login` başlat. Tarayıcıdaki girişi ben yapacağım.
> 3. `git status` ile mevcut durumu göster: hangi daldayım, commit'lenmemiş değişikliğim var mı?
>    - Commit'lenmemiş değişiklik varsa ne olduklarını bana göster ve sor. Önemsizse `git stash` ile kenara al. Önemliyse kendi dalımda commit'le. Silme.
>    - Uzak sunucuya gönderilmemiş eski dallarım varsa listele, dokunma.
>    - **`dilara/v2-taban-entegrasyon` dalında çalışma.** Bu dal arşivlendi (`arsiv/dilara-v2` etiketi) ve uzaktan silindi. Ben şu an bu daldaysam yalnızca `main`'e geç. Dalı silme, içindekileri hiçbir yere taşıma.
> 4. `git fetch --all --tags` çalıştır. `main` dalına geç ve `git pull` ile güncelle. `v3.0-final` etiketinin var olduğunu ve `main`'in bu etiketi içerdiğini doğrula. Son 10 commit'i kısaca göster ki neyin değiştiğini göreyim.
> 5. `main`'den `dilara/rapor` adında yeni bir dal aç. Bu isimde bir dal zaten varsa bana sor.
> 6. Sanal ortamı kontrol et. Yoksa oluştur. Varsa `requirements.txt` değişmiş olabileceği için yeniden kur. Sonra `pytest -q` çalıştır ve hepsinin geçtiğini göster.
> 7. `v3.0-final` GitHub Release'indeki zip'leri indir (`gh release download v3.0-final`). README'deki "Demoyu başka bilgisayarda çalıştırmak için" bölümüne göre doğru yerlere aç.
> 8. Ham veri dosyaları (`data/` altı) bende eksikse README'deki indirme talimatlarını göster. Rapor görselleri için hangilerinin gerektiğini söyle.
> 9. Masaüstümdeki `Enerji_Proje_Oneri_Formu...docx` dosyasını `docs/rapor/oneri_formu_v1_sunulan.docx` olarak kopyala.
> 10. Demoyu `streamlit run` ile bir kez başlat, açıldığını doğrula ve kapat.
> 11. Bana projenin şu anki halini 10 maddede özetle: hangi veri setleri, hangi modeller, ana sonuçlar. Bilgileri `README.md` ve `docs/` dosyalarından al.
>
> Her adımın sonucunu kısaca söyle. Bir adım başarısız olursa dur ve bana sor.

**Senin kontrolün:** Testler geçti mi, demo açıldı mı? Özet, bu rehberin 0. bölümündeki tabloyla uyuşuyor mu? Bir sorun olursa ya da git "conflict" gibi bir hata verirse Claude Code'un mesajını Şeyda'ya ilet.

---

## 3. Prompt 1 — Repoyu tanıma ve rapor planı (henüz yazı yok)

> Bu repoda bir ders projesinin final raporunu yazacağız. Önce `docs/README.md`'yi oku. Hangi dokümanın ne işe yaradığını ve **hangilerinin rapora kaynak olarak kullanılmayacağını** orası söylüyor; bu listeye uy. Sonra CLAUDE.md, README.md ve şu dosyaları oku: `final_sonuclar.md`, `hane_vs_sebeke.md`, `lead_sonuclar.md`, `lead_protokol.md`, `lead_val_ozet.md`, `final_test_protokolu.md`, `gelistirme_protokolu.md`, `deney_gunlugu.md`, `literatur.md`, `literatur_taramasi.md`, `savunma_sorulari.md`, `sunum/kod_haritasi.md`. Ayrıca `docs/rapor/oneri_formu_v1_sunulan.docx` dosyasını (hocaya sunulan ilk form) ve `docs/oneri_formu/sablon.docx` şablonunu oku.
>
> **Kurallar** (rapor boyunca geçerli; CLAUDE.md'ye "Rapor kuralları" başlığıyla ekle):
> 1. Raporda hiçbir sayı elle yazılmaz. Her sayı `results/` altındaki tablolardan ya da `docs/*_sonuclar.md` dosyalarından betikle alınır ve kaynağı betikte yorum olarak durur.
> 2. Model eğitilmez, test çalıştırılmaz, `results/` değiştirilmez.
> 3. Kaynak uydurulmaz. Kaynakçaya yalnızca DOI'si veya resmi bağlantısı doğrulanmış kaynak girer.
> 4. v2 taban kanalı ders dışından bir öğretim üyesinin önerisidir; anıldığı her yerde bu belirtilir.
> 5. Test sonrası gözlemler (ör. L1-FE full'ün testte daha iyi olması, A6-dual val-hold sonucu) "test sonrası gözlem" olarak etiketlenir, final sonuç gibi sunulmaz.
> 6. İlk formda planlanan ve uygulanmayan dört aday yöntem, nedeniyle birlikte "Plandan sapmalar" bölümünde açıkça yazılır.
> 7. Abartı yok. "En iyi", "üstün", "başarılı" gibi ifadeler yalnızca sayılarla desteklenirse kullanılır. Karşılanmayan hedefler açıkça "karşılanmadı" diye yazılır.
> 8. Ders bir **sinir ağları** dersidir. Sinir ağı mimarileri (Seq2Seq-LSTM, LSTM, GRU, LSTM-Autoencoder) ayrıntılı ve tensör boyutlarıyla anlatılır. LEAD'de finalin neden LightGBM olduğu, sinir ağı alternatiflerinin sonuçlarıyla birlikte açıklanır.
>
> **Görev:** Henüz yazı yazma. `docs/rapor/rapor_plani.md` dosyasına şunları yaz:
> - Aşağıdaki iskelete göre her bölüm için 3–6 maddelik içerik planı ve kullanılacak kaynak dosyalar.
> - Planlanan **görseller** (en az 12): her biri için kaynak dosya ve ne göstereceği.
> - Planlanan **tablolar**: her biri için kaynak dosya.
> - Repoda bulup emin olamadığın ya da birbiriyle çelişen bilgiler.
>
> İskelet:
> 1. Özet (TR) ve Abstract (EN), anahtar kelimeler
> 2. Giriş: problem, motivasyon ve iki soru (ne kadar tüketilecek / ne olağandışı)
> 3. Literatür taraması: talep tahmini, zaman serisi anomali tespiti, enerji anomali tespiti ve LEAD çalışmaları, değerlendirme metrikleri ve tuzakları
> 4. Veri setleri: hane, ELD ve LEAD; EDA görselleri; bölme stratejileri
> 5. Yöntem: ön işleme; sızıntı önlemleri; **sinir ağı mimarileri**; diğer modeller; anomali yöntemleri (sentetik enjeksiyon, artık tabanlı, LEAD L0–L5); **her modelin neden seçildiği**; hiperparametre arama; seçim kuralları ve test protokolü
> 6. Deney düzeni ve metrikler: MAE, RMSE, MAPE*, olay F1, ROC-AUC ve PR-AUC farkı, seed'ler, gürültü kuralı
> 7. Sonuçlar: hane tahmini, hane anomali, ELD, LEAD (val→test, alt türler, bina bazlı), literatür kıyası
> 8. Tartışma: **skorlar neden bu düzeyde** (hane anomalisi neden 0,70'e ulaşmadı; ELD neden çok daha yüksek; LEAD'de genel skor yüksekken `other` neden düşük; val/test farkı neden oluştu); sentetik ve gerçek etiket; sinir ağları ve ağaç modelleri
> 9. Sınırlılıklar ve plandan sapmalar
> 10. Sonuç ve gelecek çalışmalar
> 11. Kaynakça
>
> Ekler: hiperparametre tabloları, demo ekran görüntüleri, yeniden üretim komutları, ekip katkıları
>
> Planı gösterip dur.

**Senin kontrolün:** Planı oku. Çelişki listesi varsa Şeyda'ya sor. Onaylayınca devam et.

---

## 4. Prompt 2 — Görseller ve tablolar

> Onaylanan plana göre `scripts/build_report_assets.py` betiğini yaz. Betik:
> - Tüm görselleri `docs/rapor/figures/` altına PNG (300 dpi) olarak, tüm tabloları `docs/rapor/tables/` altına CSV ve Markdown olarak üretsin.
> - Yalnızca `results/` ve `data/` okusun, hiçbir şey eğitmesin.
> - Tek bir görsel stil kullansın: sade beyaz arka plan, aynı yazı tipi, Türkçe eksen etiketleri ve birimler. Renk paleti renk körlüğüne uygun olsun ve aynı model ya da aynı alt tür her grafikte aynı renkte görünsün.
> - Her görsel için `docs/rapor/figures/README.md`'ye bir satırlık açıklama ve kaynak dosya yazsın.
>
> En azından şu görseller olsun:
> - **Sinir ağı mimari diyagramları** (tensör boyutlarıyla): Seq2Seq-LSTM, LSTM, GRU (eksik veri maskeli, LEAD L2), LSTM-Autoencoder. Ayrıca artık tabanlı anomali skorunun akışı (tahmin → artık → normalizasyon → eşik). Boyutlar config'ten ve koddan okunsun.
> - Sistem akış diyagramı
> - Her veri seti için örnek seri; hane günlük/haftalık örüntü
> - Bölme zaman çizelgesi (hane ve ELD), LEAD bina bölmesi şeması
> - Tahmin örneği (gerçek ve tahmin, bir hafta), hata çubuklu model karşılaştırma grafiği
> - Sentetik anomali türleri örneği, LEAD alt tür örnekleri
> - LEAD val→test karşılaştırması, alt tür bazında PR-AUC, ROC ve PR eğrileri
> - Sinir ağları ve diğer modeller karşılaştırması (her veri setinde en iyi sinir ağı ile en iyi diğer model)
> - SHAP özet grafiği (varsa kayıtlı çıktıdan), literatür kıyas grafiği
>
> Betiği çalıştır, görselleri listele ve gösterip dur.

**Senin kontrolün:** Görselleri tek tek aç. Okunmayan, kalabalık ya da çirkin olanı söyle ("şu grafikte yazılar çok küçük" gibi) ve düzelttir.

---

## 5. Prompt 3 — Literatür ve DOI doğrulama

> Literatür taramasını güncelleyeceğiz.
> 1. `docs/literatur_taramasi.md` ve ilk formdaki 9 kaynağı başlangıç listesi olarak al.
> 2. Web'de **2022–2026** arası yeni kaynak ara. Konular: kısa vadeli yük tahmini (LSTM, GRU, Seq2Seq, Transformer, ağaç modelleri), zaman serisi anomali tespiti ve değerlendirme tuzakları, bina enerji anomali tespiti, LEAD / LEAD1.0 / ASHRAE GEP III çalışmaları, Isolation Forest, LightGBM, SHAP. Hakemli dergi ve konferanslara öncelik ver. Toplam 20–30 aday hedefle.
> 3. `scripts/verify_dois.py` yaz. Betik her DOI'yi Crossref API'sinden (`https://api.crossref.org/works/<DOI>`) sorgulasın; başlık, ilk yazar, yıl, dergi ve cilt/sayfa bilgilerini çekip bizim künyemizle karşılaştırsın. Sonucu `docs/rapor/kaynak_dogrulama.csv` dosyasına `doi_bulundu`, `baslik_eslesti`, `yazar_eslesti`, `yil_eslesti` sütunlarıyla yazsın. Veri setleri ve DOI'si olmayan kaynaklar (UCI, Kaggle) için resmi URL'yi ve erişim tarihini yaz.
> 4. Uyuşmayan künyeleri Crossref'teki doğru bilgiyle düzelt. Doğrulanamayanları kaynakçadan çıkar ve ayrı bir listede göster.
> 5. Her kaynak için 2–3 cümlelik özet yaz: ne yaptı, hangi veriyi kullandı, hangi sonucu buldu, bizim projeyle ilişkisi ne. Özetleri yalnızca okuyabildiğin abstract'tan yaz. Okuyamadıysan "yalnız künye" diye işaretle.
> 6. Çıktılar: `docs/rapor/kaynakca.md` (IEEE biçiminde) ve `docs/rapor/literatur_ozetleri.md`.
>
> Sonuç tablosunu gösterip dur.

**Senin kontrolün (atlanmaz):** En az **10 kaynağı kendin aç** (DOI linkine tıkla). Başlığın, yazarların ve yılın doğru olduğuna bak. Açık kalmış iki kaynak var: IHEPC veri setinin yılı ve Tech Science makalesinin ("An Integrated Framework of Feature Engineering and Machine Learning for Large-Scale Energy Anomaly Detection", Energy Engineering) yazarları. Savunmada "bu kaynağı okudun mu?" sorusu gelebilir, en azından özetini bil.

---

## 6. Prompt 4 — Bölüm bölüm yazım

Bu prompt'u **üç kez** ver: önce 1–4. bölümler, sonra 5–6, sonra 7–10.

> `docs/rapor/rapor_plani.md`'ye göre **[1–4 / 5–6 / 7–10]**. bölümleri `docs/rapor/final_rapor.md` dosyasına yaz.
>
> Yazım kuralları:
> - Akademik Türkçe kullan; cümleler kısa ve net olsun. Edilgen çatıyı tutarlı kullan ("…yapılmıştır").
> - Her görsele ve tabloya numaralı başlık ver ve metinde mutlaka atıf yap ("Şekil 3'te görüldüğü gibi…").
> - Sayıları yalnızca `docs/rapor/tables/` ve sonuç dosyalarından al. Her sayının yanına kaynak dosyayı HTML yorumu olarak koy (`<!-- kaynak: results/lead/tables/test_results.csv -->`).
> - Kaynaklara IEEE biçiminde atıf yap ([1], [2]). Yalnızca `kaynakca.md`'deki doğrulanmış kaynakları kullan.
> - **Her model için** neden seçildiğini 2–3 cümleyle yaz: hangi soruyu cevaplıyor, hangi referansa karşı sınanıyor.
> - **Her ana sonuç için** neden bu düzeyde çıktığını yaz. Yalnızca repodaki analizlere dayan (bileşim tablosu, alt tür sonuçları, SHAP, deney günlüğü). Kanıtı olmayan açıklamayı "olası açıklama" diye işaretle.
> - Kıyaslamaları tabloyla ver: referans modellere karşı, sinir ağları ve ağaç modelleri, val ve test, bizim sonuçlarımız ve literatür (koşul farkları sütunuyla).
> - Karşılanmayan hedefleri gizleme.
>
> Bitince bölümleri gösterip dur.

**Senin kontrolün:** Her bölümü sesli okur gibi oku. Anlamadığın cümleyi sadeleştirt. Sen anlamıyorsan savunmada da anlatamazsın.

---

## 7. Prompt 5 — Tutarlılık denetimi

> `docs/rapor/final_rapor.md` için bir denetim yap ve sonucu `docs/rapor/denetim.md` dosyasına yaz:
> 1. Metindeki **her sayıyı** kaynak dosyasıyla karşılaştır. Uyuşmayanları listele ve düzelt.
> 2. Her [n] atfının kaynakçada karşılığı var mı? Kaynakçadaki her kaynak metinde anılıyor mu?
> 3. Her şekil ve tablo metinde anılıyor mu, numaralar sıralı mı?
> 4. Rapor kurallarının 1–8. maddeleri tek tek sağlanıyor mu?
> 5. Rapordaki iddialar `docs/savunma_sorulari.md` ile çelişiyor mu?
> 6. Aynı kavram her yerde aynı adla mı geçiyor (ör. "olay F1" ve "block_f1")?
>
> Düzeltmeleri yap ve denetim özetini gösterip dur.

---

## 8. Prompt 6 — Word belgesi

> `docs/rapor/final_rapor.md`'yi `docs/oneri_formu/sablon.docx` şablonunun biçimiyle (yazı tipi, başlık stilleri, kenar boşlukları) `docs/rapor/final_rapor.docx` olarak üret. Bunun için `scripts/build_report_docx.py` yaz (pandoc + `--reference-doc` ya da python-docx).
> - Kapak: proje adı, ders, dönem ve ekip tablosu.
> - İçindekiler, şekiller listesi, tablolar listesi ve sayfa numaraları olsun.
> - Görseller sayfa genişliğine sığsın, tablolar sayfa arasında bölünmesin.
> - HTML kaynak yorumları docx'e geçmesin.
> - Ayrıca `docs/rapor/final_rapor.pdf` üret.
>
> PDF'i sayfa sayfa görüntüye çevirip kontrol et (taşan tablo, bozuk görsel, boş sayfa). Sorunları düzelt, sayfa sayısını söyleyip dur.

---

## 9. Ekip katkıları (rapora aynen girecek)

Metin Şeyda ile kararlaştırıldı:

> "Proje fikri, kapsamı ve değerlendirme tasarımı ekipçe belirlenmiştir. **Dilara YILDIRIM:** veri hazırlama, kronolojik bölme ve referans modeller; ilk demo uygulaması; literatür taraması ve kaynak doğrulama; final raporun hazırlanması ve sunum. **Şeyda ALTIN:** tahmin ve anomali modellerinin geliştirilmesi; şebeke ve LEAD deneyleri; test değerlendirmeleri; notebook'lar ve demo uygulamasının son hali. Sonuçların yorumlanması ve sözlü savunma ortak sorumluluktur. Taban kanalı (v2) fikri ve ilk uygulaması ders dışından bir öğretim üyesinin önerisidir."

---

## 10. Prompt 7 — PowerPoint sunumu

**Hocanın değerlendirme kriterleri** (%25):
1. Problemi ve amacı açıklama
2. Veri setini tanıtma
3. **Yöntemi ve sinir ağı mimarisini açıklama**
4. Deneyleri ve sonuçları sunma
5. **Sistemi çalışır durumda gösterme (canlı demo, zorunlu)**
6. Sonuçları yorumlama
7. Teknik sorulara cevap verme. **Koda hâkimiyet her öğrenci için ayrı değerlendirilir.**

> `docs/rapor/final_rapor.md` ve `docs/rapor/figures/` içeriğinden bir **PowerPoint (.pptx)** sunumu hazırla. Slaytlar hocanın kriterlerine birebir karşılık gelsin:
> 1. Problem ve amaç: iki soru (ne kadar tüketilecek / ne olağandışı). (1 slayt)
> 2. Veri setleri: hane, ELD, LEAD ve LEAD'in neden eklendiği. (2 slayt)
> 3. **Sinir ağı mimarileri:** Seq2Seq-LSTM diyagramı merkezde (kodlayıcı, çözücü, gelecek takvim girdisi, 48 saat → 24 saat). LSTM, GRU ve LSTM-Autoencoder kısaca. Sızıntı önlemleri ve test protokolü. (3 slayt)
> 4. Deneyler ve sonuçlar: hane tahmini, hane anomali, ELD, LEAD val→test. (3–4 slayt)
> 5. **Canlı demo:** tek slayt, üzerinde şu üç adım: (a) Talep Tahmini — model, gerçek tüketim ve basit yöntem; (b) Anomali Tespiti — yapay anomalinin yakalanması; (c) Gerçek Etiketli Veri — gerçek etiketler ve model alarmları. Bu slaytta tarayıcıya geçilecek. Sunumun **sonuna gizli yedek slaytlar** ekle: `docs/sunum/yedek/` altındaki her ekran görüntüsü için bir slayt. Demo çökerse bunlarla devam edilecek.
> 6. Yorum: skorlar neden bu düzeyde; LEAD finalinin neden LightGBM olduğu ve sinir ağı alternatiflerinin (GRU) sonuçları; test sonrası gözlem; sınırlılıklar. (2 slayt)
> 7. Sonuç ve gelecek çalışmalar. (1 slayt)
>
> Slayt başına en fazla 3–4 madde ve tek bir ana görsel olsun. Konuşmacı notlarında her slayt için 3–4 cümlelik anlatım yaz ve her slaytın notunda **kimin anlatacağını** belirt (dağılım aşağıda). Sayılar yalnızca rapordan alınsın. Sade ve tutarlı bir tasarım kullan: tek yazı tipi, rapordaki görsellerle aynı renk paleti, slayt numaraları. Çıktı: `docs/sunum/sunum.pptx` ve kontrol için .pdf. PDF'i sayfa sayfa görüntüye çevirip taşan metin, küçük yazı ya da bozuk görsel olup olmadığını kontrol et.
>
> Konuşmacı dağılımı: Dilara — problem, veri setleri, veri hazırlama ve bölme, sonuçların bir kısmı. Şeyda — mimariler, LEAD, canlı demo. Yorum ve sonuç ortak.

**Demoyu Şeyda yapacak.** Üç ekranı, hiçbir ayara dokunmadan varsayılan halleriyle gösterecek.

---

## 11. Soru-cevap hazırlığı (atlanmaz)

Hoca sana kod sorabilir. En azından şunları kodda gösterip anlatabilmelisin:
- Ham dakikalık veriden saatlik veriye geçiş ve eksik değer işlemi
- Kronolojik bölme ve ölçekleyicinin neden yalnızca eğitim verisine uydurulduğu
- Referans modeller (Seasonal Naive, Ridge)
- Seq2Seq-LSTM'nin girdisi ve çıktısı (boyutlar dahil)

Claude Code'a şunu ver:
> `docs/sunum/kod_haritasi.md`'deki "Dilara anlatacak" bölümünü ve `notebooks/01_hane.ipynb`'yi bana hücre hücre, sade bir dille açıkla. Her adımdan sonra bana bir soru sor ve cevabımı değerlendir.

Ayrıca `docs/savunma_sorulari.md`'yi baştan sona oku.

Sunumu Şeyda'yla **en az iki kez** prova edin. Provalardan birinde demoyu bilerek "çökmüş" sayıp yedek slaytlarla devam etmeyi deneyin.

---

## 12. Teslim

Claude Code'a şunu ver:
> Rapor ve sunum dosyalarını, yeni betikleri ve CLAUDE.md'yi commit'le (mesaj: "Final rapor ve sunum"). `dilara/rapor` dalını push et ve `main`'e bir Pull Request aç. PR açıklamasına neyin eklendiğini özetle.

Şeyda PR'ı okuyup onaylayacak, sonra birleştireceksiniz.

---

## Takıldığında
- Claude Code bir sayı uydurduysa ya da kaynak dosyasını gösteremiyorsa: "Bu sayının kaynak dosyasını göster, gösteremiyorsan sil."
- Bir sonucu anlamadıysan `docs/savunma_sorulari.md`'ye bak. Orada çoğunun sade açıklaması var.
- Teknik bir karar gerekiyorsa (bir modeli rapora alıp almamak gibi) Şeyda'ya sor. Bu kararı Claude Code'a verdirme.
- Git ile ilgili bir hata alırsan ("conflict", "rejected", "permission denied") hiçbir şeyi zorlamadan Claude Code'un mesajını Şeyda'ya ilet.
