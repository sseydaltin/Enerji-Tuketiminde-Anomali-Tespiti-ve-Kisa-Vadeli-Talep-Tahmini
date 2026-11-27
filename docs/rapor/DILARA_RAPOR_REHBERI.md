# Final Rapor Rehberi — Dilara için

**Proje:** Enerji Tüketiminde Anomali Tespiti ve Kısa Vadeli Talep Tahmini (YMÜ425)
**Durum:** Kodlama, deneyler, testler, notebook'lar ve demo Şeyda tarafından tamamlandı (`main`, etiket `v3.0-final`). Senin görevin **final raporu sıfırdan, temiz bir şekilde yazdırmak.**
**Araç:** Claude Code (kendi bilgisayarında). Aşağıdaki promptları **sırayla**, her adımın çıktısını okuyup onayladıktan sonra ver.

---

## 0. Başlamadan önce bilmen gerekenler (5 dk oku)

Projede üç veri seti var:

| Veri | Ne için | Etiket | Ana sonuç |
|---|---|---|---|
| **Hane** (UCI IHEPC) | 24 saatlik talep tahmini + anomali | Gerçek etiket yok, sentetik anomali | Tahmin: Seasonal Naive'e göre MAE %26 daha iyi. Anomali olay F1 0,384 (hedef 0,70 karşılanmadı) |
| **Şebeke** (UCI ElectricityLoadDiagrams, ELD) | Toplam yük tahmini + anomali | Sentetik | MAPE %3,24, R² 0,974, olay F1 0,927 |
| **LEAD** (ASHRAE GEP III'ten türetilmiş, 200 bina) | **Gerçek etiketli** anomali tespiti | Gerçek etiket | Final L1 behavior. En iyi nedensel model ROC-AUC 0,954 |

**LEAD neden eklendi?** Hane verisinde gerçek anomali etiketi yok ve sentetik etiketlerle skor yükseltilemedi. Hoca gerçek etiketli bir veri seti eklemeyi önerdi.

**Önemli kavramlar** (raporda doğru kullanılmalı):
- **Kronolojik bölme / bina bölmesi:** Model test verisini ya da test binalarını hiç görmez.
- **Test tek sefer:** Her veri setinde test yalnızca bir kez çalıştırıldı. Seçimler val'de yapıldı. Bu yüzden testte daha iyi çıkan bir model (L1-FE full) olmasına rağmen final model değiştirilmedi. Bu bir **bulgu** olarak yazılacak, hata olarak değil.
- **Alt türler (LEAD):** `gap_edge` (veri boşluğu kenarı), `frozen` (donmuş sayaç), `other` (davranışsal anomali). İlk ikisi kolay, `other` zor.
- **ROC-AUC ve PR-AUC farkı:** Nadir olaylarda PR-AUC çok daha sert bir ölçüttür. Rastgele tahmin PR-AUC'de anomali oranı kadar alır (`other` için ~0,005).

**Kesin kurallar (bunları Claude Code'a da vereceğiz):**
1. Hiçbir sayı elle ya da tahminle yazılmaz. Her sayı repodaki bir dosyadan gelir.
2. Hiçbir model yeniden eğitilmez, test yeniden çalıştırılmaz.
3. Hiçbir kaynak uydurulmaz. DOI'si doğrulanamayan kaynak kullanılmaz.
4. **v2 taban kanalı** ders dışından bir öğretim üyesinin önerisidir. Raporda bu açıkça yazılır.
5. Hocaya sunulan ilk formda planlanan dört aday yöntem (Transformer–K-means, POT eşikleme, nedensel LSTM-AE+POT, tahmin aralığı) **uygulanmadı.** Bunun nedeni (gerçek etiketli veriye öncelik verilmesi) raporda açıkça yazılır.
6. `docs/hocaya_model_sonuclari.md` ve v2 zip içeriği rapora kaynak olarak alınmaz.

---

## 1. Kurulum

```bash
git clone https://github.com/sseydaltin/Enerji-Tuketiminde-Anomali-Tespiti-ve-Kisa-Vadeli-Talep-Tahmini.git
cd Enerji-Tuketiminde-Anomali-Tespiti-ve-Kisa-Vadeli-Talep-Tahmini
git checkout main && git pull
git checkout -b dilara/rapor

python -m venv .venv
# Windows: .venv\Scripts\activate    |  Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest -q          # hepsi geçmeli
```

- Repoya erişimin yoksa Şeyda seni **collaborator** olarak eklemeli.
- Bazı görseller (LEAD bina grafikleri) skor dosyalarını gerektirir. Bunlar git'te yok. Şeyda'nın Drive'a yüklediği zip'i README'deki "Demoyu başka bilgisayarda çalıştırmak için" bölümüne göre aç.
- Hocaya sunulan ilk formu `docs/rapor/oneri_formu_v1_sunulan.docx` olarak koy (Şeyda'dan al). Şablon: `docs/oneri_formu/sablon.docx`.

---

## 2. Prompt 1 — Repoyu tanıma ve rapor planı (yazı yok)

> Bu repoda bir ders projesinin final raporunu yazacağız. Önce CLAUDE.md, README.md ve `docs/` altındaki tüm dosyaları oku: özellikle `final_sonuclar.md`, `hane_vs_sebeke.md`, `lead_sonuclar.md`, `lead_protokol.md`, `lead_val_ozet.md`, `final_test_protokolu.md`, `gelistirme_protokolu.md`, `deney_gunlugu.md`, `literatur.md`, `literatur_taramasi.md`, `savunma_sorulari.md`. Ayrıca `docs/rapor/oneri_formu_v1_sunulan.docx` dosyasını (hocaya sunulan ilk form) ve `docs/oneri_formu/sablon.docx` şablonunu oku.
>
> **Kurallar (rapor boyunca geçerli, CLAUDE.md'ye "Rapor kuralları" başlığıyla ekle):**
> 1. Raporda hiçbir sayı elle yazılmaz. Her sayı `results/` altındaki tablolardan ya da `docs/*_sonuclar.md` dosyalarından betikle alınır ve kaynağı betikte yorum olarak durur.
> 2. Model eğitilmez, test çalıştırılmaz, `results/` değiştirilmez.
> 3. Kaynak uydurulmaz. Kaynakçaya yalnızca DOI'si veya resmi bağlantısı doğrulanmış kaynak girer.
> 4. v2 taban kanalı ders dışından bir öğretim üyesinin önerisidir. Anıldığı her yerde bu belirtilir.
> 5. Test sonrası gözlemler (ör. L1-FE full'ün testte daha iyi olması, A6-dual val-hold sonucu) "test sonrası gözlem" olarak etiketlenir, final sonuç gibi sunulmaz.
> 6. İlk formda planlanan ve uygulanmayan dört aday yöntem, nedeniyle birlikte "Plandan sapmalar" bölümünde açıkça yazılır.
> 7. Abartı yok: "en iyi", "üstün", "başarılı" gibi ifadeler yalnızca sayılarla desteklenirse kullanılır. Hedefi karşılanmayan sonuçlar açıkça "karşılanmadı" diye yazılır.
>
> **Görev:** Henüz yazı yazma. `docs/rapor/rapor_plani.md` dosyasına şunları yaz:
> - Aşağıdaki iskelete göre her bölümün 3–6 maddelik içerik planı ve kullanılacak kaynak dosyalar.
> - Planlanan **görseller** listesi (en az 10): her biri için kaynak dosya ve neyi göstereceği.
> - Planlanan **tablolar** listesi: her biri için kaynak dosya.
> - Repoda bulup emin olamadığın ya da birbiriyle çelişen bilgiler listesi.
>
> İskelet:
> 1. Özet (TR) ve Abstract (EN), anahtar kelimeler
> 2. Giriş: problem, motivasyon, iki soru (ne kadar tüketilecek / ne olağandışı)
> 3. Literatür taraması: talep tahmini, zaman serisi anomali tespiti, enerji anomali tespiti ve LEAD çalışmaları, değerlendirme metrikleri ve tuzakları
> 4. Veri setleri: hane, ELD, LEAD; EDA görselleri; bölme stratejileri
> 5. Yöntem: ön işleme; sızıntı önlemleri; tahmin modelleri; anomali yöntemleri (sentetik enjeksiyon, artık tabanlı, LEAD L0–L5); **her modelin neden seçildiği**; hiperparametre arama; seçim kuralları ve test protokolü
> 6. Deney düzeni ve metrikler: MAE/RMSE/MAPE*, olay F1, ROC-AUC ve PR-AUC farkı, seed'ler, gürültü kuralı
> 7. Sonuçlar: hane tahmini, hane anomali, ELD, LEAD (val→test, alt türler, bina bazlı), literatür kıyası
> 8. Tartışma: **skorlar neden bu düzeyde** (hane anomalisi neden 0,70'e ulaşmadı; ELD neden çok daha yüksek; LEAD'de genel yüksek ama `other` neden düşük; val/test farkı neden oluştu); sentetik ve gerçek etiket karşılaştırması; sinir ağları ile ağaç modellerinin karşılaştırılması
> 9. Sınırlılıklar ve plandan sapmalar
> 10. Sonuç ve gelecek çalışmalar
> 11. Kaynakça
> Ekler: hiperparametre tabloları, demo ekran görüntüleri, yeniden üretim komutları, katkı beyanı
>
> Planı gösterip dur.

**Senin kontrolün:** Planı oku. Çelişki listesi varsa Şeyda'ya sor. Onaylayınca sonraki adıma geç.

---

## 3. Prompt 2 — Görseller ve tablolar

> Onaylanan plana göre `scripts/build_report_assets.py` betiğini yaz. Betik:
> - Tüm görselleri `docs/rapor/figures/` altına PNG (300 dpi) olarak üretsin. Tüm tabloları `docs/rapor/tables/` altına hem CSV hem Markdown olarak üretsin.
> - Yalnızca `results/` ve `data/` okusun, hiçbir şey eğitmesin.
> - Tek bir görsel stili kullansın: sade beyaz arka plan, aynı yazı tipi, Türkçe eksen etiketleri ve birimler. Tüm görsellerde tutarlı renk paleti olsun: aynı model ve aynı alt tür her grafikte aynı renkte. Renk körlüğüne uygun palet seç.
> - Her görsel için `docs/rapor/figures/README.md`'ye bir satırlık açıklama ve kaynak dosya yazsın.
>
> En azından şu görseller olsun: veri setlerinin genel görünümü (her biri için örnek seri), hane günlük/haftalık örüntü, bölme zaman çizelgesi (hane ve ELD), LEAD bina bölmesi şeması, sistem akış diyagramı, tahmin örneği (gerçek ve tahmin, bir hafta), model karşılaştırma çubuk grafiği (hata çubuklarıyla), sentetik anomali türleri örneği, LEAD alt tür örnekleri (her türden bir bina kesiti), LEAD val→test karşılaştırması, LEAD alt tür bazında PR-AUC, ROC ve PR eğrileri (final ve karşılaştırma modelleri), SHAP özet grafiği (varsa kayıtlı çıktıdan), literatür kıyas grafiği.
>
> Betiği çalıştır, görselleri listele ve gösterip dur.

**Senin kontrolün:** Görselleri tek tek aç. Okunmayan, kalabalık ya da çirkin olanı söyle ("şu grafikte yazılar çok küçük" gibi). İstediğin kadar düzelttir.

---

## 4. Prompt 3 — Literatür ve DOI doğrulama

> Literatür taramasını güncelleyeceğiz.
> 1. `docs/literatur_taramasi.md` ve ilk formdaki 9 kaynağı başlangıç listesi olarak al.
> 2. Web'de **2022–2026** arası yeni kaynak ara. Konular: kısa vadeli yük tahmini (LSTM, GRU, Seq2Seq, Transformer, ağaç modelleri), zaman serisi anomali tespiti ve değerlendirme tuzakları, bina enerji anomali tespiti, LEAD / LEAD1.0 / ASHRAE GEP III çalışmaları, Isolation Forest, LightGBM, SHAP. Hakemli dergi ve konferansları öncelikle al. Toplam 20–30 aday hedefle.
> 3. `scripts/verify_dois.py` yaz: her DOI'yi Crossref API'sinden (`https://api.crossref.org/works/<DOI>`) sorgulasın; başlık, ilk yazar, yıl, dergi ve cilt/sayfa bilgisini çeksin ve bizim künyemizle karşılaştırsın. Sonucu `docs/rapor/kaynak_dogrulama.csv` dosyasına yazsın: `doi_bulundu`, `baslik_eslesti`, `yazar_eslesti`, `yil_eslesti` sütunlarıyla. Veri setleri ve DOI'si olmayan kaynaklar (UCI, Kaggle) için resmi URL'yi ve erişim tarihini yaz.
> 4. Uyuşmayan her künyeyi Crossref'teki doğru bilgiyle düzelt. Doğrulanamayanları kaynakçadan çıkar ve ayrı bir listede göster.
> 5. Her kaynak için 2–3 cümlelik özet yaz: ne yaptı, hangi veri, hangi sonuç, bizim projeyle ilişkisi. Özetleri yalnızca okuyabildiğin özetten (abstract) yaz. Okuyamadıysan "yalnız künye" diye işaretle.
> 6. Çıktı: `docs/rapor/kaynakca.md` (IEEE biçiminde) ve `docs/rapor/literatur_ozetleri.md`.
>
> Sonuç tablosunu gösterip dur.

**Senin kontrolün (atlanmaz):** En az **10 kaynağı kendin aç** (DOI linkine tıkla) ve başlığın, yazarların, yılın doğru olduğuna bak. Özellikle şunlar açık kalmıştı: IHEPC veri setinin yılı ve Tech Science makalesinin ("An Integrated Framework of Feature Engineering and Machine Learning for Large-Scale Energy Anomaly Detection", Energy Engineering) yazarları. Savunmada "bu kaynağı okudun mu?" sorusu gelebilir, en az özetini bil.

---

## 5. Prompt 4 — Bölüm bölüm yazım

Bu prompt'u **üç kez** ver: önce 1–4. bölümler, sonra 5–6, sonra 7–10.

> `docs/rapor/rapor_plani.md`'ye göre **[1–4 / 5–6 / 7–10]**. bölümleri `docs/rapor/final_rapor.md` dosyasına yaz.
> Yazım kuralları:
> - Akademik Türkçe, kısa ve net cümleler. Edilgen çatı tutarlı kullanılsın ("…yapılmıştır").
> - Her görsel ve tabloya numaralı başlık ver ve metinde mutlaka atıf yap ("Şekil 3'te görüldüğü gibi…").
> - Sayıları yalnızca `docs/rapor/tables/` ve sonuç dosyalarından al. Her sayının yanında, HTML yorumu olarak kaynak dosyayı belirt (`<!-- kaynak: results/lead/tables/test_results.csv -->`).
> - Kaynaklara IEEE biçiminde atıf yap ([1], [2]). Yalnızca `kaynakca.md`'deki doğrulanmış kaynaklar kullanılsın.
> - **Her model için** neden seçildiğini 2–3 cümleyle yaz: hangi soruyu cevaplıyor, hangi referansa karşı sınanıyor.
> - **Her ana sonuç için** neden bu düzeyde çıktığını yaz. Yalnızca repodaki analizlere dayan (bileşim tablosu, alt tür sonuçları, SHAP, deney günlüğü). Kanıtı olmayan açıklamayı "olası açıklama" diye işaretle.
> - Kıyaslamaları tabloyla ver: referans modellere karşı, sinir ağlarına karşı ağaç modelleri, val ve test, bizim sonuçlarımız ve literatür (koşul farkları sütunuyla).
> - Karşılanmayan hedefleri gizleme.
>
> Bitince bölümleri gösterip dur.

**Senin kontrolün:** Her bölümü sesli okur gibi oku. Anlamadığın cümle varsa, savunmada sen de anlatamazsın: sadeleştirt.

---

## 6. Prompt 5 — Tutarlılık denetimi

> `docs/rapor/final_rapor.md` için bir denetim yap ve `docs/rapor/denetim.md` dosyasına yaz:
> 1. Metindeki **her sayıyı** kaynak dosyasıyla karşılaştır. Uyuşmayanları listele ve düzelt.
> 2. Her [n] atfının kaynakçada karşılığı var mı, kaynakçadaki her kaynak metinde anılıyor mu?
> 3. Her şekil ve tablo metinde anılıyor mu, numaralar sıralı mı?
> 4. Rapor kurallarının 1–7. maddeleri tek tek sağlanıyor mu? (v2 atfı, test sonrası etiketleri, plandan sapmalar, abartı ifadeleri)
> 5. Rapordaki iddialar `docs/savunma_sorulari.md` ile çelişiyor mu?
> 6. Hane, ELD ve LEAD için aynı kavram her yerde aynı adla mı geçiyor (ör. "olay F1" ve "block_f1")?
>
> Düzeltmeleri yap ve denetim özetini gösterip dur.

---

## 7. Prompt 6 — Word belgesine dönüştürme

> `docs/rapor/final_rapor.md`'yi `docs/oneri_formu/sablon.docx` şablonunun biçimiyle (yazı tipi, başlık stilleri, kenar boşlukları) `docs/rapor/final_rapor.docx` olarak üret. Bunun için `scripts/build_report_docx.py` betiğini yaz (pandoc + `--reference-doc` ya da python-docx).
> - Kapak: proje adı, ders, dönem, ekip tablosu.
> - İçindekiler, şekiller listesi, tablolar listesi ve sayfa numaraları olsun.
> - Görseller sayfa genişliğine sığsın, tablolar bölünmesin.
> - HTML kaynak yorumları docx'e geçmesin.
> - Ayrıca `docs/rapor/final_rapor.pdf` üret.
>
> PDF'i sayfa sayfa görüntüye çevirip kontrol et (taşan tablo, bozuk görsel, boş sayfa). Sorunları düzelt, sayfa sayısını söyleyip dur.

---

## 8. Katkı beyanı ve teslim

Raporun sonuna "Ekip katkıları" bölümü eklenecek. Metin Şeyda ile kararlaştırıldı, **aynen kullan:**

> "Proje fikri, kapsamı ve değerlendirme tasarımı ekipçe belirlenmiştir. **Dilara YILDIRIM:** veri hazırlama, kronolojik bölme ve referans modeller; ilk demo uygulaması; literatür taraması ve kaynak doğrulama; final raporun hazırlanması ve sunum. **Şeyda ALTIN:** tahmin ve anomali modellerinin geliştirilmesi; şebeke ve LEAD deneyleri; test değerlendirmeleri; notebook'lar ve demo uygulamasının son hali. Sonuçların yorumlanması ve sözlü savunma ortak sorumluluktur. Taban kanalı (v2) fikri ve ilk uygulaması ders dışından bir öğretim üyesinin önerisidir."

**Savunma için önemli:** Hoca kendi yazdığın kısımlardan soru sorabilir. Veri hazırlama ve bölme, referans modeller, literatür ve raporun tamamını rahatça anlatabilecek düzeyde bil. Diğer kısımlar için `docs/savunma_sorulari.md`'yi oku.

Son adımlar:
```bash
git add docs/rapor scripts/build_report_*.py scripts/verify_dois.py CLAUDE.md
git commit -m "Final rapor"
git push -u origin dilara/rapor
```
GitHub'da `dilara/rapor` → `main` için **Pull Request** aç. Şeyda okuyup onaylasın, sonra birleştirin.

---

## 9. Sunum (rapordan sonra)

> `docs/rapor/final_rapor.md` ve `docs/rapor/figures/` içeriğinden 12–15 slaytlık bir sunum hazırla. Akış: problem ve iki soru → veri setleri → yöntem ve sızıntı önlemleri → hane sonuçları → ELD sonuçları → LEAD (neden eklendi, val→test, alt türler) → literatür kıyası → sınırlılıklar ve dürüst bulgular → demo → sonuç. Slayt başına en fazla 3–4 madde ve tek bir ana görsel olsun. Konuşmacı notlarında her slayt için 3–4 cümlelik anlatım yer alsın. Sayılar yalnızca rapordan alınsın. Çıktı: `docs/sunum/` altında .pptx.

Sunumu Şeyda'yla birlikte prova edin. Demo kısmını Şeyda anlatabilir.

---

## Takıldığında
- Claude Code bir sayı uydurduysa ya da kaynak dosya gösteremiyorsa: "Bu sayının kaynak dosyasını göster, gösteremiyorsan sil" de.
- Bir sonucu anlamadıysan `docs/savunma_sorulari.md`'ye bak. Orada çoğunun sade açıklaması var.
- Teknik bir karar gerekiyorsa (bir modeli rapora alıp almamak gibi) Şeyda'ya sor. Claude Code'a karar verdirme.
