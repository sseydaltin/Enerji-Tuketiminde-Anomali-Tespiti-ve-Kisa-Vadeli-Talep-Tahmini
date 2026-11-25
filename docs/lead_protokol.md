# LEAD1.0 protokolü — model eğitiminden ve testten ÖNCE yazıldı

Dal: `seyda/lead` · Tarih: 2026-11-25 · Durum: **sabitlendi, henüz uygulanmadı** (L2 ve sonrası). Güncelleme 1 (2026-11-25, uygulamadan önce): site bilgisi, mutlak tarih yasağı, siteye göre bölme, paylaşılan/tekil dilimi, hesaplama bütçesi.

Bu belge, LEAD1.0 (Kaggle "energy-anomaly-detection" `train.csv`) üzerindeki modelleme ve değerlendirmenin kurallarını **deney yapılmadan önce** sabitler. Bu belgeye aykırı bir seçim yapılırsa sonuç geçersiz sayılır. Kural değişikliği gerekirse tarih ve gerekçeyle `docs/deney_gunlugu.md`'ye yazılır; eski sonuçlar silinmez.

- Hane (UCI IHEPC) ve şebeke (ELD) sonuçlarına **dokunulmaz**. LEAD çıktıları ayrı yerlerde tutulur: `configs/lead.yaml`, `results/lead/`, `models/lead/`, `results/lead/experiments.csv`.
- LEAD projeye **ek doğrulama veri seti** olarak girer: gerçek, elle konmuş etiketlerle değerlendirme (`docs/veri_seti_lead1_degerlendirme.md`).
- Temel: `docs/veri_seti_lead1_degerlendirme.md` §6'daki taslak. Değişiklikler bu belgededir.

## 1. Veri ve L1 bulguları (özet)

- Yalnız Kaggle `train.csv`: 200 bina, 2016, saatlik `meter_reading` (kWh), elle konmuş 0/1 etiket. `test.csv` etiketsiz olduğu için kullanılmaz.
- **Temizlik** (`src/datasets/lead/load.py`, `experiments/archive/lead_colab_ilk.ipynb` ile birebir): 1.749.494 → 1.641.841 satır. Yalnız eksik tüketimli 107.653 satır düştü. Eksik saatler doldurulmaz; zaman ızgarasında boşluk olarak kalır.
- **Anomali oranı:** saatlerin %2,27'si. Olaylar ve yapılar:
  - 5.811 olay var. Olayların %47,8'i tek saatlik.
  - Tek saatlik olayların %99,1'i eksik bir saate komşu.
  - Anomali saatlerinin %65,3'ü sabit okumanın içinde.
  - Olayların %74'ü en az bir başka binada **aynı saatte** başlıyor; bunların %79,6'sı boşluk kenarında.
- Kaynak tablolar: `results/lead/tables/` (`dataset_summary.csv`, `gap_adjacency.csv`, `causal_vs_twosided.csv`, `same_start_events.csv`). Ayrıntı: `docs/deney_gunlugu.md` (2026-11-25, LEAD L1 satırları).
- **Site bilgisi (güncelleme 1):** Kaggle `train_features.csv`'den **yalnız** `site_id` ve `primary_use` alındı (`--stage sites` → `data/raw/lead/building_sites.csv`). 200 / 200 bina eşleşti, 14 site. Dosyanın diğer sütunları model girdisi değildir; sütun listesi ve geleceğe bakıyor olabilecekler günlükte. Paylaşılan olayların %70,1'inde aynı saatte başlayan bütün binalar tek sitede; %89,3'ünde en az bir başka bina aynı sitede.

## 2. Ana bilimsel soru

Hedef, gerçek etiketli bina verisinde **davranışsal** anomalileri yakalamaktır. Bunlar veri boşluğu kenarı ya da takılı kalan sayaç okuması olmayan anomalilerdir (aşağıdaki `other` alt türü). Bu kümede:
- Denetimli bir sinir ağı (L2 GRU) basit kurallardan ve ağaç modelinden daha iyi mi?
- Etiketsiz tahmin artığı yöntemi (L4) ne kadar yakalıyor?

Genel (tüm etiketler) sonuçlar Kaggle ve literatürle kıyas için raporlanır. Ancak **yorumun merkezi `other` alt kümesidir.**

## 3. Etiket alt türleri (YALNIZ değerlendirme için; model girdisi değildir)

Her **anomali saati** aşağıdaki öncelik sırasıyla tek bir alt türe atanır. Tanımlar değerlendirme etiketidir. Gerçek etiketin betimlenmesi olduğu için iki yönlüdür ve sonraki saatlere bakabilir. Hiçbir model veya kural bu sütunları görmez.

| Öncelik | Alt tür | Tanım | L1'deki payı (anomali saatleri) |
|---|---|---|---|
| 1 | `gap_edge` | Aynı binada bir önceki **veya** bir sonraki saat eksik (verinin genel ilk/son saati hariç; `eda.gap_neighbors`) | 4.587 saat · %12,3 |
| 2 | `frozen` | `gap_edge` değil ve saat, **≥ 3 saatlik** sabit değer serisinin içinde (aynı bina, boşluksuz, birebir aynı `meter_reading`; `eda.constant_run_length ≥ 3`) | 23.199 saat · %62,2 |
| 3 | `other` | Geri kalan anomali saatleri (davranışsal) | 9.510 saat · %25,5 |

- **Olay alt türü:** olayın saatlerinin çoğunluk alt türü. Eşitlikte yukarıdaki öncelik sırası uygulanır.
- **Alt tür bazında metrik:** pozitifler yalnız o alt türün anomali saatleridir. Negatifler **bütün normal saatlerdir**. Diğer alt türlerin anomali saatleri o hesaptan çıkarılır. Böylece her alt türde ROC-AUC, PR-AUC ve P/R/F1 ayrı hesaplanır. Olay recall'ü alt türün olaylarıyla hesaplanır.
- **Paylaşılan / tekil dilimi (güncelleme 1):** bir anomali olayı, **başka bir binada aynı saatte başlayan bir olay varsa "paylaşılan"**, yoksa "tekil"dir (`eda.same_start_events`; L1'de olayların %74'ü paylaşılan). Bir anomali saati, ait olduğu olayın dilimini alır. Bu da yalnız değerlendirme etiketidir; model girdisi değildir. Dilim metrikleri alt türlerle aynı kuralla hesaplanır: pozitifler o dilimin anomali saatleri, negatifler bütün normal saatler.
- **Raporlama:** bütün metrikler **genel**, **`gap_edge` / `frozen` / `other`** ve **paylaşılan / tekil** olarak ayrı ayrı raporlanır.

## 4. Nedensellik

- **Bütün öznitelikler ve kurallar yalnız geçmişe bakar.** t saatindeki skor, yalnızca t ve öncesindeki okumalar, eksiklik bilgisi ve takvimle hesaplanır.
- Bina başına normalizasyon da yalnız geçmişten yapılır: bir binanın tüm yılına ait istatistik kullanılmaz.
- **Mutlak tarih öznitelikleri yasaktır (güncelleme 1):** yılın günü, tarih sırası (örneğin yılın başından beri geçen saat), zaman damgasının kendisi veya ondan türetilen sıra/yıl bilgisi kullanılmaz. Yalnız **döngüsel takvim** kullanılır: saat, haftanın günü ve ay (sin/cos), hafta sonu. Neden: tek yıllık veride mutlak tarih, ortak veri olaylarının tarihini ezberlemeye (örneğin 50 binada aynı saatte başlayan olaylar) izin verir. Bu kural, öznitelik listelerini kontrol eden bir testle güvenceye alınır (§15).
- Sonraki saatleri kullanan bir varyant denenirse (ör. iki yönlü boşluk kenarı veya merkezli pencere) adı **"offline"** ile etiketlenir, ayrı satırda raporlanır ve **ana sonuç olmaz**.
- L1'deki karşılaştırma (`causal_vs_twosided.csv`):
  - Nedensel boşluk kenarı (önceki saat eksik) anomali saatlerinin %6,4'ünü işaretliyor; iki yönlü tanım %12,3'ünü.
  - Nedensel sabitlik (son 3 saat aynı) %57,6'sını işaretliyor; iki yönlü tanım %65,3'ünü.

## 5. Bölme

- **Ana bölme: binaya göre**, 3 tekrar (bölme seed'leri 42, 7, 2026).
  - Her tekrarda binalar train / val / test olarak ayrılır: 120 / 40 / 40.
  - Oranlar Colab denemesiyle aynı: önce %20 test, kalanın %25'i val.
  - Bina listeleri `results/lead/tables/splits_seed{s}.csv` dosyasına yazılır ve değişmez.
  - Her binada en az bir anomali olduğu için her parçada iki sınıf da bulunur; bu bir testle denetlenir.
- **Ek bölme: siteye göre gruplu (güncelleme 1).** Bu bir sağlamlık kontrolüdür. 3 tekrar (seed 42, 7, 2026): siteler karıştırılır ve train / val / test'e, bina sayıları yaklaşık %60 / %20 / %20 olacak şekilde bütün olarak atanır (bir sitenin bütün binaları tek parçadadır). Her parçada iki sınıfın da bulunduğu denetlenir. Siteler çok dengesiz olduğu için (site 9: 52 bina, site 14: 47 bina, üç sitede tek bina) parça büyüklükleri tekrardan tekrara değişir; gerçekleşen sayılar raporlanır. Yalnız **kurallar (L0) ve val'de seçilen en iyi 2 model** için yapılır; ayarlar ve eşik bina bölmesinin aynı seed'inden değil, bu bölmenin kendi val'inden seçilir (Optuna tekrarlanmaz, bina bölmesinde bulunan ayarlar kullanılır).
- Pencereler bina sınırını geçmez. Bir binanın bütün saatleri aynı parçadadır.

## 6. Ön işleme ve sızıntı önlemleri

- `log1p(meter_reading)`. Ölçekleyiciler (StandardScaler vb.) **yalnız train binalarına** fit edilir; val ve test binalarına yalnız `transform` uygulanır.
- Binalar arası ölçek farkı yaklaşık 4 büyüklük mertebesi. Bu yüzden derin modellerde pencere içi göreli ölçek de verilir: pencerenin son 48 saatinin medyanına göre. Bu ölçek yalnız geçmişten hesaplanır.
- **Eksik değerler:** GRU ve CNN girdisinde bina bazında son geçerli değerle doldurulur (ileri doldurma; yalnız geçmiş). Ayrıca 0/1 **eksiklik maskesi kanalı** verilir. LightGBM'de eksik gecikmeler NaN kalır (Colab'daki gibi).
- Optuna, early stopping, eşik ve topluluk ağırlığı hiçbir zaman test binalarına bakmaz (§9–§11).

## 7. Öznitelik setleri

| Set | İçerik |
|---|---|
| **full** | Colab öznitelikleri: `meter_reading`, `log_reading`, `is_zero`, `hour`, `day_of_week`, `month`, `is_weekend`, `hour_sin/cos`, `lag_{1,24,168}h`, `log_diff_{1,24,168}h`, `lag_{1,24,168}h_missing` + **eksiklik bayrakları** (önceki saat eksik = R1 sinyali) + **geriye dönük sabitlik süresi** (`eda.backward_run_length`, R2 sinyali). Derin modellerde ek olarak eksiklik maskesi kanalı. |
| **behavior** | **Eksiklik ve sabitlikle ilgili tüm öznitelikler hariç:** `lag_*_missing`, eksiklik bayrakları, geriye dönük sabitlik süresi, `is_zero`, `log_diff_1h` (değeri 0 ise sabitliği doğrudan gösterir) ve derin modellerde maske kanalı. Eksik gecikmeler NaN bırakılmaz; ileri doldurulmuş seriden hesaplanır (yalnız geçmiş). Böylece NaN üzerinden eksiklik bilgisi sızmaz. |

- **Önceden kabul edilen sınırlılık — kısayol:** behavior setinde eksiklik ve sabitlik bilgisine giden **kısayollar** tamamen kapatılamaz. İleri doldurma boşlukları yapay sabit serilere çevirir; ham okuma ile `lag_1h`'nin eşitliği de dolaylı bilgi taşır.
- **Ölçüm:** behavior modellerinin `gap_edge` ve `frozen` alt kümelerindeki recall'ü raporlanır. Bu değer yüksekse model bir kısayol kullanıyor demektir ve böyle yorumlanır.
- L1 (LightGBM), L2 (GRU) ve L3 (CNN+GRU) **iki setle de** eğitilir.

## 8. Modeller

| Kod | Model | Tür | Ayrıntı |
|---|---|---|---|
| **L0** | Referanslar | — | (a) her şeye "normal" diyen model; (b) **R1** önceki saat eksik; (c) **R2** son k saat aynı değer, k ∈ {2, 3, 6} val'de F1 ile seçilir; (d) **R3** = R1 VEYA R2; (e) **A0** hareketli z-skoru, yalnız geçmişe bakan, hanedeki tanımla aynı. Eşik val'de seçilir. |
| **L1** | LightGBM | denetimli | Colab öznitelikleri; full ve behavior setleri; `scale_pos_weight` = train negatif/pozitif. |
| **L2** | GRU sınıflandırıcı | denetimli · **ANA SİNİR AĞI** | Girdi son 48 saat [48, F]: log-ölçekli okuma, pencere-göreli okuma, takvim, eksiklik maskesi (full setinde). Çıktı: t saatinin anomali olasılığı. Ağırlıklı BCE (`pos_weight` = train negatif/pozitif). full ve behavior setleri. |
| **L3** | 1D-CNN + GRU | denetimli | L2 ile aynı girdi; Conv1d → ReLU → GRU → doğrusal katman. Mimari karşılaştırması. |
| **L4** | A3 tahmin artığı | etiketsiz | Seq2Seq tahmin modeli (hanedeki F10 yapısı) **train binalarının tüm saatleriyle** eğitilir. Etiket kullanılmaz; eğitim verisinde anomali bulunması (kontaminasyon) sınırlılık olarak not edilir. Skor: 1 adım ileri mutlak hata, bina bazında **yalnız geçmişten** (son 168 saatin artık ölçeği) normalize edilir. Eşik val etiketleriyle seçilir. |
| **L4-semi** | A3, yarı denetimli | yarı denetimli | L4 ile aynı; train binalarının etiketli anomali saatleri eğitim hedeflerinden çıkarılır. Ayrı etiketlenir ve L4 ile karıştırılmaz. |
| **L5** | Topluluk | denetimli | L1 + L2 olasılıklarının ağırlıklı ortalaması; ağırlık {0; 0,1; …; 1} ızgarasından val PR-AUC ile seçilir. Aynı öznitelik setinin üyeleri birleştirilir. |
| Ek | Isolation Forest | etiketsiz | Yalnız literatür referansı: LEAD1.0 veri makalesindeki PyOD sonuçlarıyla (nokta F1 ≈ 0,41) kıyas için. Train binalarında fit edilir; eşik val'de seçilir; seçim yarışına girmez. |

**LSTM-AE kullanılmayacak (karar, 2026-11-25).** Hanedeki kayıtlı sonuçlarda LSTM-AE (A2) olay F1'de yöntemlerin gerisinde kaldı:
- Test ana senaryoda olay F1 0,352; bu, yöntemlerin en düşüğü. Karşılaştırma için A0'da 0,398, final yöntemde 0,384.
- Val-dev zor senaryoda olay F1 0,141; A0'da 0,302.

Kaynak: `results/tables/anomaly_protocol_test.csv` ve `anomaly_protocol_val.csv`. Not: aynı tabloda A2'nin ROC-AUC'si (test ana 0,856) ve PR-AUC'si (0,329) rastgeleden belirgin biçimde yüksektir; dışlama gerekçesi "rastgele düzey" değil, olay düzeyindeki zayıflıktır. Karar günlüğe yazıldı.

## 9. Hiperparametre ayarı

- **Optuna yalnız L1, L2 ve L3 için**, yalnız **train binalarında**, binaya göre gruplu 3 katlı CV ile yapılır (`GroupKFold`).
- Model başına 20 deneme. Amaç: katların ortalama PR-AUC'si.
- Val ve test binaları Optuna'da hiç kullanılmaz.
- Seçilen ayarlarla model bütün train binalarında yeniden eğitilir. Early stopping (L1: ağaç sayısı; L2 ve L3: epoch, patience 8) **val binalarında PR-AUC** ile yapılır.
- L4, hanedeki F10-opt ayarlarıyla eğitilir; yeni arama yapılmaz. Early stopping val binalarının tahmin MAE'si ile yapılır.
- **Hesaplama bütçesi (güncelleme 1):**
  - Optuna sırasında eğitim örneklerinde **negatif alt örnekleme**: bütün pozitifler + negatiflerin %20'si (seed'li, kat başına sabit). Katların değerlendirmesi ve bütün val/test değerlendirmeleri **tam veriyle** yapılır.
  - Derin modellerde deneme başına epoch sınırı ve Optuna **MedianPruner**.
  - Toplam GPU süresi **4 saati** aşacaksa deneme sayısı düşürülür; bu karar ve gerçekleşen süreler günlüğe yazılır.
  - Optuna sonrası son eğitimde de aynı negatif alt örnekleme kullanılır (eğitim dağılımı ayarlarla tutarlı kalsın); `pos_weight` alt örneklenmiş eğitim kümesinden hesaplanır.

## 10. Eşik

- Her model için **val binalarında nokta F1'ini en büyük yapan eşik** seçilir (F1-max). Test binalarına aynen uygulanır.
- Kuralların eşiği yoktur (ikili çıktı). R2'nin k değeri val F1 ile seçilir.

## 11. Metrikler

Hepsi genel ve alt tür bazında (§3), 3 bölme seed'inin ortalaması ± std olarak raporlanır:
- **ROC-AUC** (Kaggle ve literatürle kıyas için) ve **PR-AUC** (sınıf dengesizliği yüksek olduğu için ana sıralama ölçütü).
- Val eşiğinde **nokta düzeyinde Precision / Recall / F1**.
- **Olay recall'ü:** olayın en az bir saati işaretlenirse olay yakalanmış sayılır; tanım hanedekiyle aynı.
- **Bina bazında F1:** her test binasında ayrı F1. Medyan ve dağılım raporlanır.

## 12. Seçim kuralı

- **Ölçüt: val PR-AUC (genel)**, 3 bölmenin ortalaması.
- En yüksek modele karşı, bölme başına eşleştirilmiş farkın |ortalaması| ≤ 2·std/√n (n = 3) olan modeller gürültü düzeyinde sayılır. Bunların arasından en basiti seçilir.
- **Basitlik sırası:** L0 < L4 < L1 < L2 < L3 < L5.
- `other` alt kümesindeki val PR-AUC ayrıca raporlanır ve seçimden bağımsız olarak yorumlanır.

## 13. Test

- Her bölmenin test binaları **bir kez** kullanılır. Bu, bütün seçimler (ayarlar, eşikler, k, topluluk ağırlığı, seçilen model) val'de sabitlendikten **sonra** ve **kullanıcının açık onayıyla** yapılır.
- Testte bütün modeller raporlanır; test sonucuna göre hiçbir şey değiştirilmez.

## 14. Kaggle ve literatür kıyası

- Literatürdeki en yüksek sonuç: Kaggle birincisinin görülmemiş binalarda ROC-AUC'si 0,9866. Bizim protokolümüz Kaggle'dan üç noktada farklı:
  1. **Bölme:** 200 etiketli binayı kendimiz bölüyoruz. Kaggle'da test, etiketleri gizli başka 206 binadır.
  2. **Nedensel kısıt:** öznitelikler yalnız geçmişe bakıyor. Kaggle çözümlerinde böyle bir kısıt yoktur.
  3. **Metrik:** Kaggle yalnız ROC-AUC kullanır; biz PR-AUC ve F1'i de raporluyoruz.
- Bu yüzden 0,9866 (ve 2025 çalışmasının 0,952'si) ile kıyas **yalnız bu farklar yazılarak** yapılır; "daha iyi / daha kötü" doğrudan söylenmez.
- Ayrıca **kural referanslarının (R1–R3) ve full setinin ROC-AUC'sinin ne kadarının `gap_edge` + `frozen` alt kümesinden geldiği** gösterilir. Bunun için iki ölçüm raporlanır:
  - alt tür bazında ROC-AUC,
  - `other` dışındaki pozitifler çıkarılınca AUC'nin ne kadar düştüğü.
- PyOD karşılaştırması (nokta F1 ≈ 0,40–0,43) yalnız Isolation Forest satırı üzerinden yapılır. Orada da etiket, bölme ve eşik farkları belirtilir.

## 15. Zorunlu testler (pytest, `tests/test_lead_*.py`)

- Bina grupları kesişmiyor; her parçada iki sınıf da var; bölme seed ile tekrarlanabiliyor.
- Ölçekleyici yalnız train binalarından fit ediliyor.
- Pencere bina sınırını ve eksik saati aşmıyor; t saatinin girdisi yalnız ≤ t saatlerini içeriyor (nedensellik testi).
- R1 / R2 / geriye dönük sabitlik yalnız geçmişe bakıyor.
- Behavior setinde §7'de dışlanan sütunlar yok.
- **Hiçbir öznitelik setinde mutlak tarih türevi yok** (yılın günü, tarih sırası, yıl, zaman damgası); takvim yalnız döngüsel (güncelleme 1).
- Paylaşılan / tekil dilimi ve alt tür ataması model girdisi listelerinde yok.
- Alt tür atamasında her anomali saati tam bir alt türde ve öncelik sırası doğru.

## 16. Önceden kabul edilen sınırlılıklar

- **Etiket gürültüsü:** etiketlerin büyük kısmı veri kalitesi olaylarına karşılık geliyor; bazı belirgin tepeler etiketsiz (L1 etiket incelemesi).
- **Kapsam:** tek yıl; yalnız 200 bina; ticari binalar. Bulgular hane verisine doğrudan taşınmaz.
- **Kontaminasyon:** L4, etiketli anomalileri de içeren saatlerle eğitilir.
- **Behavior setinde kısayollar:** §7'de belirtildiği gibi tamamen kapatılamaz.
- **Siteler dengesiz:** siteye göre bölmede parça büyüklükleri ve anomali oranları tekrardan tekrara değişir; sonuçlar bu yüzden bina bölmesinden daha oynaktır.
- **Site bilgisi** Kaggle'ın işlenmiş `train_features.csv` dosyasından alındı (ASHRAE özgün metadata dosyası değil).
