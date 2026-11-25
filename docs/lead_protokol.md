# LEAD1.0 protokolü — model eğitiminden ve testten ÖNCE yazıldı

Dal: `seyda/lead` · Tarih: 2026-11-25 · Durum: **sabitlendi, henüz uygulanmadı** (L2 ve sonrası).

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
- **Site bilgisi:** `data/raw/lead/building_metadata.csv` (ASHRAE GEP III) şu an yok. Site eşleşmesi gerektiren her madde "site bilgisi varsa" koşuluna bağlıdır.

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
- **Raporlama:** bütün metrikler **genel** ve **`gap_edge` / `frozen` / `other`** olarak ayrı ayrı raporlanır.

## 4. Nedensellik

- **Bütün öznitelikler ve kurallar yalnız geçmişe bakar.** t saatindeki skor, yalnızca t ve öncesindeki okumalar, eksiklik bilgisi ve takvimle hesaplanır.
- Bina başına normalizasyon da yalnız geçmişten yapılır: bir binanın tüm yılına ait istatistik kullanılmaz.
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
- **Site bilgisi varsa ek bölme: siteye göre gruplu.** Bu bir sağlamlık kontrolüdür. Yalnız val'de seçilen **en iyi 2 model + kurallar (L0)** için yapılır. Site bilgisi yoksa yapılmaz ve "yapılmadı" diye raporlanır.
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

- **Önceden kabul edilen sınırlılık:** behavior setinde eksiklik ve sabitlik bilgisi tamamen yok edilemez. İleri doldurma boşlukları yapay sabit serilere çevirir; ham okuma ile `lag_1h`'nin eşitliği de dolaylı bilgi taşır.
- **Ölçüm:** behavior modellerinin `gap_edge` ve `frozen` alt kümelerindeki recall'ü raporlanır. Bu değer yüksekse dolaylı sızıntı var demektir ve böyle yorumlanır.
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
- Derin modellerde eğitim süresine göre normal pencereler rastgele alt örneklenebilir. Bu karar 1 epoch'luk smoke test'ten sonra günlüğe yazılır; değerlendirme her zaman tüm saatlerde yapılır.

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
- Alt tür atamasında her anomali saati tam bir alt türde ve öncelik sırası doğru.

## 16. Önceden kabul edilen sınırlılıklar

- **Etiket gürültüsü:** etiketlerin büyük kısmı veri kalitesi olaylarına karşılık geliyor; bazı belirgin tepeler etiketsiz (L1 etiket incelemesi).
- **Kapsam:** tek yıl; yalnız 200 bina; ticari binalar. Bulgular hane verisine doğrudan taşınmaz.
- **Kontaminasyon:** L4, etiketli anomalileri de içeren saatlerle eğitilir.
- **Behavior setinde dolaylı sızıntı:** §7'de belirtildiği gibi tamamen önlenemez.
- **Site bilgisi** yoksa siteye göre sağlamlık kontrolü yapılamaz. Aynı saatte başlayan olayların aynı sitede olup olmadığı da doğrulanamaz.
