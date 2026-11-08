# Anomali geliştirme protokolü (H3) — deneylerden ÖNCE yazıldı

Dal: `seyda/gelistirme` · Tarih: 2026-11-08 · İş paketi: İP4 (anomali), İP5 (hata analizi)

Bu belge, final testten sonra başlayan anomali geliştirme çalışmasının kurallarını deney yapılmadan önce sabitler. Bu belgeye aykırı bir seçim yapılırsa sonuç geçersiz sayılır. Kural değişikliği gerekirse tarih ve gerekçeyle `docs/deney_gunlugu.md`'ye yazılır; eski sonuçlar silinmez.

## 1. Dondurulan sonuçlar

- Hane test sonuçları (`docs/final_sonuclar.md` §1–§2; `results/tables/anomaly_protocol_test*.csv`, `final_*_val_vs_test.csv`) **tarihsel final sonuçtur**. Değiştirilmez, yeniden hesaplanmaz. Yeni yöntemler bu sonuçları "iyileştirmiş" gibi sunulmaz.
- **val-hold** (2010-01 – 2010-03) sonuçları görüldü (revizyon 1, A6). Yeni hiçbir seçim veya eşik optimizasyonunda kullanılmaz; yeni yöntemler için de ölçülmez.
- **ELD 160 müşteri** deneyi (`docs/hane_vs_sebeke.md`) sonuçları görüldü. "Bağımsız test" diye adlandırılmaz, tarihsel karşılaştırma olarak kalır.
- `--split test` yeni yöntemler için **hiç** çalıştırılmaz.

## 2. Geliştirme (seçim) kümeleri

| Küme | Dönem | Mevsim | Tahminler | Normalizasyon referansı |
|---|---|---|---|---|
| **kış-dev** = val-dev | 2009-10-01 – 2009-12-31 | sonbahar–kış | tüm train ile eğitilmiş modeller | OOF katları 1–3 (mevcut) veya train |
| **yaz-dev** = OOF kat 3 | 2009-06-01 – 2009-09-30 | yaz | yalnız 2009-05-31'e kadarki veriyle eğitilmiş modeller | yalnız 2009-05-31'e kadarki veri (OOF katları 1–2, train başı) |

- Yöntem, skor birleştirmesi, normalizasyon türü, yumuşatma penceresi ve eşikler **yalnız bu iki kümede** seçilir. Ölçüt iki kümenin olay F1 ortalamasıdır; ayrıca iki kümenin ayrı değerleri raporlanır (kış ile yaz arasındaki fark bulgu olarak tutulur).
- **yaz-dev bağımsızlık denetimi (zorunlu, deneyden önce yapılır):**
  1. F10-opt ve F5-opt'un Optuna araması (Aşama 4), 3 katlı CV'sinde kat 3'ü (2009-06 – 2009-09) **kullandı** (`results/tables/optuna_*_best.csv`, `folds`). Bu yüzden Optuna ayarlı modellerin kat 3 tahminleri yaz-dev için **kullanılmaz**.
  2. Yerine yaz-dev tahmin modeli, Optuna'dan **önce** sabitlenmiş yapılandırmayla (`configs/config.yaml` varsayılanları, kat 3 görülmeden belirlenmiş) yalnız 2009-05-31'e kadarki veriyle eğitilir; early stopping kattan hemen önceki ayla (2009-05) yapılır. Nested Optuna, maliyeti nedeniyle yapılmaz.
  3. Train ile fit edilen dedektörler (A1 Isolation Forest, A2 LSTM-AE) ve σ/std tabloları yaz-dev için yalnız 2009-05-31'e kadarki veriyle yeniden fit edilir. A0 yalnız geçmişe baktığı için yeniden fit gerekmez.
- **Kararlılık (isteğe bağlı, maliyet izin verirse):** OOF katları 1 ve 2 (2008-10 – 2009-01, 2009-02 – 2009-05) aynı kuralla ek geliştirme katı olarak raporlanır; seçimde kullanılmaz.

## 3. Bağımsız kontrol

- **Aday:** ELD'de 160 müşterilik toplama **girmeyen** 210 müşteri (2012-01-01'e kadar hiç sıfır dışı değeri olmayanlar). Bu grup hiçbir deneyde kullanılmadı.
- Uygunluğu (veri süresi, sıfır dolgular, yeterli müşteri sayısı) yöntemlere bakmadan, yalnız veri özellikleriyle denetlenir. Uygun değilse görülmemiş başka bir veri seti aranır; bulunamazsa yeni yöntem için bağımsız sonuç **yok** diye raporlanır.
- Kontrol protokolü (yöntem, normalizasyon, eşik, pencere, enjeksiyon senaryosu, metrikler) ölçümden **önce** ayrı bir belgede sabitlenir. **Tek ölçüm** yapılır; sonuç görüldükten sonra hiçbir şey değiştirilmez.

## 4. Değerlendirme tanımları (bütün yöntemlerde aynı)

- **Seçim ölçütü:** yeni **orta** senaryoda olay F1 (kış-dev + yaz-dev). Kolay ve zor senaryolar ek rapordur. Eski "ana" senaryo, önceki sonuçlarla bağ kurmak için ayrıca raporlanır, seçimde kullanılmaz.
- **Senaryo tanımı önce:** kolay/orta/zor büyüklükleri (spike, drop, seviye kayması, desen) yalnız **train** dağılımına (σ, saatlik düzey, gece/gündüz farkı) dayanarak ve fiziksel olarak tutarlı biçimde, hiçbir yöntem çalıştırılmadan sabitlenir.
- **Olay tanımı değişmez:** olay = aynı tipten ardışık etiketli saatler; olay recall'ü = en az bir saati işaretlenen olay oranı; olay precision'ı = gerçek anomaliyle kesişen alarm bloklarının oranı. Alarm birleştirme veya histerezis eklenirse bu tanım aynen uygulanır; anomali tanımı ya da etiketler değiştirilmez.
- **Yeni ek metrikler:** günlük yanlış alarm sayısı (hiçbir enjekte saatle kesişmeyen alarm bloğu / değerlendirilen gün), tespit gecikmesi (yakalanan olayda ilk işaretli saat − olay başlangıcı, saat), tip bazında olay recall'ü, PR-AUC.
- **Belirsizlik:** her küme için seed başına metrik; ortalama ± std ve eşleştirilmiş fark. Gürültü kuralı final protokoldeki gibi: |ort| ≤ 2·std/√n.

## 5. Geliştirme sırası (backlog)

| Sıra | Aşama | İçerik | Seçim yapar mı? |
|---|---|---|---|
| 1 | Teşhis | Mevcut A0/A1/A2/A3/A5 skorları, kış-dev ana senaryo, **eski seçilmiş** pencere/eşiklerle: yanlış alarm/gün, gecikme, saat / tip / hafta içi–sonu kırılımı, en uzun kaçırılan olaylar | Hayır (betimleme) |
| 2 | Altyapı | yaz-dev (bağımsızlık denetimli modeller), kolay/orta/zor senaryo tanımı | Hayır |
| 3 | Normalizasyon | N1 saat×gün std (mevcut), N2 saat×gün×mevsim std, N3 saat×ay robust MAD (OOF'ta saat×gün×ay hücresi ≈ 4 örneğe düştüğü için saat×ay) | Evet |
| 4 | Çoklu sinyal | A0-robust, A3-norm, A2 hata, birinci fark; önce tek tek, sonra birleşim (ağırlıklar yalnız geliştirme kümelerinde) | Evet |
| 5 | Artık kaynağı | F13-ens, LSTM, GRU, XGBoost artıkları; ufuk 1/6/12/24; ort/maks/ağırlıklı | Evet |
| 6 | Eşik ve alarm | global / mevsimsel / saat dilimi eşiği, pencere 1/3/6, histerezis, yakın blok birleştirme, minimum süre | Evet |
| 7 | AE | pencere 24/48/72, gizli 32/64/128, LSTM-AE / GRU-AE, MSE / MAE skoru, yalnız tüketim / çoklu öznitelik | Evet |
| 8 | Bağımsız kontrol | §3, tek ölçüm | Hayır |

Tahmin tarafında yeni model eklenmez. Yalnız raporlama ve analiz işleri yapılır: GRU'nun final sonuçları, XGBoost'un RMSE avantajı, mevsime göre hata, en kötü haftalar, W1'in neden katkı sağlamadığı.

## 6. Başarı düzeyleri (geliştirme kilometre taşları, evrensel ölçüt değil)

Kış-dev + yaz-dev orta senaryo olay F1: 0,50 (ilk gelişim) · 0,60 (iyi) · 0,70 (proje hedefi H3) · 0,80+ (bağımsız doğrulama şart). Bu düzeylerin hiçbiri test kümesiyle ölçülmez.
