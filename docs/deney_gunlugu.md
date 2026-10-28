# Deney Günlüğü

Her önemli karar (eşik, öznitelik seti, hiperparametre, protokol) buraya **tarih + karar + gerekçe + ilgili exp_id** ile eklenir. Sonuç sayıları yalnızca `results/` dosyalarından alınır.

| Tarih | Karar | Gerekçe | exp_id | Karar veren |
|---|---|---|---|---|
| 2026-10-28 | `experiments.csv` sütun adları İngilizce: `date`, `train_time_s`, `note` (README'deki `tarih`, `süre_sn`, `not` yerine) | CLAUDE.md kuralı: değişken/sütun adları İngilizce; README ile CLAUDE.md arasındaki çelişki giderildi | — | Şeyda |
| 2026-10-28 | Tahmin modellerinde early stopping **val MAE (kW, orijinal ölçek)** izler (README'deki "val loss" yerine) | Model seçim kriteriyle aynı metrik olsun; "en iyi epoch" ile "kaydedilen model" tutarlı olur | F4–F9 | Şeyda |
| 2026-10-28 | `src/utils/logging.py` → `src/utils/exp_logger.py` | Python'un standart `logging` modülüyle isim çakışmasını önlemek | — | Şeyda |
| 2026-10-28 | Öznitelik seti C'ye `Global_intensity` alınmadı | Hedefle (`Global_active_power`) neredeyse doğrusal ilişkili; ek bilgi taşımaz. EDA korelasyon matrisiyle doğrulanacak | F7 | README §4.2 |
| 2026-10-28 | Ridge `alpha=1.0` ve MLP `[128, 64]` geçici varsayılan | README'de belirtilmemiş; F3/F4 öncesi doğrulama kümesiyle netleştirilecek | F3, F4 | Açık |
| 2026-10-28 | Gün 1'de baseline’lar için test metrikleri yanlışlıkla hesaplandı. Parametresiz modeller olduğu için model seçimine etkisi yok. Satırlar silindi, final değerlendirmede yeniden hesaplanacak. | Test kümesinin erken kullanımını görünür kılmak ve CLAUDE.md kuralı 5'e uygun olarak bundan sonra yalnız açık `--split test` ile değerlendirmek | — | İkisi |
