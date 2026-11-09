# OKUBENI — v2 güncellemesi (2026-11-09)

Bu zip'i proje kök klasörüne (`enerji_projesi/`) aç. Klasör yapısı projeyle aynı.
İçinde yalnız YENİ dosyalar var; mevcut hiçbir dosyanın üzerine yazılmaz.

## Yeni dosyalar

- `docs/h3_neden_karsilanmadi.md`: rapora eklenecek "H3 neden karşılanmadı" bölümü
- `results/figures/H3_A3-s2s-norm_dev_type_recall.png`: bu bölümün grafiği
- `src/plot_h3_type_recall.py`: grafiği üretir (`python -m src.plot_h3_type_recall`)
- `docs/anomaly_v2_floor_test.md`: v2 protokolü (koşudan önce yazıldı) ve sonuçları
- `src/anomaly_v2_floor_test.py`: v2'yi hesaplar (`python -m src.anomaly_v2_floor_test --device cpu`)
- `results/tables/anomaly_v2_floor_test_per_seed.csv`, `_summary.csv`, `_paired.csv`: v2 sonuç tabloları
- `results/logs/anomaly_v2_floor_test.log`: koşunun çıktısı

## Elle eklenecek iki satır (dosyaların SONUNA)

### results/experiments.csv

```
V2-A3+taban-test,2026-11-09,anomali A3-s2s-norm + taban kanalı (TEST SONRASI v2; docs/anomaly_v2_floor_test.md),,,,,,,,,,42,,,,,,"Test, ana: olay F1 0,506 ± 0,135 (A3-s2s-norm aynı kopyalar 0,384 ± 0,170; fark +0,123, 5/5 seed); zor: 0,460 ± 0,059 (0,318). Test daha önce görüldü; bağımsız değil, final değil. Hocanın onayıyla tek koşu."
```

### docs/deney_gunlugu.md

```
| 2026-11-09 | **v2 test koşusu (TEST SONRASI, final değil; `docs/anomaly_v2_floor_test.md`, protokol koşudan önce; `python -m src.anomaly_v2_floor_test`):** A3-s2s-norm (val-dev eşiği/penceresi aynen) VEYA taban kanalı (train 0,5. yüzdelik = 0,155 kW, 2 ardışık saat). Yeniden üretim tamam (A3-s2s-norm ana 0,3835). Ana senaryo olay F1 0,384 → 0,506 ± 0,135 (fark +0,123, 5/5 seed); zor 0,318 → 0,460. Kazancın tamamı drop recall'ünden (0,10 → 1,00); diğer türler değişmedi; enjeksiyonsuz testte taban alarmı 0. | Hocanın onayıyla tek koşu. Test daha önce görüldüğü için bağımsız değil; final yöntem A3-s2s-norm kalır, v2 raporda ayrı satır. H3 (≥ 0,70) yine karşılanmadı. | V2-A3+taban-test | Danışman oturumu |
```

## Ayrıca

- `README.md` ve `docs/hocaya_model_sonuclari.md` içindeki anomali tablosuna v2'yi resmî sonucun ALTINA, "v2 (test sonrası, final değil)" etiketiyle ayrı bir satır olarak ekle.
- Final yöntem A3-s2s-norm olarak kalır. H3 (≥ 0,70) karşılanmadı.
