# Öneri formundaki sayıların kaynakları

Kural (CLAUDE.md): formdaki her sayı bir kod çıktısından gelir. Bölüm sırasıyla listelenir. "Notebook" kaynakları `notebooks/01_eda.ipynb` içindeki kayıtlı hücre çıktılarıdır (yalnız eğitim kümesi; eksik veri ve bölme tüm dönem).

## Proje Ekibi
| Sayı | Kaynak |
|---|---|
| 230290018, 230290046 (öğrenci numaraları) | README.md, ekip tablosu |

## Projenin Özgün Değeri
| Sayı | Kaynak |
|---|---|
| 24 saatlik tahmin ufku; 168 saat (haftalık naif) | `configs/config.yaml` (forecast), README §1 |
| 160 müşteri (ELD toplamı) | `docs/eld_protokol.md` §1 "Müşteri seçimi: 160 / 370" (`src/data/eld.py` çıktısı) |
| İki zorluk senaryosu, dört anomali tipi | README §4.9, `docs/final_test_protokolu.md` |
