# H3 hedefi neden karşılanmadı? (rapora eklenecek bölüm)

Bu bölüm **yeni deney içermez**. Bütün sayılar daha önce üretilmiş `results/tables/` dosyalarından alınmıştır.
Figür: `results/figures/H3_A3-s2s-norm_dev_type_recall.png` (üretmek için `python -m src.plot_h3_type_recall`).

---

## 1. Sonuç

H3 hedefi (olay F1 ≥ 0,70) karşılanmadı. Final yöntem A3-s2s-norm, tek seferlik test kümesinde (ana senaryo, 5 seed) olay F1 = **0,384 ± 0,170** elde etti. Kaynak: `final_anomaly_val_vs_test.csv`.

Aşağıdaki analiz, bu sonucun model seçiminden değil, **veri ve enjekte edilen anomali türlerinin yapısından** kaynaklandığını gösterir.

## 2. Anomali türüne göre yakalama oranı

A3-s2s-norm, geliştirme kümeleri, orta senaryo, 10 seed ortalaması (kaynak: `anomaly_eval_audit_summary.csv`, Aşama 2):

| Anomali türü | Olay recall'ü (kış-dev) | Olay recall'ü (yaz-dev) |
|---|---|---|
| Spike (ani sıçrama) | 0,90 | 0,80 |
| Seviye kayması | 1,00 | 0,90 |
| Drop (kesinti) | 0,15 | 0,00 |
| Gece deseni | 0,90 | 0,00 |

Yöntem, tüketimi **yukarı** iten ve belirgin olan anomalileri (spike, seviye kayması) büyük ölçüde yakalıyor. Drop ve yaz gece deseni ise neredeyse hiç yakalanmıyor. Bu durum yalnız A3'e özgü değil: drop recall'ü A0, A1 ve S0'da da 0,00–0,04 aralığında.

## 3. Neden bu iki tür yakalanamıyor?

**Drop (kesinti).** Enjekte edilen olaylar 1,0–1,8 kW'tan 0,05 kW'a iniyor. Tek bir hanede bu büyüklükte düşüşler olağan (bir cihazın kapanması, evden çıkılması). Tahmin artığına dayalı S0 skoru (A3-s2s-hmean, eşiğe bölünmüş ölçek) drop olaylarında en fazla 0,80 / 0,81'e (kış / yaz) çıkıyor, karar eşiği ise 1,16. Yani düşüş, modelin olağan tahmin hatasının içinde kalıyor (kaynak: `docs/anomaly_floor_channel.md`).

**Yaz gece deseni.** Bu anomali gece saatlerini gündüz seviyesine çekiyor. Yazın gündüz ve gece tüketimi birbirine yakın olduğu için yaratılan sapma yalnız 0,50 kW. Modelin bir saat sonrası için olağan hatası (h=1 MAE) ise 0,34 kW. Sapma gürültüden ayırt edilemiyor (kaynak: `docs/anomaly_e1_sustained.md`).

## 4. Hedef bu kurulumda ulaşılabilir miydi? (oracle analizi)

Teşhis amaçlı bir deneyde, S0 alarmlarına tek bir türün **bütün** saatleri eklendi. Böylece "bu tür kusursuz yakalansaydı F1 ne olurdu?" sorusu cevaplandı (kaynak: `anomaly_eval_audit_oracle.csv`):

| Kusursuz yakalanan tür | Olay F1 kış-dev | Olay F1 yaz-dev |
|---|---|---|
| Yok (mevcut S0) | 0,551 | 0,535 |
| Drop | 0,644 | 0,604 |
| Gece deseni | 0,583 | 0,774 |

- Drop'u kusursuz yakalamak bile F1'i iki mevsimde de 0,70'in altında bırakıyor.
- Gece desenini kusursuz yakalamak yalnız yazda 0,70'i geçiriyor.
- Hiçbir tür tek başına iki mevsimde birden hedefe ulaştırmıyor.

Bu analiz bir başarı tahmini değildir, yalnız tavanı gösterir.

*Not: Seviye kayması satırı (kış 0,459, yaz 0,510) mevcut değerin altında kalıyor. Bunun sebebi, zaten yakalanan bu türün bütün saatleri eklenince alarm bloklarının birleşmesi ve blok sayımının değişmesidir. Yöntemle ilgili bir bulgu değildir.*

## 5. Belirsizlik

Test kümesinde her seed'de ortalama yalnız 15,4 olay var ve F1'in seed'ler arası sapması ±0,170. Bu yüzden testte yöntemler arasındaki farklar (A0 0,398, A3-s2s-norm 0,384) istatistiksel olarak ayırt edilemiyor.

## 6. Değerlendirme ve sınırlılıklar

- H3'teki 0,70 eşiği, anomali türlerinin bu tek hanede ne kadar ayırt edilebilir olduğu bilinmeden konmuştu. Analiz, dört türden ikisinin bu veride tahmin artığı veya istatistiksel yöntemlerle ayırt edilemediğini gösteriyor.
- Test kümesi bir kez kullanıldı. Sonrasında yapılan geliştirme denemeleri bağımsız olarak doğrulanamadığı için final sonuca katılmadı.
- Temel sınırlılıklar: tek hane verisi, gerçek anomali etiketi olmaması (sentetik enjeksiyon), test kümesindeki az sayıda olay.

## 7. Gelecek çalışma

- **Fiziksel taban kuralı (v2, test sonrası):** Kural şu: "Tüketim, hanenin eğitim verisindeki 0,5. yüzdelik düzeyin (0,155 kW) altına iki saat üst üste inerse alarm."
  - Geliştirme verisinde drop olaylarını yakaladı (`docs/anomaly_floor_channel.md`).
  - Testte tek bir koşuda, final yönteme eklendiğinde ana senaryoda olay F1'i 0,384'ten 0,506 ± 0,135'e çıkardı (zor senaryoda 0,318'den 0,460'a). 5 seed'in hepsinde artış oldu ve gerçek veride yanlış alarm eklenmedi (`docs/anomaly_v2_floor_test.md`).
  - Test kümesi daha önce görüldüğü için bu sonuç bağımsız değildir. Ayrıca kazanç, enjekte edilen değerin (0,05 kW) tabanın altında olmasına bağlıdır. Bu yüzden final yöntem değiştirilmedi ve H3 yine karşılanmadı.
- Gerçek etiketli veya çok haneli bir veri setiyle değerlendirme yapmak.
- Anomali türlerini, hanenin olağan değişkenliğine göre ölçeklenmiş büyüklüklerle tanımlamak.
