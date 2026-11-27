# docs/ — belgelerin haritası ve okuma sırası

Bu dosya, final raporu yazacak kişi (Dilara) için giriş noktasıdır. Raporun nasıl yazılacağı: `rapor/DILARA_RAPOR_REHBERI.md`. Rapordaki her sayı yalnız `results/` altındaki tablolardan ya da aşağıdaki **sonuç** belgelerinden alınır.

## Önce bunları oku (sırayla)

1. `../README.md` §1–§3 — projenin amacı, veri setleri, durum tablosu.
2. `final_test_protokolu.md` — hane için test öncesi sabitlenen kurallar (donmuş).
3. `final_sonuclar.md` — hane final sonuçları: tahmin ve anomali, val ve test yan yana.
4. `eld_protokol.md` → `hane_vs_sebeke.md` — şebeke deneyi kuralları (donmuş) ve hane ile karşılaştırma.
5. `veri_seti_lead1_degerlendirme.md` → `lead_protokol.md` → `lead_val_ozet.md` → `lead_sonuclar.md` — LEAD'in neden eklendiği, kuralları (donmuş), val kapanışı ve final test.
6. `tahmin_deneyleri_ozet.md` — tahmin modellerinin val'de nasıl seçildiği (Aşama 1–5).
7. `literatur_taramasi.md` (anlatı ve kaynakça) → `literatur.md` (matris) → `literatur_katalog.csv` (ham tarama, 271 kayıt).
8. `savunma_sorulari.md` — proje boyunca birikmiş sade soru–cevaplar.
9. `deney_gunlugu.md` — bütün kararların tarihli kaydı (donmuş; bir kararın gerekçesi aranırken).

## Bütün belgeler

| Belge | Ne anlatır | Durum |
|---|---|---|
| `README.md` | Bu dosya: harita ve okuma sırası | — |
| `deney_gunlugu.md` | Tarih + karar + gerekçe; bütün deneylerin ve kural değişikliklerinin kaydı | **donmuş** (yalnız yeni satır eklenir) |
| `final_test_protokolu.md` | Hane final testinin kuralları; test öncesi yazıldı | **donmuş** |
| `gelistirme_protokolu.md` | Test sonrası anomali geliştirmesinin kuralları (H3) | **donmuş** |
| `eld_protokol.md` | Şebeke (ELD) deneyinin kuralları; test öncesi yazıldı | **donmuş** |
| `lead_protokol.md` | LEAD1.0 kuralları ve güncellemeleri (§17–§21); her güncelleme test öncesi | **donmuş** |
| `final_sonuclar.md` | Hane final sonuçları (tahmin + anomali; val ve test) | donmuş sonuç |
| `hane_vs_sebeke.md` | Aynı yöntemlerin hane ve şebekede karşılaştırması (test) | donmuş sonuç |
| `lead_val_ozet.md` | LEAD val kapanışı: seçim, site bölmesi analizi, I-turu, test kontrol listesi | sonuç (val) |
| `lead_sonuclar.md` | LEAD final test: val ile yan yana, alt türler, literatür, **test sonrası gözlem** (§7) | sonuç (test) |
| `tahmin_deneyleri_ozet.md` | Tahmin deneylerinin özeti (yalnız val) | sonuç (val) |
| `veri_seti_lead1_degerlendirme.md` | LEAD1.0'ın yönerge kriterlerine göre değerlendirmesi ve literatürü | arka plan |
| `literatur_taramasi.md` | Literatür anlatısı, karşılaştırma ve kaynakça | arka plan (künyeler raporda yeniden doğrulanmalı) |
| `literatur.md` | Literatür matrisi (kaynak başına özet) | arka plan |
| `literatur_katalog.csv` | Ham literatür taraması (elle elenmemiş) | ham |
| `savunma_sorulari.md` | Savunmada sorulabilecek sorular ve cevapları (tarihli) | yardımcı |
| `oneri_formu/` | Hocaya sunulan öneri formunun içerik kaynağı, şablon, yönerge, sayı kaynakları | tarihsel (plandan sapmalar için) |
| `rapor/DILARA_RAPOR_REHBERI.md` | Final raporun adım adım yazım rehberi | **rapor için başlangıç** |
| `sunum/kod_haritasi.md` | Demoda gösterilen her parçanın kodda yeri; veri hazırlama ve bölme (Dilara anlatacak) | sunum |
| `sunum/yedek/` | Demo sayfalarının 1280×720 yedek ekran görüntüleri | sunum |
| `ders/` | Ders yönergesi ve değerlendirme formu (PDF) | kaynak |
| `anomaly_*.md` (6 dosya) | Hane anomali geliştirme deneyleri (aşağıya bakın) | geliştirme deneyi |
| `archive/` | Eski notlar, taslaklar ve eski demo görüntüleri (aşağıya bakın) | arşiv |

## Rapora kaynak olarak kullanma

Aşağıdakiler **final sonuç değildir**. Raporda yalnız **"test sonrası ek analiz"** ya da **"geliştirme deneyi"** olarak, bu etiketle anılabilir; sayıları final sonuç gibi sunulmaz.

- **v2 ile ilgili her şey** (taban kanalı fikri ders dışından bir öğretim üyesinin önerisidir; anıldığı her yerde belirtilir):
  - `final_sonuclar.md` → "Test sonrası geliştirme" bölümü (A6-dual, A6-dual-floor; test kümesinde çalıştırılmadı),
  - `anomaly_floor_channel.md` (fiziksel taban kanalı deneyi),
  - birleştirilmeyen `dilara/v2-taban-entegrasyon` dalının içeriği (git etiketi `arsiv/dilara-v2`; v2 zip içeriği, `anomaly_v2_floor_test`, `h3_neden_karsilanmadi`, `v2_entegrasyon_sonuc_ozeti`) — **hiçbiri kaynak alınmaz**,
  - `docs/hocaya_model_sonuclari.md` (repoda yok; bulunursa da kaynak alınmaz).
- **`archive/`** klasörünün tamamı: eski durum özetleri, ilk inceleme planı, ilk literatür matrisi, eski rapor taslağı (`rapor_taslak_2026-11-08.md`), eski demo görüntüleri (`demo_v2.0/`).
- **Geliştirme deneyi belgeleri** (`anomaly_*.md`): `anomaly_candidates.md`, `anomaly_e1_sustained.md`, `anomaly_floor_channel.md`, `anomaly_or_channels.md`, `anomaly_short_event_experiments.md`, `anomaly_supervised.md`. Hane anomali hedefinin (olay F1 0,70) neden karşılanamadığını anlatırken "geliştirme deneyi" olarak anılabilir.
- **LEAD'de test sonrası gözlemler:** `lead_sonuclar.md` §7 (L1-FE full'ün testte daha iyi genelleşmesi) final sonuç değil, "test sonrası gözlem"dir; final model L1 behavior'dır.
