# Arşivlenmiş deneyler

Bu klasördeki betikler **final sonuç üretmez**. Raporda anılan test sonrası geliştirme
deneyleri, yaz-dev altyapısı ve teşhis/analiz betikleridir. 2026-11-25'te `src/` altından
buraya taşındılar (yeniden yapılandırma; `docs/deney_gunlugu.md`). Hane anomali
geliştirmesi aynı tarihte donduruldu (`docs/gelistirme_protokolu.md`).

- Ürettikleri sonuç dosyaları **yerinde** kalır: `results/tables/`, `results/figures/`, `results/logs/`.
- Kod çalışır durumda tutulur: çekirdek modülleri (`src.core`, `src.pipelines`, ...) import eder;
  testleri `experiments/archive/tests/` altındadır ve `pytest -q` ile çalışır.
- Çekirdek kod bu klasörden **hiçbir şey import etmez**.
- Çalıştırma: proje kökünden `python -m experiments.archive.<betik>`.

| Betik | Ne yaptı | Belge |
|---|---|---|
| `anomaly_diagnostics.py` | Aşama 1: mevcut yöntemlerin teşhisi (yanlış alarm/gün, gecikme, kırılımlar) | `docs/gelistirme_protokolu.md` §5 |
| `anomaly_dev.py` | Aşama 2: kış-dev + yaz-dev değerlendirme altyapısı, kolay/orta/zor senaryolar | `docs/gelistirme_protokolu.md` |
| `train_summer_dev.py` | Yaz-dev için kat 3'ü görmemiş modelleri (`S-*`) eğitir | `docs/gelistirme_protokolu.md` §2 |
| `anomaly_short_event.py` | Aşama 3: A3 + kısa olay sinyalleri (S0–S5) | `docs/anomaly_short_event_experiments.md` |
| `anomaly_eval_audit.py` | Değerlendirme denetimi: iki olay metriği, arka plan alarmları, yeniden üretim manifestosu | `docs/archive/inceleme_ve_iyilestirme_plani_2026-11-08.md` |
| `anomaly_e1_sustained.py` | E1: sürekli işaretli artık | `docs/anomaly_e1_sustained.md` |
| `anomaly_candidates.py` | C1 (sağlam profil sapması), C2 (C1 + A3) | `docs/anomaly_candidates.md` |
| `anomaly_or_channels.py` | Kanal başına ayrı yumuşatma, VEYA birleşimi (keşifsel) | `docs/anomaly_or_channels.md` |
| `anomaly_floor_channel.py` | S0 + fiziksel taban kanalı (deneysel; `configs/experimental_s0_floor.yaml`) | `docs/anomaly_floor_channel.md` |
| `anomaly_supervised.py` | Sentetik anomalilerle eğitilen gözetimli dedektör | `docs/anomaly_supervised.md` |
| `explore_tcn_ae.py` | A7: TCN-Autoencoder keşif deneyi (yalnız val) | `results/tables/A7_explore_side_by_side.md` |
| `weather_eda.py` | Aşama 3b EDA: sıcaklık – günlük tüketim (yalnız train) | `docs/tahmin_deneyleri_ozet.md` |
| `analyze_forecast_errors.py` | Kayıtlı val tahminleriyle hata analizi | `docs/tahmin_deneyleri_ozet.md` |

Final yöntemler ve sonuçları: `docs/final_sonuclar.md`, `docs/hane_vs_sebeke.md`.
