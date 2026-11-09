"""A3 + taban (v2) kuralı: deney betiği (src/anomaly_v2_floor_test.py) ve demo AYNI fonksiyonu kullanır.

Tanım (test sonrası, final değil; docs/v2_entegrasyon_sonuc_ozeti.md):
- Ana kanal: A3-s2s-norm, yumuşatma ve eşik kayıtlı seçimden (anomaly_protocol_selection.csv).
- Taban kanalı: f_t = (y_t < taban) VE (y_{t−1} < taban); taban = train 0,005 yüzdeliği (train_floor).
  İki kayıt gerçekten ardışık saat olmalı; doldurulmuş saat "düşük" sayılmaz.
- Alarm = A3 alarmı VEYA taban alarmı. Eşdeğer skor r = max(s̃ / T, 2·f); alarm r ≥ 1.
Etiketler hiçbir adımda kullanılmaz.
"""

import numpy as np

from src.anomaly_floor_channel import floor_flags, train_floor  # noqa: F401  (train_floor dışarı açılır)

BASE_METHOD = "A3-s2s-norm"
V2_METHOD = "A3 + taban (v2)"
V2_LABEL = "A3 + taban (v2)"


def v2_score(base_smoothed, threshold, values, floor, index, imputed):
    """[N] eşdeğer skor ve [N] taban bayrakları.

    Girdi: yumuşatılmış A3-s2s-norm skoru [N] (NaN olabilir), A3 eşiği, ham kW [N], taban (kW),
    zaman indeksi [N], doldurulmuş saat maskesi [N].
    Çıktı: (r [N], f [N]); r ≥ 1 ⇔ (s̃ ≥ T) VEYA f. A3 skoru NaN olan saatte r yalnız tabana bağlıdır;
    değerlendirme maskesi çağıran tarafta iki yöntem için aynı uygulanır.
    """
    flags = floor_flags(values, floor, index=index, imputed=imputed)
    with np.errstate(invalid="ignore"):
        score = np.fmax(np.asarray(base_smoothed, dtype=float) / threshold, 2.0 * flags)
    return score, flags
