"""Doğrulama/test kopyalarına kontrollü sentetik anomali enjeksiyonu (README §4.9)."""

import numpy as np
import pandas as pd

ANOMALY_TYPES = ("spike", "drop", "level_shift", "pattern")


def _segment_is_free(labels, imputed, start, end):
    """[start, end) aralığı ve bir saatlik komşuluğu boş ve doldurulmamış mı?

    Neden: Anomaliler üst üste binmez ve bitişik olmaz (tip bazında recall
    ölçülebilsin); doldurulmuş (is_imputed=1) saatlere enjeksiyon yapılmaz.
    """
    if start < 0 or end > len(labels):
        return False
    padded = slice(max(start - 1, 0), min(end + 1, len(labels)))
    return not labels[padded].any() and not imputed[start:end].any()


def inject_anomalies(frame, target, settings, seed):
    """Bir küme tablosunun KOPYASINA dört tip anomali enjekte eder.

    Girdi: saatlik küme tablosu (val veya test), hedef sütunu (kW), config
    "injection" bölümü ve seed (val=1, test=2).
    Çıktı: (enjekte edilmiş kopya, etiket tablosu). Etiket tablosu aynı
    indeksle `is_anomaly` (0/1) ve `anomaly_type` (yoksa "none") içerir.
    Neden: Gerçek etiket olmadığından bilinen etiketli kontrollü anomaliler
    üretilir. Ham kW üzerinde yapılır (ölçekleme sonra, train'den gelir);
    orijinal tablo asla değiştirilmez. Toplam etiketli saat oranı
    target_ratio'ya ulaşınca durulur ve max_ratio aşılmaz.
    """
    rng = np.random.default_rng(seed)
    injected = frame.copy()
    values = injected[target].to_numpy(dtype=np.float64).copy()
    original = frame[target].to_numpy(dtype=np.float64)
    imputed = frame["is_imputed"].to_numpy(dtype=bool)
    n_hours = len(frame)
    labels = np.zeros(n_hours, dtype=bool)
    types = np.array(["none"] * n_hours, dtype=object)
    hours = frame.index.hour.to_numpy()
    target_count = settings["target_ratio"] * n_hours
    max_count = settings["max_ratio"] * n_hours

    attempts = 0
    while labels.sum() < target_count:
        attempts += 1
        if attempts > 10_000:
            raise RuntimeError("Enjeksiyon hedef oranına ulaşamadı.")
        kind = ANOMALY_TYPES[rng.integers(len(ANOMALY_TYPES))]
        if kind == "pattern":
            # Rastgele bir günün gece saatleri (00–05) o günün gündüz seviyesine.
            night = settings["pattern"]["night_hours"]
            midnights = np.flatnonzero(hours == 0)
            start = int(rng.choice(midnights))
            end = start + len(night)
            day = slice(start + 8, start + 21)  # aynı günün 08:00–20:00 saatleri
            if day.stop > n_hours:
                continue
        else:
            spec = settings[kind]
            length = int(rng.integers(spec["min_len"], spec["max_len"] + 1))
            start = int(rng.integers(0, n_hours - length + 1))
            end = start + length
        if not _segment_is_free(labels, imputed, start, end):
            continue
        if labels.sum() + (end - start) > max_count:
            continue

        segment = slice(start, end)
        if kind == "spike":
            spec = settings["spike"]
            factors = rng.uniform(spec["factor_low"], spec["factor_high"], end - start)
            values[segment] = original[segment] * factors
        elif kind == "drop":
            values[segment] = settings["drop"]["value_kw"]
        elif kind == "level_shift":
            values[segment] = original[segment] * settings["level_shift"]["factor"]
        else:
            values[segment] = original[day].mean()
        labels[segment] = True
        types[segment] = kind

    injected[target] = values
    label_table = pd.DataFrame(
        {"is_anomaly": labels.astype("int8"), "anomaly_type": types},
        index=frame.index,
    )
    return injected, label_table
