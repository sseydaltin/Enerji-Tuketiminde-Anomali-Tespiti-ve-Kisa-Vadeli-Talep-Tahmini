"""LEAD EDA hesapları: kapsama, anomali olayları, sıfır ve sabit (donmuş) okumalar.

Bütün fonksiyonlar temiz tablo (src.datasets.lead.load.clean çıktısı; (bina, zaman) sıralı,
bina-saat başına tek satır) üzerinde çalışır ve dosya yazmaz. Bu modül yalnız betimler:
model, eşik veya seçim yoktur.
"""

import numpy as np
import pandas as pd

HOUR = pd.Timedelta(hours=1)


def _continues(df):
    """Her satır bir önceki satırla aynı binada ve tam 1 saat sonra mı? [N] bool.

    Neden: Eksik saatler (temizlikte düşen satırlar) ardışıklığı bozar; boşluğun iki yanı
    aynı olaya veya aynı sabit seriye sayılmaz.
    """
    same_building = df["building_id"].eq(df["building_id"].shift())
    one_hour = df["timestamp"].diff().eq(HOUR)
    return (same_building & one_hour).to_numpy()


def coverage(df):
    """Bina başına gözlem sayısı ve eksik saat.

    Çıktı: DataFrame [bina] (building_id, n_obs, first, last, expected_hours, missing_hours).
    Beklenen saat sayısı, verinin genel ilk–son saati arasındaki tam saat ızgarasıdır
    (2016 artık yıl: 8.784 saat); bina bu ızgarada olmayan her saat için eksik sayılır.
    """
    start, end = df["timestamp"].min(), df["timestamp"].max()
    expected = int((end - start) / HOUR) + 1
    out = df.groupby("building_id")["timestamp"].agg(n_obs="size", first="min", last="max").reset_index()
    out["expected_hours"] = expected
    out["missing_hours"] = expected - out["n_obs"]
    return out


def anomaly_events(df):
    """Anomali olayları: aynı binada, boşluksuz ardışık anomaly=1 saatleri.

    Çıktı: DataFrame [olay] (building_id, start, end, length_hours).
    Neden: "Olay" tanımı household'dakiyle aynı (ardışık etiketli saatler); eksik saat olayı böler.
    """
    is_anomaly = df["anomaly"].to_numpy() == 1
    previous_anomaly = np.r_[False, is_anomaly[:-1]]
    starts = is_anomaly & ~(previous_anomaly & _continues(df))     # yeni olay başlangıcı
    event_id = np.cumsum(starts)
    rows = df.loc[is_anomaly, ["building_id", "timestamp"]].assign(event=event_id[is_anomaly])
    events = rows.groupby("event").agg(building_id=("building_id", "first"),
                                       start=("timestamp", "min"), end=("timestamp", "max"),
                                       length_hours=("timestamp", "size"))
    return events.reset_index(drop=True)


def constant_run_length(df):
    """Her satırın içinde bulunduğu sabit değer serisinin uzunluğu (saat). [N] int.

    Sabit seri: aynı binada, boşluksuz ardışık saatlerde birebir aynı meter_reading.
    Tek başına duran bir değerin seri uzunluğu 1'dir.
    """
    values = df["meter_reading"].to_numpy()
    same_as_previous = _continues(df) & (values == np.r_[np.nan, values[:-1]])
    run_id = np.cumsum(~same_as_previous)
    return pd.Series(run_id).groupby(run_id).transform("size").to_numpy()


def reading_flags(df, frozen_min_run):
    """Satır bayrakları: is_zero (okuma = 0) ve is_frozen (≥ frozen_min_run saat aynı değer).

    Çıktı: DataFrame [N] (is_zero, is_frozen, run_length). Sıfırdan oluşan sabit seri ikisine birden girer.
    """
    run_length = constant_run_length(df)
    return pd.DataFrame({
        "is_zero": (df["meter_reading"].to_numpy() == 0),
        "is_frozen": run_length >= frozen_min_run,
        "run_length": run_length,
    }, index=df.index)


def length_bins(lengths):
    """Olay uzunluklarını rapor kutularına ayırır: 1, 2, 3–5, 6–12, 13–24, 25–72, 73–168, >168 saat."""
    edges = [0, 1, 2, 5, 12, 24, 72, 168, np.inf]
    labels = ["1", "2", "3–5", "6–12", "13–24", "25–72", "73–168", ">168"]
    return pd.cut(lengths, bins=edges, labels=labels)
