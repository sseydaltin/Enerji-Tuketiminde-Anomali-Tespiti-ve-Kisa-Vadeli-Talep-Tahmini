"""LEAD öznitelikleri (protokol §4, §6, §7) ve değerlendirme etiketleri (§3).

- Bütün öznitelikler YALNIZ GEÇMİŞE bakar (t ve öncesi).
- Mutlak tarih türevi YOKTUR (yılın günü, yıl, tarih sırası, zaman damgası); takvim yalnız döngüsel.
- Değerlendirme etiketleri (alt tür, paylaşılan/tekil) model girdisi DEĞİLDİR; ayrı fonksiyondadır.
"""

import numpy as np
import pandas as pd

from src.datasets.lead.eda import (anomaly_events, backward_run_length, constant_run_length, gap_neighbors,
                                   previous_missing, same_start_events)

HOUR = pd.Timedelta(hours=1)
CALENDAR = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend"]
LAGS = (1, 24, 168)
# full = Colab öznitelikleri (takvim döngüsel hâle getirildi) + eksiklik bayrağı + geriye dönük sabitlik süresi
FULL = (["meter_reading", "log_reading", "is_zero"] + CALENDAR
        + [f"lag_{h}h" for h in LAGS] + [f"log_diff_{h}h" for h in LAGS] + [f"lag_{h}h_missing" for h in LAGS]
        + ["prev_missing", "backward_run_length"])
# behavior = eksiklik ve sabitlikle ilgili her şey hariç (§7): bayraklar, sabitlik süresi, is_zero, log_diff_1h
SHORTCUT_COLUMNS = ["is_zero", "log_diff_1h", "lag_1h_missing", "lag_24h_missing", "lag_168h_missing",
                    "prev_missing", "backward_run_length"]
BEHAVIOR = [c for c in FULL if c not in SHORTCUT_COLUMNS]
FEATURE_SETS = {"full": FULL, "behavior": BEHAVIOR}
# Mutlak tarih türevi sayılan adlar: testler bu parçaları hiçbir öznitelik adında kabul etmez.
FORBIDDEN_NAME_PARTS = ("dayofyear", "day_of_year", "doy", "year", "timestamp", "date", "time_index",
                        "hours_since", "days_since", "ordinal", "week_of_year", "weekofyear")


def full_grid(df):
    """Her binayı verinin genel ilk–son saat ızgarasına genişletir (eksik saat = NaN okuma).

    Girdi: temiz tablo (bina, zaman sıralı). Çıktı: [200 × 8784] satır; observed (0/1) sütunu eklenir;
    eksik saatlerin anomaly değeri NaN'dır (değerlendirmeye girmez).
    """
    hours = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    index = pd.MultiIndex.from_product([np.sort(df["building_id"].unique()), hours],
                                       names=["building_id", "timestamp"])
    grid = df.set_index(["building_id", "timestamp"]).reindex(index).reset_index()
    grid["observed"] = grid["meter_reading"].notna().astype("int8")
    return grid


def calendar(timestamps):
    """Döngüsel takvim: saat, haftanın günü, ay (sin/cos) ve hafta sonu. [N, 7]."""
    t = pd.DatetimeIndex(timestamps)
    out = pd.DataFrame(index=range(len(t)))
    for name, value, period in (("hour", t.hour, 24), ("dow", t.dayofweek, 7), ("month", t.month - 1, 12)):
        out[f"{name}_sin"] = np.sin(2 * np.pi * value / period)
        out[f"{name}_cos"] = np.cos(2 * np.pi * value / period)
    out["is_weekend"] = (t.dayofweek >= 5).astype("int8")
    return out[CALENDAR]


def tabular_features(grid, feature_set):
    """LightGBM / Isolation Forest öznitelikleri ızgaranın her saati için (yalnız geçmiş).

    full: gecikmeler ham seriden (eksikse NaN) + eksiklik bayrakları + sabitlik süresi (Colab'daki gibi).
    behavior: gecikmeler bina bazında İLERİ DOLDURULMUŞ seriden (NaN üzerinden eksiklik bilgisi sızmasın);
    eksiklik/sabitlik sütunları yok.
    Çıktı: DataFrame [ızgara satırı, öznitelik] (float32).
    """
    g = grid
    reading = g["meter_reading"]
    by_building = g.groupby("building_id", sort=False)
    filled = by_building["meter_reading"].ffill()            # yalnız geçmişten doldurma
    base = reading if feature_set == "full" else filled
    X = pd.DataFrame(index=g.index)
    X["meter_reading"] = base
    X["log_reading"] = np.log1p(base)
    X["is_zero"] = (reading == 0).astype("float32")
    X = pd.concat([X, calendar(g["timestamp"]).set_index(g.index)], axis=1)
    source = base.groupby(g["building_id"], sort=False)
    for h in LAGS:
        past = source.shift(h)                                # aynı binanın h saat önceki değeri
        X[f"lag_{h}h"] = past
        X[f"log_diff_{h}h"] = X["log_reading"] - np.log1p(past)
        X[f"lag_{h}h_missing"] = reading.groupby(g["building_id"], sort=False).shift(h).isna().astype("float32")
    observed = g[g["observed"] == 1]
    X["prev_missing"] = np.nan
    X["backward_run_length"] = np.nan
    X.loc[observed.index, "prev_missing"] = previous_missing(observed.reset_index(drop=True)).astype("float32")
    X.loc[observed.index, "backward_run_length"] = backward_run_length(observed.reset_index(drop=True))
    return X[FEATURE_SETS[feature_set]].replace([np.inf, -np.inf], np.nan).astype("float32")


def evaluation_labels(df, frozen_min_run=3):
    """Değerlendirme etiketleri (§3; model girdisi DEĞİL): her temiz satır için alt tür ve dilim.

    Çıktı: DataFrame [temiz satır] (subtype, shared, event_id, event_subtype). Normal saatlerde subtype
    "normal", shared NaN. Alt tür önceliği: gap_edge > frozen (≥ frozen_min_run) > other.
    Olay alt türü: olay saatlerinin çoğunluk alt türü (eşitlikte öncelik sırası).
    """
    anomaly = df["anomaly"].to_numpy() == 1
    gap = gap_neighbors(df)
    frozen = constant_run_length(df) >= frozen_min_run
    subtype = np.where(~anomaly, "normal", np.where(gap, "gap_edge", np.where(frozen, "frozen", "other")))
    events = same_start_events(anomaly_events(df))
    events["event_id"] = np.arange(len(events))
    key = pd.Series(np.arange(len(df)), index=pd.MultiIndex.from_frame(df[["building_id", "timestamp"]]))
    event_id = np.full(len(df), -1)
    for e in events.itertuples():                              # olay saatlerine olay kimliği
        first = key[(e.building_id, e.start)]
        event_id[first:first + e.length_hours] = e.event_id
    out = pd.DataFrame({"subtype": subtype, "event_id": event_id})
    priority = {"gap_edge": 0, "frozen": 1, "other": 2}
    majority = (out[out["event_id"] >= 0].groupby("event_id")["subtype"]
                .agg(lambda s: sorted(s.value_counts().items(), key=lambda kv: (-kv[1], priority[kv[0]]))[0][0]))
    shared = events.set_index("event_id")["n_buildings_same_start"] >= 2
    out["event_subtype"] = out["event_id"].map(majority)
    out["shared"] = out["event_id"].map(shared)
    return out
