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


# ============================================================================= L-FE öznitelik grupları (§17)
# Bütün gruplar YALNIZ t ve öncesine bakar ve bina içinde hesaplanır. Izgara bina × saat düzenli olduğu için
# seri [saat, bina] matrisine çevrilir; pandas'ın rolling/shift işlemleri sütun (bina) başına ayrı çalışır,
# böylece bir binanın penceresi başka binaya taşmaz.
ROLL_WINDOWS = (6, 24, 168)
ROLL_STATS = ("mean", "std", "min", "max", "median", "iqr")
DISTINCT_WINDOWS = (6, 24)
PROFILE_MIN_HISTORY = 3          # aynı hafta-saatinin en az 3 geçmiş gözlemi yoksa profil NaN + bayrak

G1_COLS = (["diff_1h", "diff_24h", "absdiff_1h", "absdiff_24h", "change_ratio_1h", "changed"]
           + [f"n_distinct_{w}h" for w in DISTINCT_WINDOWS] + ["unchanged_h"])
G2_COLS = ([f"roll_{s}_{w}h" for w in ROLL_WINDOWS for s in ROLL_STATS]
           + [f"robust_z_{w}h" for w in ROLL_WINDOWS])
G3_COLS = ["log_ratio_med168h"]
G4_COLS = ["profile_mean", "profile_std", "profile_z", "profile_short"]
G5_COLS = ["missing_24h", "missing_168h", "gap_age_h", "last_gap_len"]
G6_COLS = ["primary_use"]
G7_COLS = ["l4semi_score"]
FE_GROUPS = {"G1": G1_COLS, "G2": G2_COLS, "G3": G3_COLS, "G4": G4_COLS, "G5": G5_COLS, "G6": G6_COLS,
             "G7": G7_COLS}
# behavior varyantı (§17.3): G1'deki değişmeden geçen süre ve G5 (eksiklik bağlamı) yok
FE_BEHAVIOR_DROP = ["unchanged_h"] + G5_COLS


def to_wide(grid, values):
    """Izgara sırasındaki [B·H] değerleri [H, B] DataFrame'e çevirir (sütun = bina)."""
    B = grid["building_id"].nunique()
    return pd.DataFrame(np.asarray(values, dtype=np.float64).reshape(B, -1).T)


def from_wide(wide):
    """[H, B] → ızgara sırasında [B·H] dizi (to_wide'ın tersi)."""
    return wide.to_numpy().T.reshape(-1)


def base_series(grid, variant):
    """Taban seri [H, B]: full → ham (eksik = NaN); behavior → bina içi ileri doldurulmuş (yalnız geçmiş)."""
    wide = to_wide(grid, grid["meter_reading"])
    return wide if variant == "full" else wide.ffill()


def run_length_wide(wide):
    """Her saatte, değerin değişmeden kaç saattir aynı olduğu (t dahil; NaN saatlerde NaN). [H, B]."""
    out = np.full(wide.shape, np.nan)
    values = wide.to_numpy()
    for j in range(values.shape[1]):                        # bina başına basit döngü
        run = 0
        prev = np.nan
        for i, v in enumerate(values[:, j]):
            if np.isnan(v):
                run, prev = 0, np.nan
                continue
            run = run + 1 if v == prev else 1
            prev = v
            out[i, j] = run
    return pd.DataFrame(out)


def distinct_count_wide(wide, window):
    """Son `window` saatte (t dahil) gözlenen farklı değer sayısı. [H, B]; NaN değerler sayılmaz."""
    values = wide.to_numpy()
    H, B = values.shape
    padded = np.vstack([np.full((window - 1, B), np.nan), values])       # başta yapay geçmiş = NaN
    out = np.empty((H, B))
    for j in range(B):
        win = np.lib.stride_tricks.sliding_window_view(padded[:, j], window)   # [H, window], yalnız geçmiş
        s = np.sort(win, axis=1)                                          # NaN'lar sona gider
        valid = ~np.isnan(s)
        new_value = np.ones_like(valid)
        new_value[:, 1:] = s[:, 1:] != s[:, :-1]
        out[:, j] = (new_value & valid).sum(axis=1)
    out[np.isnan(values)] = np.nan
    return pd.DataFrame(out)


def g1_change(wide):
    """G1 değer değişimi: farklar, değişim oranı, changed bayrağı, farklı değer sayısı, değişmeden geçen süre."""
    out = {}
    out["diff_1h"] = wide - wide.shift(1)
    out["diff_24h"] = wide - wide.shift(24)
    out["absdiff_1h"] = out["diff_1h"].abs()
    out["absdiff_24h"] = out["diff_24h"].abs()
    out["change_ratio_1h"] = out["diff_1h"] / (wide.shift(1).abs() + 1.0)     # +1 kWh: sıfıra bölme yok
    out["changed"] = (out["diff_1h"] != 0).astype(float).where(out["diff_1h"].notna())
    for w in DISTINCT_WINDOWS:
        out[f"n_distinct_{w}h"] = distinct_count_wide(wide, w)
    out["unchanged_h"] = run_length_wide(wide)
    return out


def g2_rolling(wide):
    """G2 kayan istatistikler (pencere t'yi içerir) ve robust z: (x − önceki pencerenin medyanı) / IQR'si."""
    out = {}
    for w in ROLL_WINDOWS:
        roll = wide.rolling(w, min_periods=max(2, w // 4))             # yalnız geçmiş: t-w+1 … t
        q25, q75 = roll.quantile(0.25), roll.quantile(0.75)
        out[f"roll_mean_{w}h"], out[f"roll_std_{w}h"] = roll.mean(), roll.std()
        out[f"roll_min_{w}h"], out[f"roll_max_{w}h"] = roll.min(), roll.max()
        out[f"roll_median_{w}h"], out[f"roll_iqr_{w}h"] = roll.median(), q75 - q25
        # z'nin referansı t-1'de biten pencere: mevcut değer kendi referansına girmesin
        ref_median = out[f"roll_median_{w}h"].shift(1)
        ref_iqr = out[f"roll_iqr_{w}h"].shift(1)
        out[f"robust_z_{w}h"] = (wide - ref_median) / (ref_iqr + 1e-3)
    return out


def g3_scale_free(wide):
    """G3: log(değer / son 168 saatin medyanı); log1p ile (sıfır okumalarda tanımlı). Medyan t-168 … t-1."""
    median = wide.shift(1).rolling(168, min_periods=24).median()
    return {"log_ratio_med168h": np.log1p(wide) - np.log1p(median)}


def g4_profile(wide, timestamps):
    """G4 profil sapması: aynı hafta-saatinin (dow·24 + saat) GEÇMİŞ değerlerinin genişleyen ortalama/std'si.

    timestamps: ızgaranın saatleri [H] (bütün binalarda aynı). Mevcut değer kendi profiline girmez (shift).
    Geçmiş gözlem sayısı PROFILE_MIN_HISTORY'den azsa ortalama/std/z NaN, profile_short = 1.
    """
    t = pd.DatetimeIndex(timestamps)
    slot = (t.dayofweek * 24 + t.hour).to_numpy()
    mean, std, count = (pd.DataFrame(np.nan, index=wide.index, columns=wide.columns) for _ in range(3))
    for s in np.unique(slot):
        rows = np.flatnonzero(slot == s)                          # aynı hafta-saatinin saatleri, zaman sırasıyla
        past = wide.iloc[rows].shift(1)                           # yalnız önceki haftalar
        mean.iloc[rows] = past.expanding().mean().to_numpy()
        std.iloc[rows] = past.expanding().std().to_numpy()
        count.iloc[rows] = past.notna().cumsum().to_numpy()
    short = count < PROFILE_MIN_HISTORY
    mean, std = mean.mask(short), std.mask(short)
    return {"profile_mean": mean, "profile_std": std, "profile_z": (wide - mean) / (std + 1e-3),
            "profile_short": short.astype(float)}


def g5_missing(grid):
    """G5 eksiklik bağlamı (ham gözlem durumundan; t'den ÖNCEKİ saatler): son 24/168 saatte eksik saat sayısı,
    son eksik saatten beri geçen süre, son tamamlanmış boşluğun uzunluğu. Önceki boşluk yoksa NaN."""
    missing = to_wide(grid, grid["observed"] == 0)
    out = {"missing_24h": missing.shift(1).rolling(24, min_periods=1).sum(),
           "missing_168h": missing.shift(1).rolling(168, min_periods=1).sum()}
    values = missing.to_numpy()
    age = np.full(values.shape, np.nan)
    last_len = np.full(values.shape, np.nan)
    for j in range(values.shape[1]):
        last_missing, run, completed = None, 0, np.nan
        for i in range(values.shape[0]):
            if last_missing is not None:
                age[i, j] = i - last_missing                       # t-1'e kadar bilinen bilgi
                last_len[i, j] = completed if run == 0 else run
            if values[i, j]:                                      # t eksik: sonraki saatler için güncelle
                last_missing, run = i, run + 1
            else:
                if run:
                    completed = run
                run = 0
    out["gap_age_h"] = pd.DataFrame(age)
    out["last_gap_len"] = pd.DataFrame(last_len)
    return out


def normalized_residual(errors, window):
    """G7 / L4 skoru: |1 adım hatası| / son `window` saatin (t-window … t-1) medyan hatası. [H, B] → [H, B]."""
    frame = pd.DataFrame(errors)
    scale = frame.shift(1).rolling(window, min_periods=24).median()     # yalnız geçmiş saatlerin hatası
    return frame / (scale + 1e-3)


def fe_group(grid, group, variant, primary_use=None, stacked=None):
    """Bir grubun öznitelikleri, ızgaranın her saati için. Çıktı: DataFrame [B·H, grup sütunları] (float32).

    primary_use: bina → kategori kodu (G6). stacked: ızgara sırasında [B·H] G7 skoru (pipeline'dan, OOF).
    """
    if group == "G0":
        return tabular_features(grid, variant)
    wide = base_series(grid, variant)
    if group == "G1":
        parts = g1_change(wide)
    elif group == "G2":
        parts = g2_rolling(wide)
    elif group == "G3":
        parts = g3_scale_free(wide)
    elif group == "G4":
        H = len(wide)
        parts = g4_profile(wide, grid["timestamp"].iloc[:H])
    elif group == "G5":
        parts = g5_missing(grid)
    elif group == "G6":
        codes = grid["building_id"].map(primary_use).to_numpy(dtype=np.float64)
        return pd.DataFrame({"primary_use": codes}, index=grid.index).astype("float32")
    elif group == "G7":
        return pd.DataFrame({"l4semi_score": np.asarray(stacked)}, index=grid.index).astype("float32")
    else:
        raise ValueError(group)
    columns = [c for c in FE_GROUPS[group] if variant == "full" or c not in FE_BEHAVIOR_DROP]
    X = pd.DataFrame({c: from_wide(parts[c]) for c in columns}, index=grid.index)
    return X.replace([np.inf, -np.inf], np.nan).astype("float32")


def fe_columns(groups, variant):
    """Seçilen grupların sütun listesi (G0 = §7 seti). behavior'da FE_BEHAVIOR_DROP yok."""
    cols = list(FEATURE_SETS[variant]) if "G0" in groups else []
    for g in groups:
        if g != "G0":
            cols += [c for c in FE_GROUPS[g] if variant == "full" or c not in FE_BEHAVIOR_DROP]
    return cols
