"""LEAD demo servisleri: kayıtlı final test skorlarını, val eşiklerini ve gerçek etiketleri okur.

Bu modül HİÇBİR model eğitmez ve skor hesaplamaz. Skorlar tek seferlik final testin kayıtlı çıktısıdır
(`results/lead/scores/test_bina_{seed}_{model}_{set}.npy`, git dışı), eşikler val'de seçilmiştir
(`results/lead/tables/test_results.csv`'deki `threshold` sütunu = val F1-max eşiği). Etiket alt türleri repodaki
`evaluation_labels` fonksiyonundan gelir (yalnız değerlendirme içindir, model girdisi değildir).
"""

import numpy as np
import pandas as pd
import streamlit as st

from src.core.config import PROJECT_ROOT, load_config
from src.core.results_io import resolve_path
from src.datasets.lead.features import evaluation_labels
from src.datasets.lead.splits import building_split

# Gösterilen modeller (sıra = grafik sırası). Final model protokol §21.2.
LEAD_MODELS = {
    ("L1", "behavior"): "LightGBM, behavior seti (final)",
    ("L1-FE", "full"): "LightGBM + öznitelik grupları (full)",
    ("L4", "-"): "Seq2Seq tahmin artığı (etiketsiz)",
}
FINAL_LEAD_MODEL = ("L1", "behavior")
# Sonuçlar sayfasındaki val→test tablosu: iç kodlar yerine okunur adlar (sıra = tablo sırası).
LEAD_LABELS = {
    ("L1", "behavior"): "LightGBM, behavior (final)",
    ("L1-FE", "full"): "LightGBM + öznitelik grupları, full",
    ("L1-FE", "behavior"): "LightGBM + öznitelik grupları, behavior",
    ("L2-FE", "full"): "GRU + öznitelik grupları, full",
    ("L2-FE", "behavior"): "GRU + öznitelik grupları, behavior",
    ("L5-FE", "full"): "LightGBM + GRU topluluğu, full",
    ("L5-FE", "behavior"): "LightGBM + GRU topluluğu, behavior",
    ("L4", "-"): "Seq2Seq tahmin artığı (etiketsiz)",
    ("L4-semi", "-"): "Seq2Seq tahmin artığı (yarı denetimli)",
    ("L0-A0", "-"): "Hareketli z-skoru",
    ("L0-R3", "-"): "Kural: eksik saat veya sabit okuma",
    ("IF", "behavior"): "Isolation Forest (literatür referansı)",
    ("L1-OFF", "full"): "LightGBM, iki yönlü öznitelik (offline, yalnız kıyas)",
}
SUBTYPE_LABELS = {"gap_edge": "Boşluk kenarı", "frozen": "Donmuş okuma", "other": "Davranışsal"}
SLICE_LABELS = {"other": "Davranışsal (other)", "genel": "Genel", "gap_edge": "Boşluk kenarı", "frozen": "Donmuş okuma",
                "paylasilan": "Paylaşılan olay", "tekil": "Tekil olay"}
SEEDS = (42, 7, 2026)


def lead_paths():
    """LEAD config'i ve dosya yolları: (config, temiz veri, tablolar klasörü, skorlar klasörü)."""
    config = load_config(PROJECT_ROOT / "configs" / "lead.yaml")
    processed = resolve_path(PROJECT_ROOT, config["paths"]["processed_file"])
    tables = resolve_path(PROJECT_ROOT, config["paths"]["tables_dir"])
    return config, processed, tables, tables.parent / "scores"


def missing_files(seed):
    """Sayfanın ihtiyaç duyduğu ama yerelde olmayan dosyalar (veri ve skorlar git'e girmez)."""
    _, processed, tables, scores = lead_paths()
    needed = [processed, tables / "test_results.csv"]
    needed += [scores / f"test_bina_{seed}_{m}_{s}.npy" for m, s in LEAD_MODELS]
    return [str(p) for p in needed if not p.is_file()]


@st.cache_data(show_spinner=False)
def load_test_frame(seed):
    """Bir bölme seed'inin test binaları: [satır] building_id, timestamp, meter_reading, anomaly, subtype + skorlar.

    Satır sırası final testtekiyle aynıdır (temiz tablo bina-zaman sıralı; test binalarının satırları), bu yüzden
    kayıtlı skor dizileri doğrudan sütun olarak eklenir; uzunluk eşitliği denetlenir.
    """
    config, processed, _, scores = lead_paths()
    clean = pd.read_parquet(processed).sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    labels = evaluation_labels(clean, config["evaluation"]["frozen_min_run"])
    split = building_split(np.sort(clean["building_id"].unique()), seed, config["split"]["n_val"],
                           config["split"]["n_test"])
    test = clean["building_id"].isin(split["test"]).to_numpy()
    frame = clean.loc[test, ["building_id", "timestamp", "meter_reading", "anomaly"]].reset_index(drop=True)
    frame["subtype"] = labels.loc[test, "subtype"].to_numpy()
    for model, fset in LEAD_MODELS:
        values = np.load(scores / f"test_bina_{seed}_{model}_{fset}.npy")
        if len(values) != len(frame):
            raise ValueError(f"{model}/{fset} skor uzunluğu {len(values)} ≠ test satırı {len(frame)}")
        frame[f"{model}|{fset}"] = values
    return frame


@st.cache_data(show_spinner=False)
def load_thresholds(seed):
    """{(model, set): eşik} — val'de seçilmiş (F1-max) ve testte aynen uygulanmış eşikler."""
    _, _, tables, _ = lead_paths()
    results = pd.read_csv(tables / "test_results.csv")
    part = results[(results["bolme"] == "bina") & (results["seed"] == seed) & (results["dilim"] == "genel")]
    return {key: float(part[(part["model"] == key[0]) & (part["set"] == key[1])]["threshold"].iloc[0])
            for key in LEAD_MODELS}


@st.cache_data(show_spinner=False)
def slice_results():
    """Final test, alt tür (dilim) bazında: model × dilim, 3 seed ort. ± std (PR-AUC, ROC-AUC, F1, olay recall)."""
    _, _, tables, _ = lead_paths()
    results = pd.read_csv(tables / "test_results.csv")
    results = results[results["bolme"] == "bina"]
    rows = []
    for (model, fset), label in LEAD_MODELS.items():
        part = results[(results["model"] == model) & (results["set"] == fset)]
        for dilim, name in SLICE_LABELS.items():
            x = part[part["dilim"] == dilim]
            rows.append({"model": label, "dilim": name, "n_seed": x["seed"].nunique(),
                         **{f"{m}_mean": x[m].mean() for m in ("pr_auc", "roc_auc", "f1", "event_recall")},
                         **{f"{m}_std": x[m].std() for m in ("pr_auc", "roc_auc", "f1", "event_recall")},
                         "taban": (x["n_pos"] / (x["n_pos"] + x["n_neg"])).mean()})
    return pd.DataFrame(rows)


def building_subtype_table(frame, building, thresholds):
    """Seçili binada alt tür başına anomali saati ve her modelin eşiği aşan (yakalanan) saat sayısı.

    Yalnız sayma yapar (skorlar ve eşikler kayıtlıdır). Çıktı: DataFrame [alt tür].
    """
    part = frame[frame["building_id"] == building]
    rows = []
    for subtype, label in SUBTYPE_LABELS.items():
        hours = part[part["subtype"] == subtype]
        row = {"Alt tür": label, "Anomali saati": len(hours)}
        for key, name in LEAD_MODELS.items():
            caught = int((hours[f"{key[0]}|{key[1]}"].fillna(-np.inf) >= thresholds[key]).sum())
            row[f"Yakalanan: {name}"] = caught
        rows.append(row)
    normal = part[part["subtype"] == "normal"]
    row = {"Alt tür": "Normal (yanlış alarm)", "Anomali saati": len(normal)}
    for key, name in LEAD_MODELS.items():
        row[f"Yakalanan: {name}"] = int((normal[f"{key[0]}|{key[1]}"].fillna(-np.inf) >= thresholds[key]).sum())
    rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def building_f1(seed):
    """Final modelin test binalarındaki bina bazlı F1'i (kayıtlı test_building_f1.csv; anomalisi olan binalar).

    Çıktı: Series [building_id → F1]. Yalnız görüntüleme için (varsayılan / kolay / zor bina seçimi); hiçbir model,
    eşik ya da ayar seçimi buna göre yapılmaz.
    """
    _, _, tables, _ = lead_paths()
    table = pd.read_csv(tables / "test_building_f1.csv")
    part = table[(table["model"] == FINAL_LEAD_MODEL[0]) & (table["set"] == FINAL_LEAD_MODEL[1]) & (table["seed"] == seed)]
    return part.set_index("building_id")["f1"].sort_index()


def median_building(f1):
    """F1'i medyana en yakın bina (Talep sayfasındaki "medyana en yakın gün" ile aynı mantık; eşitlikte küçük kimlik)."""
    distance = (f1 - f1.median()).abs()
    return int(distance.sort_values(kind="stable").index[0])


def busiest_window(frame, building, days=14):
    """Binanın en çok anomali saati içeren `days` günlük aralığı: (başlangıç, bitiş). Yalnız görünüm aralığı."""
    part = frame[frame["building_id"] == building].set_index("timestamp")["anomaly"]
    daily = part.resample("D").sum()
    totals = daily.rolling(days, min_periods=1).sum()
    end = totals.idxmax()
    start = max(daily.index[0], end - pd.Timedelta(days=days - 1))
    return start, end + pd.Timedelta(days=1)


@st.cache_data(show_spinner=False)
def val_test_summary():
    """Kayıtlı val ve test özetleri (scripts/lead_results_report.py çıktısı; 3 seed ort. ± std, bina bölmesi)."""
    _, _, tables, _ = lead_paths()
    return pd.read_csv(tables / "test_summary_val_vs_test.csv")
