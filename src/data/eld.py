"""UCI ElectricityLoadDiagrams20112014 (ELD): şebeke düzeyi toplam yükü saatlik tabloya çevirir.

Kaynak: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
(370 müşteri, Portekiz, 15 dakikalık kW; ';' ayırıcı, ',' ondalık).
Zaman damgası aralık SONUdur (ilk satır 2011-01-01 00:15). Saat H = (H:00, H+1:00]
aralığındaki dört çeyreğin ortalaması (kW).

Yaz saati (UCI açıklaması): Mart geçiş gününde 01:00–02:00 değerleri sıfır,
Ekim geçiş gününde iki saatin toplamı. Veride bu dört çeyrek 01:00, 01:15,
01:30, 01:45 damgalı (keşifle doğrulandı). Düzeltme toplam yük üzerinde, 15
dakikalık düzeyde yapılır; etkilenen saatler is_imputed=1 ile işaretlenir.
"""

import argparse
import gc
from copy import deepcopy

import numpy as np
import pandas as pd

from src.data.load import PROJECT_ROOT, load_config
from src.data.preprocess import add_calendar_features
from src.utils.console import setup_console
from src.utils.results_io import resolve_path

DST_STAMPS = ("01:00", "01:15", "01:30", "01:45")   # etkilenen çeyreklerin damgaları
# eld.yaml'da bütünüyle DEĞİŞTİRİLEN (birleştirilmeyen) bölümler: household'un val_dev/val_hold
# tarihleri ve F10-opt/F5-opt topluluk anahtarları ELD config'ine sızmasın.
REPLACED_SECTIONS = (("split",), ("anomaly", "residual", "ensemble"))


def deep_merge(base, override, replaced=(), path=()):
    """İç içe sözlükleri birleştirir; override kazanır. `replaced` yollarındaki sözlükler bütünüyle değişir."""
    result = deepcopy(base)
    for key, value in override.items():
        key_path = path + (key,)
        if isinstance(value, dict) and isinstance(result.get(key), dict) and key_path not in replaced:
            result[key] = deep_merge(result[key], value, replaced, key_path)
        else:
            result[key] = deepcopy(value)
    return result


def eld_config(model_config=None):
    """ELD config'i: household config.yaml + (isteğe bağlı) model config'inin forecast'ı + configs/eld.yaml.

    Girdi: model_config = household Optuna config yolu (ör. configs/optuna_seq2seq_best.yaml);
    verilirse yalnız `forecast` bölümü alınır (hiperparametreler AYNEN).
    Neden: Yöntem ayarları household'dan değişmeden gelir; eld.yaml yalnız veri
    setine özgü ayarları (yollar, tarihler, hedef adı, tatil ülkesi, seed'ler) içerir.
    """
    config = load_config()
    if model_config is not None:
        config["forecast"] = load_config(PROJECT_ROOT / model_config)["forecast"]
    return deep_merge(config, load_config(PROJECT_ROOT / "configs/eld.yaml"), REPLACED_SECTIONS)


def read_raw(path):
    """Ham dosyayı float32 sütunlarla okur. Çıktı: [140.256 çeyrek, 370 müşteri] DataFrame (kW)."""
    columns = pd.read_csv(path, sep=";", nrows=0).columns
    return pd.read_csv(path, sep=";", decimal=",", index_col=0, parse_dates=True,
                       dtype={column: np.float32 for column in columns[1:]})


def active_clients(raw, active_before):
    """`active_before` damgasına kadar en az bir sıfır dışı değeri olan müşteriler.

    Neden: Geç başlayan müşterilerin başlangıç öncesi sıfırla doldurulmuş;
    onları toplama katmak toplam yükte yapay bir artış eğilimi yaratır.
    """
    return raw.columns[(raw.loc[:active_before] != 0).any()]


def dst_change_days(years):
    """Avrupa yaz saati geçiş günleri: Mart ve Ekim'in son Pazarları. Çıktı: (mart, ekim) listeleri."""
    def last_sunday(year, month):
        last = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
        return last - pd.Timedelta(days=(last.dayofweek - 6) % 7)
    return [last_sunday(y, 3) for y in years], [last_sunday(y, 10) for y in years]


def fix_dst(total, years):
    """Yaz saati kalıntılarını toplam yükte (15 dk) tespit edip düzeltir.

    Girdi: toplam yük Series (15 dk, aralık sonu damgalı, kW), düzeltilecek yıllar
    (çalışma dönemi; 2011'de bu desen görülmedi).
    Çıktı: (düzeltilmiş seri, düzeltilen damgalar [bool Series], rapor tablosu).
    Kurallar:
    - Mart: dört çeyrek komşu değerlerin < %5'i olmalı (tespit); 00:45 ve 02:00
      damgaları arasında doğrusal ara değerle doldurulur.
    - Ekim: dört çeyreğin ortalaması komşuların 1,5–2,5 katı olmalı (tespit); ikiye bölünür.
    Beklenen desen görülmezse hata verilir (sessizce düzeltme yapılmaz).
    """
    fixed = total.astype(np.float64).copy()
    changed = pd.Series(False, index=total.index)
    march_days, october_days = dst_change_days(years)
    rows = []
    for kind, days in (("mart", march_days), ("ekim", october_days)):
        for day in days:
            stamps = pd.DatetimeIndex([f"{day.date()} {s}" for s in DST_STAMPS])
            before, after = day + pd.Timedelta("00:45:00"), day + pd.Timedelta("02:00:00")
            if not (stamps.isin(fixed.index).all() and before in fixed.index and after in fixed.index):
                continue
            neighbors = (fixed[before] + fixed[after]) / 2
            ratio = fixed[stamps].mean() / neighbors
            if kind == "mart":
                if ratio >= 0.05:
                    raise ValueError(f"{day.date()}: beklenen Mart sıfırları yok (oran {ratio:.2f}).")
                steps = np.arange(1, 5) / 5           # 00:45 → 02:00 arasında 4 ara nokta
                fixed[stamps] = fixed[before] + steps * (fixed[after] - fixed[before])
            else:
                if not 1.5 <= ratio <= 2.5:
                    raise ValueError(f"{day.date()}: beklenen Ekim çift saati yok (oran {ratio:.2f}).")
                fixed[stamps] = fixed[stamps] / 2
            changed[stamps] = True
            rows.append({"day": day.date(), "type": kind, "ratio_to_neighbors": ratio})
    return fixed, changed, pd.DataFrame(rows)


def to_hourly(total, changed):
    """15 dk → saatlik ortalama kW. Saat H = (H:00, H+1:00] (damga aralık sonu).

    Çıktı: [saat, (yük, is_imputed)] — düzeltilmiş çeyrek içeren saat is_imputed=1.
    """
    load = total.resample("1h", closed="right", label="left").mean()
    imputed = changed.resample("1h", closed="right", label="left").max().astype("int8")
    return pd.DataFrame({"load": load, "is_imputed": imputed})


def build_hourly(config):
    """Ham dosyadan saatlik ELD tablosu: [saat, (hedef, is_imputed, takvim)] ve rapor.

    Bellek: ham tablo float32 (~210 MB); müşteri seçimi ve toplamdan sonra silinir.
    """
    settings = config["eld"]
    raw = read_raw(resolve_path(PROJECT_ROOT, settings["raw_file"]))
    clients = active_clients(raw, settings["active_before"])
    total = raw[clients].sum(axis=1)                       # [çeyrek] toplam kW
    report = {"clients_total": raw.shape[1], "clients_selected": len(clients)}
    del raw
    gc.collect()

    years = range(pd.Timestamp(settings["start"]).year, pd.Timestamp(settings["end"]).year + 1)
    fixed, changed, dst_report = fix_dst(total, years)
    hourly = to_hourly(fixed, changed)
    hourly = hourly.loc[settings["start"]:f"{settings['end']} 23:00"]
    target = config["preprocess"]["target"]
    hourly = hourly.rename(columns={"load": target})
    if hourly[target].isna().any() or len(hourly) != len(
            pd.date_range(settings["start"], f"{settings['end']} 23:00", freq="h")):
        raise ValueError("Saatlik ELD tablosunda eksik saat var.")
    hourly = add_calendar_features(hourly, config["preprocess"]["holiday_country"])
    report.update({"hours": len(hourly), "imputed_hours": int(hourly["is_imputed"].sum()),
                   "load_mean_kw": hourly[target].mean(), "load_min_kw": hourly[target].min(),
                   "load_max_kw": hourly[target].max()})
    return hourly, report, dst_report


def main():
    """data/processed/eld_hourly.parquet ve results/eld/tables/eld_data_report.csv yazar."""
    setup_console()
    argparse.ArgumentParser(description=__doc__).parse_args()
    config = eld_config()
    hourly, report, dst_report = build_hourly(config)
    path = resolve_path(PROJECT_ROOT, config["paths"]["processed_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(path)
    tables_dir = resolve_path(PROJECT_ROOT, config["paths"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([report]).to_csv(tables_dir / "eld_data_report.csv", index=False)
    dst_report.to_csv(tables_dir / "eld_dst_fixes.csv", index=False)
    print(pd.Series(report).to_string())
    print(dst_report.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
