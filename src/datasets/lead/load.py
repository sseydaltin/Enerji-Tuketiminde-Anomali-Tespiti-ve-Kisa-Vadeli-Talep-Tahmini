"""LEAD1.0 (Kaggle "energy-anomaly-detection") train.csv: okuma ve temizlik.

Temizlik adımları `experiments/archive/lead_colab_ilk.ipynb` (3. hücre) ile BİREBİR aynıdır;
yalnız fonksiyonlara ayrıldı ve günlük tablosuna önce/sonra satır sayıları eklendi.
Veri Kaggle'dan elle indirilir; bu modül ağa bağlanmaz ve kimlik bilgisi kullanmaz.
"""

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["building_id", "timestamp", "meter_reading", "anomaly"]


def load_raw(path):
    """train.csv'yi okur.

    Girdi: dosya yolu. Çıktı: DataFrame [N, 4] (building_id, timestamp, meter_reading, anomaly), ham tipler.
    Neden: Dosya yoksa sessizce boş veriyle devam etmek yerine nereden alınacağını söyleyen açık hata verilir.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"LEAD verisi bulunamadı: {path}. Kaggle 'energy-anomaly-detection' yarışmasının train.csv "
            "dosyasını elle indirip bu yola koyun (kimlik bilgisi koda yazılmaz).")
    return pd.read_csv(path, usecols=COLUMNS)


def clean(raw):
    """Notebook'taki temizlik adımlarını sırayla uygular; orijinal tablo değişmez.

    Girdi: ham DataFrame. Çıktı: (temiz DataFrame, temizlik günlüğü DataFrame).
    Adımlar (notebook sırası):
      1. Tip dönüşümü: timestamp → datetime, diğerleri → sayı (dönüşmeyen → NaN); ±inf → NaN.
      2. Eksik/geçersiz temel alanlar: kimlik, zaman veya tüketimi eksik olanlar ve etiketi 0/1 olmayanlar.
      3. Negatif tüketim (geçersiz kabul edilir).
      4. Tam tekrarlar (dört sütunu da aynı satırlar; biri kalır).
      5. Çakışan bina-saat kayıtları: aynı bina ve saatte FARKLI satırlar varsa hiçbiri tutulmaz
         (keyfî birini seçmek etikete veya tüketime dair uydurma bir karar olurdu).
    Sonra tipler daraltılır ve satırlar (bina, zaman) sırasına konur.
    Not: Eksik tüketimli satırlar (2. adım) zaman ızgarasında boşluk olarak kalır; doldurulmaz.
    """
    df = raw.copy()                       # orijinali yerinde değiştirme
    log = []

    def record(step, before):
        log.append({"İşlem": step, "Önce": before, "Sonra": len(df), "Silinen satır": before - len(df)})

    initial = len(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for column in ["building_id", "meter_reading", "anomaly"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    before = len(df)
    df = df.dropna(subset=["building_id", "timestamp", "meter_reading"])
    df = df[df["anomaly"].isin([0, 1])].copy()
    record("Eksik/geçersiz temel alanlar", before)

    before = len(df)
    df = df[df["meter_reading"] >= 0].copy()
    record("Negatif tüketim", before)

    before = len(df)
    df = df.drop_duplicates()
    record("Tam tekrarlar", before)

    before = len(df)
    conflicting = df.duplicated(["building_id", "timestamp"], keep=False)
    df = df.loc[~conflicting].copy()
    record("Çakışan bina-zaman kayıtları", before)

    df["building_id"] = df["building_id"].astype("int32")
    df["meter_reading"] = df["meter_reading"].astype("float32")
    df["anomaly"] = df["anomaly"].astype("int8")
    df = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)

    log.append({"İşlem": "Toplam", "Önce": initial, "Sonra": len(df), "Silinen satır": initial - len(df)})
    return df, pd.DataFrame(log)


SITE_COLUMNS = ["building_id", "site_id", "primary_use"]


def extract_sites(features_path):
    """Kaggle train_features.csv'den (veya onu içeren zip'ten) YALNIZ building_id → site_id, primary_use.

    Girdi: train_features.csv ya da energy-anomaly-detection.zip yolu.
    Çıktı: DataFrame [bina] (building_id, site_id, primary_use).
    Neden: Site, bölme (siteye göre sağlamlık kontrolü) ve olay analizi için gerekir. Dosyanın diğer
    sütunları (hava, hedef kodlamaları vb.) model girdisi DEĞİLDİR ve okunmaz (bkz. deney günlüğü).
    Her binanın tek bir site ve kullanım türü olduğu denetlenir.
    """
    import zipfile
    path = Path(features_path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive, archive.open("train_features.csv") as handle:
            frame = pd.read_csv(handle, usecols=SITE_COLUMNS)
    else:
        frame = pd.read_csv(path, usecols=SITE_COLUMNS)
    sites = frame.drop_duplicates()
    if sites["building_id"].duplicated().any():
        raise ValueError("Bir binaya birden fazla site_id/primary_use eşlenmiş.")
    return sites.sort_values("building_id").reset_index(drop=True)
