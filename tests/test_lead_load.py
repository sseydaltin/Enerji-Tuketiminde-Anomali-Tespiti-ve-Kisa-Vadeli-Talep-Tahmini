"""LEAD temizliği: çakışan kayıt kalmıyor, tipler doğru, idempotent, orijinal değişmiyor, dosya yoksa açık hata."""

import numpy as np
import pandas as pd
import pytest

from src.datasets.lead.load import clean, load_raw


def _raw():
    """Her temizlik adımını tetikleyen küçük ham tablo (CSV'den okunmuş gibi metin/sayı karışık)."""
    return pd.DataFrame({
        "building_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, "x", 3],
        "timestamp": ["2016-01-01 00:00:00", "2016-01-01 01:00:00", "2016-01-01 01:00:00",   # tam tekrar
                      "2016-01-01 02:00:00", "2016-01-01 03:00:00",
                      "2016-01-01 00:00:00", "2016-01-01 00:00:00",                          # çakışma
                      "2016-01-01 01:00:00", "bozuk", "2016-01-01 00:00:00", "2016-01-01 00:00:00"],
        "meter_reading": [1.0, 2.0, 2.0, np.nan, -1.0, 5.0, 6.0, np.inf, 3.0, 1.0, 4.0],
        "anomaly": [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 2],                                        # 2 geçersiz etiket
    })


def test_steps_and_log():
    """Her adım beklenen satırları siler; günlük toplamı tutarlı."""
    df, log = clean(_raw())
    removed = dict(zip(log["İşlem"], log["Silinen satır"]))
    # NaN tüketim, inf→NaN, bozuk zaman, bozuk kimlik, etiket=2 → 5 satır
    assert removed["Eksik/geçersiz temel alanlar"] == 5
    assert removed["Negatif tüketim"] == 1
    assert removed["Tam tekrarlar"] == 1
    assert removed["Çakışan bina-zaman kayıtları"] == 2       # bina 2, 00:00 – iki satır da gider
    assert removed["Toplam"] == len(_raw()) - len(df) == 9
    assert list(df["meter_reading"]) == [1.0, 2.0]


def test_no_conflicting_building_hours_remain():
    """Temiz tabloda (bina, saat) tekil."""
    df, _ = clean(_raw())
    assert not df.duplicated(["building_id", "timestamp"]).any()


def test_types_after_cleaning():
    """Tipler daraltılmış ve geçerli; zaman datetime, etiket yalnız 0/1, tüketim ≥ 0."""
    df, _ = clean(_raw())
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert df["building_id"].dtype == np.int32
    assert df["meter_reading"].dtype == np.float32
    assert df["anomaly"].dtype == np.int8
    assert set(df["anomaly"]) <= {0, 1}
    assert (df["meter_reading"] >= 0).all() and np.isfinite(df["meter_reading"]).all()
    assert df.equals(df.sort_values(["building_id", "timestamp"]).reset_index(drop=True))


def test_cleaning_is_idempotent():
    """Temiz tabloyu yeniden temizlemek hiçbir şey silmez ve aynı tabloyu verir."""
    once, _ = clean(_raw())
    twice, log = clean(once)
    pd.testing.assert_frame_equal(once, twice)
    assert (log["Silinen satır"] == 0).all()


def test_original_not_modified():
    """Temizlik ham tabloyu yerinde değiştirmez."""
    raw = _raw()
    copy = raw.copy()
    clean(raw)
    pd.testing.assert_frame_equal(raw, copy)


def test_missing_file_gives_clear_error(tmp_path):
    """Dosya yoksa nereden alınacağını söyleyen açık hata."""
    with pytest.raises(FileNotFoundError, match="energy-anomaly-detection"):
        load_raw(tmp_path / "yok.csv")
