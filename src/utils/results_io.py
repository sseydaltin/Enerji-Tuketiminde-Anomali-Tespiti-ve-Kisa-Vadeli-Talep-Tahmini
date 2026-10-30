"""Tahmin sonuçlarını tüm modeller için tek biçimde kaydeden yardımcılar."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.metrics import calculate_metrics, metrics_by_horizon


def resolve_path(root, configured_path):
    """Config'teki göreli yolu proje köküne göre mutlak yola çevirir.

    Girdi: proje kökü (Path) ve config'teki mutlak/göreli yol.
    Çıktı: Path nesnesi.
    """
    path = Path(configured_path)
    return path if path.is_absolute() else Path(root) / path


def upsert_csv(path, rows, key_columns):
    """CSV'deki aynı anahtarlı satırları yeniler, diğer satırları korur.

    Girdi: dosya yolu, sözlük satırları, tekil kaydı belirleyen sütunlar.
    Çıktı: güncellenmiş CSV dosyası.
    Neden: Aynı exp_id tekrar koşulursa satır çoğalmaz; farklı exp_id'ler
    (iyileştirmeyenler dahil) tabloda kalır.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_frame = pd.DataFrame(rows)
    if path.exists():
        old_frame = pd.read_csv(path)
        new_keys = set(new_frame[key_columns].itertuples(index=False, name=None))
        old_keys = old_frame[key_columns].itertuples(index=False, name=None)
        old_frame = old_frame.loc[[key not in new_keys for key in old_keys]]
        new_frame = pd.concat((old_frame, new_frame), ignore_index=True)
    new_frame.to_csv(path, index=False)


def record_forecast_results(
    project_root,
    config,
    split,
    exp_id,
    model_label,
    y_true,
    y_pred,
    target_times,
    train_time_s,
    device,
    feature_set,
    lookback,
    note="",
):
    """Bir koşunun metriklerini hesaplar; özet, ufuk ve tahminleri kaydeder.

    Girdi: y_true/y_pred [pencere, ufuk] kW dizileri (inverse transform
    SONRASI), target_times [pencere] her pencerenin ilk hedef zamanı.
    Çıktı: calculate_metrics sözlüğü. Yan etki olarak:
      - results/tables/forecast_{split}_results.csv  (anahtar: exp_id)
      - results/tables/forecast_{split}_horizon.csv  (anahtar: exp_id, horizon)
      - results/predictions/{exp_id}_{split}.npz     (y_true, y_pred, zamanlar)
    Neden: Tüm modeller aynı fonksiyonla, aynı metrik ve aynı dosya biçimiyle
    değerlendirilir; kayıtlı tahminler sonradan tutarlılık kontrolü, hata
    analizi ve topluluk (ensemble) için yeniden eğitim gerektirmeden kullanılır.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mape_min_y = config["evaluation"]["mape_min_y"]
    scores = calculate_metrics(y_true, y_pred, mape_min_y=mape_min_y)

    tables_dir = resolve_path(project_root, config["paths"]["tables_dir"])
    predictions_dir = resolve_path(
        project_root, config["paths"]["predictions_dir"]
    )

    summary = {
        "exp_id": exp_id,
        "split": split,
        "model": model_label,
        "feature_set": feature_set,
        "lookback": lookback,
        "windows": len(y_true),
        **scores,
        "train_time_s": train_time_s,
        "device": device,
        "date": date.today().isoformat(),
        "note": note,
    }
    upsert_csv(tables_dir / f"forecast_{split}_results.csv", [summary], ["exp_id"])

    horizon_table = metrics_by_horizon(y_true, y_pred, mape_min_y=mape_min_y)
    horizon_table.insert(0, "model", model_label)
    horizon_table.insert(0, "exp_id", exp_id)
    upsert_csv(
        tables_dir / f"forecast_{split}_horizon.csv",
        horizon_table.to_dict("records"),
        ["exp_id", "horizon"],
    )

    predictions_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        predictions_dir / f"{exp_id}_{split}.npz",
        y_true=y_true.astype(np.float32),
        y_pred=y_pred.astype(np.float32),
        # Zamanlar int64 nanosaniye olarak saklanır (npz'de datetime yok).
        target_times=pd.DatetimeIndex(target_times).asi8,
    )
    return scores


def load_predictions(project_root, config, exp_id, split="val"):
    """record_forecast_results ile kaydedilmiş tahminleri okur.

    Çıktı: (y_true [N, ufuk], y_pred [N, ufuk], target_times DatetimeIndex).
    """
    predictions_dir = resolve_path(
        project_root, config["paths"]["predictions_dir"]
    )
    data = np.load(predictions_dir / f"{exp_id}_{split}.npz")
    return (
        data["y_true"],
        data["y_pred"],
        pd.DatetimeIndex(data["target_times"]),
    )
