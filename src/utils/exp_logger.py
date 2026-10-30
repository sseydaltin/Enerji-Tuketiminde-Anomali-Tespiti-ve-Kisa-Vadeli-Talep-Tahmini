"""Deney kayıt yardımcısı (stdlib `logging` ile çakışmasın diye bu ad)."""

from pathlib import Path

import pandas as pd


def resolve_path(root, configured_path):
    """Config'teki göreli yolu proje köküne göre mutlak yola çevirir."""
    path = Path(configured_path)
    return path if path.is_absolute() else Path(root) / path


def upsert_csv(path, rows, key_columns):
    """CSV'deki aynı anahtarlı satırları yeniler, diğer satırları korur.

    Neden: Aynı deney tekrar koşulursa satır çoğalmaz; farklı exp_id'ler kalır.
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
