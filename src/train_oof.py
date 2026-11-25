"""GEÇİCİ YÖNLENDİRME (2026-11-25): eski komut `python -m src.train_oof`.

Yeni komut: `python -m src.pipelines.train_oof`. Bu dosya yalnız eski komutların (README, belgeler,
deney günlüğü) çalışmaya devam etmesi içindir. Kütüphane adları burada YOKTUR:
import için yeni modül kullanılır. Eski yol → yeni yol tablosu: docs/deney_gunlugu.md.
"""

import runpy
import sys

TARGET = "src.pipelines.train_oof"

if __name__ == "__main__":
    print(f"Uyarı: bu komut taşındı; yeni komut: python -m {TARGET}", file=sys.stderr)
    runpy.run_module(TARGET, run_name="__main__", alter_sys=True)
