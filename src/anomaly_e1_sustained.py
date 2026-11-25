"""GEÇİCİ YÖNLENDİRME (2026-11-25): eski komut `python -m src.anomaly_e1_sustained`.

Yeni komut: `python -m experiments.archive.anomaly_e1_sustained`. Bu dosya yalnız eski komutların (README, belgeler,
deney günlüğü) çalışmaya devam etmesi içindir. Kütüphane adları burada YOKTUR:
import için yeni modül kullanılır. Eski yol → yeni yol tablosu: docs/deney_gunlugu.md.
"""

import runpy
import sys

TARGET = "experiments.archive.anomaly_e1_sustained"

if __name__ == "__main__":
    print(f"Uyarı: bu komut taşındı; yeni komut: python -m {TARGET}", file=sys.stderr)
    runpy.run_module(TARGET, run_name="__main__", alter_sys=True)
