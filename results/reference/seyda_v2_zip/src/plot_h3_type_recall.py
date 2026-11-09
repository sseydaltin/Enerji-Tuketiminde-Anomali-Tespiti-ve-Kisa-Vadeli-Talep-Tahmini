"""H3 hedefinin (olay F1 >= 0,70) neden karşılanmadığını gösteren figür.

Yeni deney ÇALIŞTIRMAZ; yalnız daha önce üretilmiş iki tabloyu okur:
  - results/tables/anomaly_eval_audit_summary.csv  (Aşama 2, A3-s2s-norm satırı)
      -> anomali türü başına olay recall'ü, kış-dev ve yaz-dev (orta senaryo, 10 seed ort.)
  - results/tables/anomaly_eval_audit_oracle.csv
      -> S0 referansında tek bir tür "kusursuz yakalansaydı" block F1 ne olurdu?

Çıktı: results/figures/H3_A3-s2s-norm_dev_type_recall.png (dpi 200)

Çalıştırma:  python -m src.plot_h3_type_recall
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # ekran olmadan PNG üretmek için
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
OUT = ROOT / "results" / "figures" / "H3_A3-s2s-norm_dev_type_recall.png"

TYPES = ["spike", "drop", "level_shift", "pattern"]
TYPE_LABELS = ["Spike\n(ani sıçrama)", "Drop\n(kesinti)", "Seviye\nkayması", "Gece\ndeseni"]
H3_TARGET = 0.70


def read_type_recall(method="A3-s2s-norm", stage="Aşama 2"):
    """Denetim tablosundan bir yöntemin tür başına olay recall'ünü okur.

    Dönüş: {"winter": [4 değer], "summer": [4 değer]}  (sıra TYPES ile aynı)
    """
    with open(TABLES / "anomaly_eval_audit_summary.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["method"] == method and row["stage"] == stage:
                return {
                    "winter": [float(row[f"winter_dev_event_recall_{t}"]) for t in TYPES],
                    "summer": [float(row[f"summer_dev_event_recall_{t}"]) for t in TYPES],
                }
    raise ValueError(f"{method} / {stage} satırı bulunamadı")


def read_oracle():
    """Oracle tablosu: 'none' = S0'ın kendisi, diğerleri = o tür kusursuz yakalansaydı.

    Dönüş: {"winter": {tip: block_f1}, "summer": {tip: block_f1}}
    """
    out = {"winter": {}, "summer": {}}
    with open(TABLES / "anomaly_eval_audit_oracle.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            season = "winter" if row["set"] == "winter_dev" else "summer"
            out[season][row["added_type"]] = float(row["block_f1"])
    return out


def main():
    recall = read_type_recall()
    oracle = read_oracle()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = range(len(TYPES))
    w = 0.38

    # Sol panel: hangi tür yakalanıyor, hangisi hiç yakalanmıyor?
    ax1.bar([i - w / 2 for i in x], recall["winter"], w, label="Kış-dev", color="#3b6ea8")
    ax1.bar([i + w / 2 for i in x], recall["summer"], w, label="Yaz-dev", color="#e08a2c")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(TYPE_LABELS)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Olay recall'ü (0–1)")
    ax1.set_title("A3-s2s-norm: anomali türüne göre yakalama\n(geliştirme kümeleri, orta senaryo, 10 seed ort.)")
    ax1.legend()

    # Sağ panel: tek bir tür kusursuz yakalansa bile F1 0,70'e ulaşıyor mu?
    keys = ["none"] + TYPES
    labels = ["Mevcut\n(S0)"] + TYPE_LABELS
    xs = range(len(keys))
    ax2.bar([i - w / 2 for i in xs], [oracle["winter"][k] for k in keys], w, label="Kış-dev", color="#3b6ea8")
    ax2.bar([i + w / 2 for i in xs], [oracle["summer"][k] for k in keys], w, label="Yaz-dev", color="#e08a2c")
    ax2.axhline(H3_TARGET, color="crimson", linestyle="--", label="H3 hedefi (0,70)")
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("Olay F1 (block_f1)")
    ax2.set_title("Oracle: bu tür kusursuz yakalansaydı F1 ne olurdu?\n(S0 referansı, geliştirme kümeleri, orta senaryo)")
    ax2.legend(loc="upper left")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print(f"Kaydedildi: {OUT}")


if __name__ == "__main__":
    main()
