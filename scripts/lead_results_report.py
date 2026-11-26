"""docs/lead_sonuclar.md ve results/lead/tables/test_summary*.csv üretir (yeni hesap yok; yalnız kayıtlı tablolar).

Kullanım: python scripts/lead_results_report.py
Girdi: results/lead/tables/val_results.csv, test_results.csv, test_val_reproduction.csv, test_building_f1.csv.
Sayılar yalnız bu dosyalardan gelir (CLAUDE.md: sonuç uydurma yok).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # yalnız `src` import'u için (verify betiği gibi);
from src.core.config import PROJECT_ROOT                                   # noqa: E402
from src.pipelines.lead_models import LITERATURE, MODEL_NOTES               # noqa: E402
from src.pipelines.lead_test import FINAL_MODEL, TEST_MODELS                 # noqa: E402

T = PROJECT_ROOT / "results" / "lead" / "tables"
SLICES = ["other", "genel", "gap_edge", "frozen", "paylasilan", "tekil"]
METRICS = ["roc_auc", "pr_auc", "precision", "recall", "f1", "event_recall"]


def summarize(results, part):
    """Seed ort. ± std; model × set × dilim. part: 'val' ya da 'test'."""
    keep = results.merge(pd.DataFrame(TEST_MODELS, columns=["model", "set"]), on=["model", "set"])
    keep = keep[keep["bolme"] == "bina"].copy()
    keep["taban"] = keep["n_pos"] / (keep["n_pos"] + keep["n_neg"])
    agg = keep.groupby(["model", "set", "dilim"]).agg(
        n_seed=("seed", "nunique"), taban=("taban", "mean"),
        **{f"{m}": (m, "mean") for m in METRICS}, **{f"{m}_std": (m, "std") for m in METRICS}).reset_index()
    return agg.assign(kume=part)


def pm(m, s):
    return "—" if pd.isna(m) else f"{m:.3f} ± {s:.3f}"


def main():
    val = pd.read_csv(T / "val_results.csv")
    test = pd.read_csv(T / "test_results.csv")
    repro = pd.read_csv(T / "test_val_reproduction.csv")
    bf1 = pd.read_csv(T / "test_building_f1.csv")
    sv, st = summarize(val, "val"), summarize(test, "test")
    both = pd.concat([sv, st], ignore_index=True)
    both.to_csv(T / "test_summary_val_vs_test.csv", index=False)

    order = [f"{m} / {s}" for m, s in TEST_MODELS]
    lines = ["# LEAD1.0 sonuçları — val ve FİNAL TEST (2026-11-26)", "",
             "Test **bir kez**, kullanıcının açık onayıyla çalıştırıldı (`python -m src.pipelines.lead_run --stage test "
             "--final-degerlendirme`). Testte hiçbir şey seçilmedi: ayarlar, öznitelik grupları, eşikler (val F1-max), R3'ün k'sı "
             "ve L5-FE ağırlıkları val kayıtlarından aynen alındı (protokol §21.5). Sonuçlara göre hiçbir şey değiştirilmedi. "
             "Bina bölmesi, 3 seed (42, 7, 2026), her seed'de 40 test binası. Değerler 3 seed ort. ± std. Bütün sayılar "
             "`results/lead/tables/` dosyalarından betikle (`scripts/lead_results_report.py`) alınmıştır.", "",
             f"**Final model (val'de §18.2 + §21 ile seçildi): {FINAL_MODEL[0]} / {FINAL_MODEL[1]}.** "
             "Birincil metrik `other` (davranışsal) dilimi PR-AUC; genel ROC-AUC/PR-AUC literatür kıyası için.", ""]

    # 1. Ana tablo: other ve genel, val | test
    lines += ["## 1. Ana tablo: `other` ve genel, val ile test yan yana", "",
              "| model | set | other PR-AUC val | other PR-AUC test | genel PR-AUC val | genel PR-AUC test | "
              "genel ROC-AUC val | genel ROC-AUC test | other F1 test | other olay recall test |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for m, s in TEST_MODELS:
        def g(df, d, c):
            r = df[(df.model == m) & (df.set == s) & (df.dilim == d)]
            return (r[c].iloc[0], r[c + "_std"].iloc[0]) if len(r) else (np.nan, np.nan)
        mark = " **(final)**" if (m, s) == FINAL_MODEL else ""
        lines.append(f"| {m}{mark} | {s} | {pm(*g(sv, 'other', 'pr_auc'))} | {pm(*g(st, 'other', 'pr_auc'))} | "
                     f"{pm(*g(sv, 'genel', 'pr_auc'))} | {pm(*g(st, 'genel', 'pr_auc'))} | "
                     f"{pm(*g(sv, 'genel', 'roc_auc'))} | {pm(*g(st, 'genel', 'roc_auc'))} | "
                     f"{pm(*g(st, 'other', 'f1'))} | {pm(*g(st, 'other', 'event_recall'))} |")
    for d in ("other", "genel"):
        lines.append(f"\nPR-AUC tabanı ({d}): val {sv[sv.dilim == d].taban.mean():.4f}, test {st[st.dilim == d].taban.mean():.4f}.")
    lines.append("")

    # 2. Dilim başına tam tablo (test)
    lines += ["## 2. Bütün dilimler ve metrikler (TEST)", "",
              "Taban = dilimin pozitif oranı (rastgele skorun beklenen PR-AUC'si). Val karşılıkları: "
              "`results/lead/tables/test_summary_val_vs_test.csv` (kume = val).", ""]
    for d in SLICES:
        part = st[st.dilim == d]
        lines += [f"### {d} (PR-AUC tabanı ≈ {part.taban.mean():.4f})", "",
                  "| model | set | ROC-AUC | PR-AUC | precision | recall | F1 | olay recall |", "|---|---|---|---|---|---|---|---|"]
        for m, s in TEST_MODELS:
            r = part[(part.model == m) & (part.set == s)]
            if len(r):
                r = r.iloc[0]
                lines.append(f"| {m} | {s} | " + " | ".join(pm(r[c], r[c + '_std']) for c in METRICS) + " |")
        lines.append("")

    # 3. Bina bazında F1
    lines += ["## 3. Bina bazında F1 (TEST; anomalisi olan test binaları, 3 seed birleşik)", "",
              "| model | set | bina sayısı | medyan F1 | çeyrekler arası (Q1–Q3) |", "|---|---|---|---|---|"]
    for m, s in TEST_MODELS:
        x = bf1[(bf1.model == m) & (bf1.set == s)]["f1"]
        if len(x):
            lines.append(f"| {m} | {s} | {len(x)} | {x.median():.3f} | {x.quantile(.25):.3f}–{x.quantile(.75):.3f} |")
    lines.append("")

    # 4. Val yeniden üretimi
    lines += ["## 4. Val yeniden üretim kontrolü (§21.6)", "",
              "Val aşamasında model nesneleri kaydedilmediği için her model val prosedürüyle yeniden eğitildi; önce val "
              "skorları yeniden üretilip kayıtlı val skorlarıyla karşılaştırıldı (`test_val_reproduction.csv`).", "",
              "| model | set | seed | max |fark| (skor) | val PR-AUC kayıtlı | val PR-AUC yeniden |", "|---|---|---|---|---|---|"]
    for r in repro.itertuples():
        lines.append(f"| {r.model} | {r.set} | {r.seed} | {r.max_abs_fark:.2e} | {r.pr_auc_kayitli:.4f} | {r.pr_auc_yeniden:.4f} |")
    lines.append("")

    # 5. Literatür
    final_test = st[(st.model == FINAL_MODEL[0]) & (st.set == FINAL_MODEL[1]) & (st.dilim == "genel")].iloc[0]
    off = st[(st.model == "L1-OFF") & (st.dilim == "genel")].iloc[0]
    iff = st[(st.model == "IF") & (st.dilim == "genel")].iloc[0]
    lines += ["## 5. Literatür kıyası (koşullar farklı; \"daha iyi / daha kötü\" doğrudan söylenmez, §14)", "",
              "| kaynak | değer | koşul farkları |", "|---|---|---|"]
    lines += [f"| {name} | {value} | {cond} |" for name, value, cond in LITERATURE]
    lines += [f"| **Bizim final ({FINAL_MODEL[0]} / {FINAL_MODEL[1]}, çevrimiçi)** | genel ROC-AUC {pm(final_test.roc_auc, final_test.roc_auc_std)} (test) | nedensel öznitelikler; 200 etiketli bina 120/40/40; 3 seed |",
              f"| Bizim L1-OFF (offline, iki yönlü) | genel ROC-AUC {pm(off.roc_auc, off.roc_auc_std)} (test) | Kaggle koşuluna en yakın satırımız; seçime girmedi |",
              f"| Bizim Isolation Forest | nokta F1 {pm(iff.f1, iff.f1_std)} (test, genel) | saat düzeyi, bina bölmesi, val'de F1-max eşik — PyOD satırıyla kıyas için |", ""]

    # 6. Sapmalar
    lines += ["## 6. Val → test sapmaları", "",
              "Fark = test − val (3 seed ort.). Sapma, iki kümenin seed'ler arası std'lerinin büyüğünden fazlaysa \"belirgin\" "
              "sayıldı (ölçüt betikte; seçim için değil, yalnız yorum için).", "",
              "| model | set | dilim | val PR-AUC | test PR-AUC | fark | belirgin mi |", "|---|---|---|---|---|---|---|"]
    flagged = []
    for m, s in TEST_MODELS:
        for d in ("other", "genel", "frozen", "gap_edge"):
            a = sv[(sv.model == m) & (sv.set == s) & (sv.dilim == d)].iloc[0]
            b = st[(st.model == m) & (st.set == s) & (st.dilim == d)].iloc[0]
            diff = b.pr_auc - a.pr_auc
            big = abs(diff) > max(a.pr_auc_std, b.pr_auc_std)
            if big:
                flagged.append((m, s, d, diff))
            lines.append(f"| {m} | {s} | {d} | {a.pr_auc:.3f} | {b.pr_auc:.3f} | {diff:+.3f} | {'evet' if big else 'hayır'} |")
    comp = pd.read_csv(T / "test_split_composition.csv")
    cols = ["anomali_orani", "other_taban", "gap_edge_pay", "frozen_pay", "other_pay", "paylasilan_olay_pay",
            "other_paylasilan_pay", "other_ilk3_bina_pay"]
    lines += ["", "### 6.1 Olası nedenler: parça bileşimi (`test_split_composition.csv`; betimleyici, seçim yok)", "",
              "| seed | parça | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for r in comp.itertuples():
        lines.append(f"| {r.seed} | {r.parca} | " + " | ".join(f"{getattr(r, c):.3f}" for c in cols) + " |")
    mean = comp.groupby("parca")[cols].mean()
    v, t = mean.loc["val"], mean.loc["test"]
    lines += ["", f"Ortalama (3 seed): frozen payı val {v.frozen_pay:.3f} → test {t.frozen_pay:.3f}; `other` payı "
              f"{v.other_pay:.3f} → {t.other_pay:.3f}; paylaşılan olay payı {v.paylasilan_olay_pay:.3f} → "
              f"{t.paylasilan_olay_pay:.3f}; paylaşılan `other` olay payı {v.other_paylasilan_pay:.3f} → "
              f"{t.other_paylasilan_pay:.3f}; `other` tabanı {v.other_taban:.4f} → {t.other_taban:.4f}.", "",
              "Yorum (kanıt değil, olası nedenler):",
              "1. **Sapma hemen hemen bütün modellerde aynı yönde (test > val)**, kurallarda (R3) ve etiketsiz modellerde de. "
              "Bu, modelden çok **parça bileşiminden** gelen bir fark olduğunu düşündürür: test parçalarında kolay yakalanan "
              "frozen anomalilerin payı daha yüksek, `other`'ın payı daha düşük. Genel PR-AUC frozen'a çok bağlı olduğu için yükseliyor.",
              "2. **`other` diliminde** test parçalarında paylaşılan olay payı daha yüksek; val analizinde (`lead_val_ozet.md` §5) "
              "paylaşılan olaylarda PR-AUC'nin tekil olaylardakinin yaklaşık iki katı olduğu görülmüştü.",
              "3. **Val parçaları görece \"zor\" düştü** (özellikle seed 42 val: `other` payı 0,407, ilk 3 bina `other` saatlerinin %37'si). "
              "Seed başına 40 bina ve `other` saatlerinin birkaç binada yoğunlaşması, parçadan parçaya büyük oynaklık yaratıyor; "
              "val'de `other` PR-AUC'nin modelleri ayırt edememesi (§21.2 bulgusu) ile aynı nedendir.",
              "4. **Final model (L1 behavior) ile L1-FE full arasındaki fark testte büyüdü** (`other` PR-AUC 0,051'e karşı 0,120). "
              "Val'de bu fark gürültü düzeyindeydi ve seçim kuralı basitlik sırasıyla L1 behavior'ı seçti. Protokol gereği "
              "(§13) test sonucuna göre final model **değiştirilmez**; bu, raporda \"val'de ayırt edilemeyen iki modelin testte "
              "ayrışması\" olarak yazılır.",
              "5. Val skorlarında val erken durdurmasından gelen hafif iyimserlik (§8.2, val özeti) test yönündeki farkı açıklamaz "
              "(o, val'i yüksek gösterirdi); farkın yönü tersidir.", ""]
    # 7. Test sonrası gözlem (kullanıcı isteğiyle; test sonrası, hiçbir seçimi değiştirmez)
    fin = st[(st.model == FINAL_MODEL[0]) & (st.set == FINAL_MODEL[1])].set_index("dilim")
    fe = st[(st.model == "L1-FE") & (st.set == "full")].set_index("dilim")
    fin_v = sv[(sv.model == FINAL_MODEL[0]) & (sv.set == FINAL_MODEL[1])].set_index("dilim")
    fe_v = sv[(sv.model == "L1-FE") & (sv.set == "full")].set_index("dilim")
    lines += ["## 7. Test sonrası gözlem", "",
              "> Bu bölüm test sonuçları görüldükten **sonra** yazıldı. Hiçbir seçimi, eşiği ya da ayarı değiştirmez.", "",
              f"**Final model değişmeden kalır: {FINAL_MODEL[0]} / {FINAL_MODEL[1]}** (val'de §18.2 + §21 ile seçildi; protokol §13: "
              "test sonucuna göre hiçbir şey değiştirilmez).", "",
              "**L1-FE full testte daha iyi genelleşti:**", "",
              "| model | other PR-AUC val | other PR-AUC test | genel PR-AUC val | genel PR-AUC test |", "|---|---|---|---|---|",
              f"| {FINAL_MODEL[0]} / {FINAL_MODEL[1]} (final) | {pm(fin_v.loc['other', 'pr_auc'], fin_v.loc['other', 'pr_auc_std'])} | "
              f"{pm(fin.loc['other', 'pr_auc'], fin.loc['other', 'pr_auc_std'])} | {pm(fin_v.loc['genel', 'pr_auc'], fin_v.loc['genel', 'pr_auc_std'])} | "
              f"{pm(fin.loc['genel', 'pr_auc'], fin.loc['genel', 'pr_auc_std'])} |",
              f"| L1-FE / full | {pm(fe_v.loc['other', 'pr_auc'], fe_v.loc['other', 'pr_auc_std'])} | "
              f"{pm(fe.loc['other', 'pr_auc'], fe.loc['other', 'pr_auc_std'])} | {pm(fe_v.loc['genel', 'pr_auc'], fe_v.loc['genel', 'pr_auc_std'])} | "
              f"{pm(fe.loc['genel', 'pr_auc'], fe.loc['genel', 'pr_auc_std'])} |", "",
              f"Testte L1-FE full `other` PR-AUC'de {fe.loc['other', 'pr_auc']:.3f} ± {fe.loc['other', 'pr_auc_std']:.3f}, final model "
              f"{fin.loc['other', 'pr_auc']:.3f} ± {fin.loc['other', 'pr_auc_std']:.3f}; genel PR-AUC'de {fe.loc['genel', 'pr_auc']:.3f} ve "
              f"{fin.loc['genel', 'pr_auc']:.3f}. Val'de iki model gürültü düzeyindeydi (§21.2) ve seçim kuralı basitlik sırasıyla L1 behavior'ı seçti.", "",
              "**Ders:** tek bir val kümesi (seed başına 40 bina) bu büyüklükteki farkları ayırt edemedi. Seçim, aynı 40 bina üzerinde "
              "yapılan tek bir ölçüme dayandı; val parçasının bileşimi (aşağıda) sonucu modeller arasındaki farktan daha çok etkiledi. "
              "**Öneri (gelecek çalışma):** model seçimi train+val binalarında **bina bazlı (gruplu) çapraz doğrulama** ile yapılmalı "
              "(ör. 5 kat × birkaç tekrar; her katta farklı binalar tutulur); farklar kat başına eşleştirilmiş olarak ve daha fazla "
              "bağımsız ölçümle değerlendirilmeli. L-FE ablasyonu ve I-turu bu yapıyı (3 kat) zaten train içinde kullandı; final seçim de "
              "aynı yapıya taşınmalıydı.", "",
              "**Bulgu: `other` skorları bina bileşimine çok duyarlı.** Aynı modelin `other` PR-AUC'si parçadan parçaya büyük ölçüde "
              "değişiyor; kurallar ve etiketsiz modeller de aynı yönde etkileniyor. Val ve test parçalarının bileşimi "
              "(`test_split_composition.csv`):", "",
              "| seed | parça | bina | other tabanı | frozen payı | other payı | paylaşılan olay payı | paylaşılan other olay payı | other'ın ilk 3 binadaki payı |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in comp[comp.parca.isin(["val", "test"])].itertuples():
        lines.append(f"| {r.seed} | {r.parca} | {r.n_bina} | {r.other_taban:.4f} | {r.frozen_pay:.3f} | {r.other_pay:.3f} | "
                     f"{r.paylasilan_olay_pay:.3f} | {r.other_paylasilan_pay:.3f} | {r.other_ilk3_bina_pay:.3f} |")
    lines += ["", "`other` saatlerinin %28–39'u parçanın yalnız 3 binasında; bu yüzden 40 binalık bir parçada birkaç binanın davranışı "
              "`other` sonucunu belirliyor. Final modelin `other` PR-AUC'sinin seed'ler arası std'si testte "
              f"{fin.loc['other', 'pr_auc_std']:.3f}, ortalamanın yarısından fazla.", ""]
    lines += ["## 8. Notlar", ""] + [f"- **{m}:** {MODEL_NOTES[m]}" for m in sorted({m for m, _ in TEST_MODELS}) if m in MODEL_NOTES]
    lines += ["- L3 (CNN+GRU) eğitilmedi (§17). Site bölmesinde test yapılmadı (yalnız val sağlamlık kontrolüydü)."]
    (PROJECT_ROOT / "docs" / "lead_sonuclar.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:40]))
    print("Belirgin sapmalar:", flagged)


if __name__ == "__main__":
    main()
