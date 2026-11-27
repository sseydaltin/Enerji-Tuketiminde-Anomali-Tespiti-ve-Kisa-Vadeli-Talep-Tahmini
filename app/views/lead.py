"""Gerçek etiketli veri (LEAD) sayfası: bir test binasında tüketim, gerçek etiketler (alt türlere göre renkli),
kayıtlı final test skorları ve val eşikleri; altında alt tür bazında sonuç tablosu.

Hesap yok: skorlar ve eşikler tek seferlik final testin kayıtlı çıktısıdır (app/lead_services.py).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.lead_services import (
    FINAL_LEAD_MODEL,
    LEAD_MODELS,
    SEEDS,
    SUBTYPE_LABELS,
    building_subtype_table,
    load_test_frame,
    load_thresholds,
    missing_files,
    slice_results,
)
from app.ui import (ACTUAL_COLOR, LINE_ACTUAL, MODEL_COLOR, SECOND_COLOR, SUBTYPE_COLORS, THIRD_COLOR, kpi_row, number,
                    page_intro, show_chart)

# Aynı model ve aynı alt tür her sayfada aynı renk (app/theme.py; renk körlüğüne uygun Okabe–Ito tabanlı).
MODEL_COLORS = {("L1", "behavior"): MODEL_COLOR, ("L1-FE", "full"): SECOND_COLOR, ("L4", "-"): THIRD_COLOR}


def make_figure(frame, building, thresholds):
    """Seçili bina için 1 + modeller kadar satırlı figür.

    Üst satır: tüketim (kWh) ve gerçek anomali saatleri alt türüne göre renkli noktalar. Alt satırlar: her modelin
    kayıtlı test skoru ve val eşiği (kesikli çizgi). Skor ölçekleri farklı olduğu için her model ayrı satırda.
    """
    part = frame[frame["building_id"] == building]
    n = 1 + len(LEAD_MODELS)
    titles = ["Tüketim (kWh) ve gerçek etiketler"] + [f"Skor: {name}" for name in LEAD_MODELS.values()]
    figure = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=titles,
                           row_heights=[0.4] + [0.6 / len(LEAD_MODELS)] * len(LEAD_MODELS))
    figure.add_trace(go.Scatter(x=part["timestamp"], y=part["meter_reading"], mode="lines", name="Tüketim",
                                line=LINE_ACTUAL), row=1, col=1)
    for subtype, label in SUBTYPE_LABELS.items():
        hours = part[part["subtype"] == subtype]
        figure.add_trace(go.Scatter(x=hours["timestamp"], y=hours["meter_reading"], mode="markers",
                                    name=f"Gerçek etiket: {label}",
                                    marker={"color": SUBTYPE_COLORS[subtype], "size": 6}), row=1, col=1)
    for row, (key, name) in enumerate(LEAD_MODELS.items(), start=2):
        values = part[f"{key[0]}|{key[1]}"]
        figure.add_trace(go.Scatter(x=part["timestamp"], y=values, mode="lines", name=name,
                                    line={"color": MODEL_COLORS[key], "width": 1.4}), row=row, col=1)
        figure.add_hline(y=thresholds[key], line={"color": ACTUAL_COLOR, "dash": "dash", "width": 1}, row=row, col=1)
        finite = values[np.isfinite(values)]
        if len(finite):                                        # uç değerler grafiği ezmesin: üst sınır %99,5'lik
            top = max(float(np.quantile(finite, 0.995)), thresholds[key]) * 1.15
            figure.update_yaxes(range=[min(0.0, float(finite.min())), top], row=row, col=1)
    figure.update_layout(height=200 + 150 * n, margin={"t": 130},     # iki satırlık lejant başlığın altına sığsın
                         title=f"Bina {building} — test dönemi (kesikli çizgi: val'de seçilmiş eşik)")
    figure.update_yaxes(title_text="kWh", row=1, col=1)
    return figure


def results_table(table):
    """Alt tür bazında sonuç tablosu (3 seed ort. ± std), Türkçe ondalık virgülü."""
    def pm(row, metric):
        return f"{number(row[metric + '_mean'])} ± {number(row[metric + '_std'])}"
    rows = [{"Model": r["model"], "Dilim": r["dilim"], "PR-AUC": pm(r, "pr_auc"), "Taban (PR-AUC)": number(r["taban"], 4),
             "ROC-AUC": pm(r, "roc_auc"), "F1": pm(r, "f1"), "Olay recall": pm(r, "event_recall")}
            for _, r in table.iterrows()]
    return pd.DataFrame(rows)


def final_kpis(table):
    """Final modelin test KPI'ları (3 seed ort. ± std): kayıtlı test_results.csv'den (slice_results)."""
    final = table[table["model"] == LEAD_MODELS[FINAL_LEAD_MODEL]].set_index("dilim")

    def pm(dilim, metric):
        row = final.loc[dilim]
        return f"{number(row[metric + '_mean'])} ± {number(row[metric + '_std'])}"
    other_base = number(final.loc["Davranışsal (other)", "taban"], 4)
    return [("Davranışsal (other) PR-AUC", pm("Davranışsal (other)", "pr_auc"),
             f"Birincil metrik. Rastgele skorun beklenen değeri (taban) ≈ {other_base}."),
            ("Genel PR-AUC", pm("Genel", "pr_auc"), "Bütün anomali saatleri."),
            ("Genel ROC-AUC", pm("Genel", "roc_auc"), "Literatür kıyası için (koşul farkları docs/lead_sonuclar.md)."),
            ("Davranışsal olay recall", pm("Davranışsal (other)", "event_recall"),
             "Olayın en az bir saati eşiği aşarsa olay yakalanmış sayılır.")]


def render(dataset=None):
    """Sayfa: üstte seed ve bina şeridi → final model KPI'ları → grafik → bina sayımı → (açılır) alt tür tablosu."""
    page_intro("Gerçek Etiketli Veri (LEAD)",
               "200 ticari binanın elle etiketlenmiş anomalileri (2016, saatlik). Bir test binası seçin: tüketim serisi, "
               "gerçek etiketler (alt türlerine göre renkli) ve üç modelin skoru val'de seçilmiş eşikleriyle gösterilir. "
               "Skorlar tek seferlik final testin kayıtlı çıktısıdır.")
    seed_col, building_col = st.columns([1, 2])
    seed = seed_col.selectbox("Bölme (seed)", SEEDS, format_func=lambda s: f"Bina bölmesi, seed {s}")
    missing = missing_files(seed)
    if missing:
        st.warning("Bu sayfa için gereken yerel dosyalar yok (veri ve skorlar git'e girmez; README §4.14): "
                   + ", ".join(missing))
        return
    frame = load_test_frame(seed)
    thresholds = load_thresholds(seed)
    counts = frame.groupby("building_id")["anomaly"].sum().astype(int)
    building = building_col.selectbox("Test binası (40 bina)", counts.index.tolist(),
                                      format_func=lambda b: f"Bina {b} — {counts[b]} anomali saati")
    table = slice_results()
    st.markdown(f"**Final model: {LEAD_MODELS[FINAL_LEAD_MODEL]}** — test sonuçları (40 test binası × 3 seed):")
    kpi_row(final_kpis(table))
    show_chart(make_figure(frame, building, thresholds),
               alt=f"Bina {building} test dönemi tüketimi, gerçek etiketler ve üç modelin skoru")
    st.subheader("Seçili binada alt tür başına yakalanan saat")
    st.caption("Eşiği aşan (alarm verilen) saat sayısı; son satır normal saatlerdeki yanlış alarmlardır.")
    st.dataframe(building_subtype_table(frame, building, thresholds), hide_index=True, width="stretch")
    with st.expander("Alt tür bazında final test sonuçları: üç model × altı dilim"):
        st.caption("Kaynak: results/lead/tables/test_results.csv. Taban = dilimin pozitif oranı. Val ile kıyas ve "
                   "test sonrası gözlem: docs/lead_sonuclar.md.")
        st.dataframe(results_table(table), hide_index=True, width="stretch")
