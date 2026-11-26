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
from app.ui import ACTUAL_COLOR, ANOMALY_COLOR, BASELINE_COLOR, LINE_ACTUAL, MODEL_COLOR, SECOND_COLOR, number, show_chart

SUBTYPE_COLORS = {"gap_edge": "#E69F00", "frozen": "#7B61FF", "other": ANOMALY_COLOR}
MODEL_COLORS = {("L1", "behavior"): MODEL_COLOR, ("L1-FE", "full"): SECOND_COLOR, ("L4", "-"): BASELINE_COLOR}


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
    figure.update_layout(height=230 * n, title=f"Bina {building} — TEST dönemi (kesikli çizgi: val'de seçilmiş eşik)")
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


def render(dataset="household"):
    """Sayfa: seed ve test binası seçimi → grafik → seçili binanın alt tür sayımı → genel alt tür tablosu."""
    st.title("Gerçek etiketli veri (LEAD)")
    st.caption("LEAD1.0: 200 ticari bina, 2016, saatlik; etiketler elle konmuş. Ana veri setlerindeki (hane, şebeke) "
               "yaklaşımın gerçek etiketlerle ek doğrulaması. Kenar çubuğundaki veri seti seçimi bu sayfayı etkilemez. "
               "Skorlar tek seferlik final testin kayıtlı çıktısıdır; eşikler val'de seçildi.")
    seed = st.selectbox("Bölme (seed)", SEEDS, format_func=lambda s: f"Bina bölmesi, seed {s} (40 test binası)")
    missing = missing_files(seed)
    if missing:
        st.warning("Bu sayfa için gereken yerel dosyalar yok (veri ve skorlar git'e girmez): " + ", ".join(missing))
        return
    frame = load_test_frame(seed)
    thresholds = load_thresholds(seed)
    counts = frame.groupby("building_id")["anomaly"].sum().astype(int)
    building = st.selectbox("Test binası", counts.index.tolist(),
                            format_func=lambda b: f"Bina {b} ({counts[b]} anomali saati)")
    final_label = LEAD_MODELS[FINAL_LEAD_MODEL]
    st.markdown(f"**Final model:** {final_label}. Gerçek etiket renkleri: turuncu = boşluk kenarı, mor = donmuş "
                "okuma, kırmızı = davranışsal (other).")
    show_chart(make_figure(frame, building, thresholds),
               alt=f"Bina {building} test dönemi tüketimi, gerçek etiketler ve üç modelin skoru")
    st.subheader("Seçili binada alt tür başına yakalanan saat")
    st.caption("Eşiği aşan (alarm verilen) saat sayısı; son satır normal saatlerdeki yanlış alarmlar.")
    st.dataframe(building_subtype_table(frame, building, thresholds), hide_index=True, width="stretch")
    st.subheader("Alt tür bazında final test sonuçları (40 test binası × 3 seed)")
    st.caption("Kaynak: results/lead/tables/test_results.csv. Birincil metrik: davranışsal (other) PR-AUC. "
               "Taban = dilimin pozitif oranı. Ayrıntı ve val ile kıyas: docs/lead_sonuclar.md.")
    st.dataframe(results_table(slice_results()), hide_index=True, width="stretch")
