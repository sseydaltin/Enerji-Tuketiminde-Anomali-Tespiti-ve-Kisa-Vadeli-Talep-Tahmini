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
    building_f1,
    busiest_window,
    median_building,
    LEAD_MODELS,
    SEEDS,
    SUBTYPE_LABELS,
    building_subtype_table,
    load_test_frame,
    load_thresholds,
    missing_files,
    slice_results,
)
from app.ui import (ACTUAL_COLOR, LINE_ACTUAL, MODEL_COLOR, SECOND_COLOR, SUBTYPE_COLORS, THIRD_COLOR, compact_table,
                    kpi_cards, number, page_intro, show_chart)

# Aynı model ve aynı alt tür her sayfada aynı renk (app/theme.py; renk körlüğüne uygun Okabe–Ito tabanlı).
MODEL_COLORS = {("L1", "behavior"): MODEL_COLOR, ("L1-FE", "full"): SECOND_COLOR, ("L4", "-"): THIRD_COLOR}


def make_figure(frame, building, thresholds, models=None, window=None, alarm_model=None):
    """Seçili bina: üstte tüketim + gerçek etiketler (alt türlere göre renkli), altta seçili modellerin skorları.

    models: skor paneli çizilecek modeller (None → hepsi; [] → yalnız üst grafik). window: (başlangıç, bitiş) görünüm
    aralığı (None → tüm dönem). alarm_model: bu modelin eşiği aştığı saatler üst grafikte işaretlenir.
    Lejant en çok 4 öğe: üç alt tür + alarm saatleri (tüketim çizgisi ve skorlar başlıklarla anlatılır).
    """
    models = list(LEAD_MODELS) if models is None else list(models)
    part = frame[frame["building_id"] == building]
    n = 1 + len(models)
    # Üst panelin alt başlığı yok (figür başlığı anlatır): lejant onunla çakışmasın.
    titles = [""] + [f"Skor: {LEAD_MODELS[key]}" for key in models]
    figure = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.08 if n > 1 else 0.0,
                           subplot_titles=titles, row_heights=[0.55] + [0.45 / len(models)] * len(models) if models else None)
    figure.add_trace(go.Scatter(x=part["timestamp"], y=part["meter_reading"], mode="lines", name="Tüketim",
                                line=LINE_ACTUAL, showlegend=False), row=1, col=1)
    for subtype, label in SUBTYPE_LABELS.items():
        hours = part[part["subtype"] == subtype]
        figure.add_trace(go.Scatter(x=hours["timestamp"], y=hours["meter_reading"], mode="markers",
                                    name=label,                       # kısa lejant: dar ekranda tek satıra sığsın
                                    marker={"color": SUBTYPE_COLORS[subtype], "size": 7}), row=1, col=1)
    if alarm_model is not None:
        flagged = part[part[f"{alarm_model[0]}|{alarm_model[1]}"].fillna(-np.inf) >= thresholds[alarm_model]]
        figure.add_trace(go.Scatter(x=flagged["timestamp"], y=flagged["meter_reading"], mode="markers",
                                    name="Final model alarmı",
                                    marker={"color": "rgba(0,0,0,0)", "size": 12,
                                            "line": {"color": MODEL_COLOR, "width": 1.6}}), row=1, col=1)
    for row, key in enumerate(models, start=2):
        values = part[f"{key[0]}|{key[1]}"]
        figure.add_trace(go.Scatter(x=part["timestamp"], y=values, mode="lines", name=LEAD_MODELS[key],
                                    showlegend=False, line={"color": MODEL_COLORS[key], "width": 1.8}), row=row, col=1)
        figure.add_hline(y=thresholds[key], line={"color": ACTUAL_COLOR, "dash": "dash", "width": 1}, row=row, col=1)
        finite = values[np.isfinite(values)]
        if len(finite):                                        # uç değerler grafiği ezmesin: üst sınır %99,5'lik
            top = max(float(np.quantile(finite, 0.995)), thresholds[key]) * 1.15
            figure.update_yaxes(range=[min(0.0, float(finite.min())), top], row=row, col=1)
    if window is not None:
        figure.update_xaxes(range=list(window))
    figure.update_layout(height=420 + 170 * (n - 1), margin={"t": 110},
                         title=f"Bina {building}, test: tüketim ve gerçek etiketler (renkli noktalar)"
                         + (" (kesikli çizgi: val'de seçilmiş eşik)" if n > 1 else ""))
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
    """Final modelin üç test KPI'ı (3 seed ort. ± std; kayıtlı test_results.csv'den, slice_results)."""
    final = table[table["model"] == LEAD_MODELS[FINAL_LEAD_MODEL]].set_index("dilim")

    def pm(dilim, metric):
        row = final.loc[dilim]
        return f"{number(row[metric + '_mean'])} ± {number(row[metric + '_std'])}"
    base = number(final.loc["Davranışsal (other)", "taban"], 4)
    return [("Davranışsal PR-AUC", pm("Davranışsal (other)", "pr_auc"),
             f"Zor alt türde başarı; rastgele skor ≈ {base} alır."),
            ("Genel PR-AUC", pm("Genel", "pr_auc"), "Bütün anomali saatlerinde başarı."),
            ("Genel ROC-AUC", pm("Genel", "roc_auc"), "Literatürle kıyas için kullanılan ölçüt.")]


SUBTYPE_TABLE_COLUMNS = {f"Yakalanan: {name}": (short, f"Eşiği aşan saat sayısı — {name}")
                         for (key, name), short in zip(LEAD_MODELS.items(), ("Final", "L1-FE full", "L4"))}
SUBTYPE_TABLE_COLUMNS["Anomali saati"] = ("Saat", "Bu alt türdeki gerçek anomali saati (son satır: normal saat).")


def _pick_building(seed, f1, which):
    """Düğme geri çağrısı: kolay (en yüksek F1) ya da zor (en düşük F1) örnek binayı seçer."""
    st.session_state[f"lead-building-{seed}"] = int(f1.idxmax() if which == "easy" else f1.idxmin())
    st.session_state[f"lead-reason-{seed}"] = which


def render(dataset=None):
    """Seed, bina ve örnek düğmeleri → 3 KPI → tek grafik (isteğe bağlı skor panelleri) → bina sayımı → (açılır) tablo."""
    page_intro("Gerçek Etiketli Veri (LEAD)",
               "200 ticari binanın elle etiketlenmiş anomalileri (2016, saatlik). Bir test binasında tüketim, alt "
               "türlerine göre renkli gerçek etiketler ve final modelin alarm verdiği saatler gösterilir; skorlar tek "
               "seferlik final testin kayıtlı çıktısıdır.")
    seed_col, building_col, buttons_col = st.columns([2, 3, 2], vertical_alignment="bottom")
    seed = seed_col.selectbox("Bölme (seed)", SEEDS, format_func=lambda s: f"Bina bölmesi, seed {s}")
    missing = missing_files(seed)
    if missing:
        st.warning("Bu sayfa için gereken yerel dosyalar yok (veri ve skorlar git'e girmez; README §4.14): "
                   + ", ".join(missing))
        return
    frame = load_test_frame(seed)
    thresholds = load_thresholds(seed)
    f1 = building_f1(seed)
    counts = frame.groupby("building_id")["anomaly"].sum().astype(int)
    key = f"lead-building-{seed}"
    if st.session_state.get(key) not in counts.index:
        st.session_state[key] = median_building(f1)
        st.session_state[f"lead-reason-{seed}"] = "median"
    with buttons_col.container(horizontal=True):            # iki düğme her genişlikte yan yana
        st.button("Kolay örnek", width="stretch", on_click=_pick_building, args=(seed, f1, "easy"))
        st.button("Zor örnek", width="stretch", on_click=_pick_building, args=(seed, f1, "hard"))
    building = building_col.selectbox("Test binası", counts.index.tolist(), key=key,
                                      format_func=lambda b: f"Bina {b} — {counts[b]} anomali saati",
                                      on_change=lambda: st.session_state.update({f"lead-reason-{seed}": "user"}))
    reasons = {"median": "final modelin bina F1'i 40 test binasının medyanına en yakın bina (ne en iyi ne en kötü örnek)",
               "easy": "final modelin bina F1'i en yüksek bina", "hard": "final modelin bina F1'i en düşük bina",
               "user": "kullanıcının seçtiği bina"}
    f1_text = f"; bu binada final modelin F1'i {number(f1[building])}" if building in f1.index else ""
    st.caption(f"Neden bu bina: {reasons[st.session_state.get(f'lead-reason-{seed}', 'user')]}{f1_text} "
               f"(medyan {number(f1.median())}). Yalnız görüntüleme seçimidir.")
    table = slice_results()
    kpi_cards(final_kpis(table))

    full_col, score_col, compare_col = st.columns([1, 1, 2], vertical_alignment="bottom")
    full_period = full_col.toggle("Tüm dönem", value=False, key="lead-full")
    show_scores = score_col.toggle("Model skorlarını göster", value=False, key="lead-scores")
    others = [k for k in LEAD_MODELS if k != FINAL_LEAD_MODEL]
    comparison = compare_col.selectbox("Karşılaştırma modeli", others, format_func=LEAD_MODELS.get,
                                       disabled=not show_scores, key="lead-compare")
    window = None if full_period else busiest_window(frame, building)
    models = [FINAL_LEAD_MODEL, comparison] if show_scores else []
    if window is not None:
        st.caption(f"Görünüm: bu binada en çok anomali içeren iki hafta ({window[0]:%d.%m} – "
                   f"{window[1] - pd.Timedelta(days=1):%d.%m.%Y}). Bütün yıl için \"Tüm dönem\"i açın.")
    show_chart(make_figure(frame, building, thresholds, models=models, window=window, alarm_model=FINAL_LEAD_MODEL),
               alt=f"Bina {building} test dönemi tüketimi, gerçek etiketler ve final modelin alarm saatleri")
    st.subheader("Seçili binada alt tür başına yakalanan saat")
    compact_table(building_subtype_table(frame, building, thresholds), SUBTYPE_TABLE_COLUMNS)
    with st.expander("Alt tür bazında final test sonuçları: üç model × altı dilim"):
        st.caption("Kaynak: results/lead/tables/test_results.csv. Taban = dilimin pozitif oranı. Val ile kıyas ve "
                   "test sonrası gözlem: Sonuçlar sayfasının LEAD sekmesi ve docs/lead_sonuclar.md.")
        full = results_table(table)
        names = {"Taban (PR-AUC)": ("Taban", "Dilimin pozitif oranı = rastgele skorun beklenen PR-AUC'si."),
                 "Olay recall": ("Olay rec.", "Olayın en az bir saati eşiği aşarsa yakalanmış sayılır.")}
        # İki dar tablo: dar ekranda yatay kaydırma olmasın.
        compact_table(full[["Model", "Dilim", "PR-AUC", "Taban (PR-AUC)"]], names)
        compact_table(full[["Model", "Dilim", "ROC-AUC", "F1", "Olay recall"]], names)
