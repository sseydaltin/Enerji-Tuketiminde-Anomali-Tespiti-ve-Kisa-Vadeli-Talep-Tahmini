"""Anomali Tespiti sayfası: hane test döneminde skor, eşik ve olay görünümü."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.anomaly_services import (
    METHODS,
    analyze_test_range,
    anomaly_demo_paths,
    checkpoint_signature,
    complete_test_days,
    load_anomaly_calibration,
    validate_injection_range,
)
from app.services import load_f13_models, load_household_data
from app.ui import (
    ACTUAL_COLOR,
    ANOMALY_COLOR,
    ANOMALY_LABELS,
    BASELINE_COLOR,
    CARD_HEIGHT,
    DATASET_LABELS,
    FINAL_METHOD,
    LINE_ACTUAL,
    LINE_MODEL,
    METHOD_LABELS,
    number,
    show_chart,
)
from src.anomaly_protocol import scenario_settings
from src.data.inject_anomalies import ANOMALY_TYPES
from src.utils.anomaly_metrics import runs

NO_INJECTION = "none"
INJECTION_LABELS = {NO_INJECTION: "Yok (gerçek ölçüm)", **ANOMALY_LABELS}
SEVERITY_LABELS = {"main": "Ana senaryo", "hard": "Zor senaryo"}
CAUGHT = "Yakalandı"
MISSED = "Kaçırıldı"
EVENT_COLORS = {
    CAUGHT: "background-color: #DDF3E4; color: #14532D",
    MISSED: "background-color: #FBE0E0; color: #7F1D1D",
}


def _flagged_hours(result, method):
    """Eşiği aşan geçerli saatleri bool dizisi [N] olarak verir (kural: skor >= eşik)."""
    scores = result["scores"][method]
    return (result["valid"] & scores.ge(result["thresholds"][method])).to_numpy()


def _event_outcomes(result, method):
    """Enjekte olayları yakalandı/kaçırıldı diye sınıflar, yanlış alarmları sayar.

    Girdi: analyze_test_range çıktısı ve yöntem kodu.
    Çıktı: (olay tablosu, yanlış alarm blok sayısı).
    Neden: Tanımlar src/utils/anomaly_metrics.event_metrics ile aynıdır. Olay en az
    bir saati işaretlenince yakalanmış sayılır; hiçbir etiketli saatle kesişmeyen
    alarm bloğu yanlış alarmdır.
    """
    types = result["labels"]["anomaly_type"].to_numpy()
    flagged = _flagged_hours(result, method)
    scores = result["scores"][method]
    index = scores.index
    rows = []
    for kind in ANOMALY_TYPES:
        for start, end in runs(types == kind):
            rows.append({
                "Başlangıç": index[start],
                "Bitiş": index[end - 1],
                "Tür": ANOMALY_LABELS[kind],
                "Süre (saat)": end - start,
                "En yüksek skor": float(scores.iloc[start:end].max()),
                "Durum": CAUGHT if flagged[start:end].any() else MISSED,
            })
    table = pd.DataFrame(
        rows, columns=["Başlangıç", "Bitiş", "Tür", "Süre (saat)", "En yüksek skor", "Durum"],
    ).sort_values("Başlangıç").reset_index(drop=True)
    anomalous = types != NO_INJECTION
    false_alarms = sum(not anomalous[start:end].any() for start, end in runs(flagged))
    return table, false_alarms


def _make_figure(result, anomaly_type, method):
    """Tüketimi ve seçili yöntemin skorunu ortak zaman ekseninde çizer.

    Girdi: analyze_test_range çıktısı, enjekte tür (yoksa None) ve yöntem kodu.
    Çıktı: 2 satırlı Plotly figürü: üstte kW, altta skor + eşik. Enjekte olaylar iki
    satırda da kırmızı yarı saydam bant, alarm saatleri skor üzerinde noktadır.
    """
    figure = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.42, 0.58],
    )
    index = result["real_kw"].index
    figure.add_trace(go.Scatter(
        x=index, y=result["real_kw"].where(~result["imputed"]), mode="lines",
        name="Gerçek ölçüm", line=LINE_ACTUAL,
    ), row=1, col=1)
    imputed = result["real_kw"].where(result["imputed"])
    if imputed.notna().any():
        figure.add_trace(go.Scatter(
            x=index, y=imputed, mode="markers", name="Doldurulmuş saat (skor dışı)",
            marker={"color": BASELINE_COLOR, "size": 6},
        ), row=1, col=1)

    injected = result["seed"] is not None
    if injected:
        figure.add_trace(go.Scatter(
            x=index, y=result["injected_kw"], mode="lines", name="Sentetik seri",
            line={"color": ANOMALY_COLOR, "width": 1.5, "dash": "dash"},
        ), row=1, col=1)
        labelled = result["labels"]["is_anomaly"].astype(bool).to_numpy()
        figure.add_trace(go.Scatter(
            x=index[labelled], y=result["injected_kw"].to_numpy()[labelled], mode="markers",
            name=f"Sentetik {ANOMALY_LABELS[anomaly_type]}",
            marker={"color": ANOMALY_COLOR, "symbol": "x", "size": 8},
        ), row=1, col=1)

    scores = result["scores"][method]
    threshold = result["thresholds"][method]
    figure.add_trace(go.Scatter(
        x=index, y=scores, mode="lines", name=METHOD_LABELS[method], line=LINE_MODEL,
    ), row=2, col=1)
    figure.add_hline(
        y=threshold, line_dash="dash", line_color=ACTUAL_COLOR, line_width=1.5, row=2, col=1,
        annotation_text=f"Eşik {number(threshold)}", annotation_position="top left",
    )
    flagged = _flagged_hours(result, method)
    figure.add_trace(go.Scatter(
        x=index[flagged], y=scores.to_numpy()[flagged], mode="markers", name="İşaretlenen saat",
        marker={"color": ANOMALY_COLOR, "size": 7, "line": {"color": "white", "width": 1}},
    ), row=2, col=1)

    if injected:
        # Bantlar en sonda: Plotly, henüz izi olmayan alt grafiğe şekil eklemez.
        labelled = result["labels"]["is_anomaly"].astype(bool).to_numpy()
        half_hour = pd.Timedelta(minutes=30)
        for start, end in runs(labelled):
            for row in (1, 2):  # bant hem tüketim hem skor satırında görünsün
                figure.add_vrect(
                    x0=index[start] - half_hour, x1=index[end - 1] + half_hour,
                    fillcolor=ANOMALY_COLOR, opacity=0.2, line_width=0, row=row, col=1,
                )
        # add_vrect lejanda görünmez; bandın anlamı boş bir izle lejanda yazılır.
        figure.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name="Enjekte edilen anomali",
            marker={"color": ANOMALY_COLOR, "symbol": "square", "size": 12, "opacity": 0.3},
        ))

    figure.update_yaxes(title_text="Aktif güç (kW)", hoverformat=".3f", row=1, col=1)
    figure.update_yaxes(title_text="Skor (birimsiz)", hoverformat=".3f", row=2, col=1)
    figure.update_xaxes(tickformat="%d.%m", hoverformat="%d.%m.%Y %H:%M")
    figure.update_xaxes(title_text="Zaman (gün.ay)", row=2, col=1)
    figure.update_layout(
        title=f"{DATASET_LABELS['household']}, test kümesi: {METHOD_LABELS[method]} ve eşik",
        height=560,
        margin={"t": 95},
    )
    return figure


def _metric_cards(result, method, outcomes, false_alarms):
    """Gösterge kartlarını çizer; gerçek etiket yokken başarı sayısı iddia etmez.

    Enjeksiyon kapalıyken doğrulanmış anomali etiketi bulunmadığından yakalanan olay
    ve yanlış alarm hesaplanamaz; kartlarda bu nedenle "—" görünür.
    """
    injected = result["seed"] is not None
    blocks = len(runs(_flagged_hours(result, method)))
    columns = st.columns(4)
    if injected:
        caught = int((outcomes["Durum"] == CAUGHT).sum())
        columns[0].metric("Yakalanan olay / toplam olay", f"{caught} / {len(outcomes)}", border=True, height=CARD_HEIGHT)
        columns[1].metric("Yanlış alarm sayısı", false_alarms, border=True, height=CARD_HEIGHT)
    else:
        no_label = "Gerçek ölçümde anomali etiketi yoktur; değerlendirme için enjeksiyon seçin."
        columns[0].metric("Yakalanan olay / toplam olay", "—", help=no_label, border=True, height=CARD_HEIGHT)
        columns[1].metric("Yanlış alarm sayısı", "—", help=no_label, border=True, height=CARD_HEIGHT)
    columns[2].metric("Alarm bloğu", blocks, border=True, height=CARD_HEIGHT)
    columns[3].metric(
        "Skorlanan saat", f"{int(result['valid'].sum())} / {len(result['valid'])}", border=True, height=CARD_HEIGHT,
    )


def _event_table(outcomes):
    """Enjekte olay tablosunu gösterir: yakalandı yeşil, kaçırıldı kırmızı."""
    styled = (
        outcomes.style
        .map(EVENT_COLORS.get, subset=["Durum"])
        .format({
            "Başlangıç": "{:%d.%m.%Y %H:%M}",
            "Bitiş": "{:%d.%m.%Y %H:%M}",
            "En yüksek skor": lambda value: number(value),
        }, na_rep="—")
        .hide(axis="index")
    )
    st.dataframe(styled, width="stretch", hide_index=True)


def _alarm_table(result, method):
    """Yöntemin eşik üstü alarm bloklarını kayıtlı olay tablosundan gösterir."""
    alarms = result["events"].loc[result["events"]["Yöntem"] == method].drop(columns="Yöntem")
    if alarms.empty:
        st.info("Seçilen aralıkta eşiği aşan geçerli alarm bloğu yok.")
        return
    alarms = alarms.rename(columns={"Sentetik olayla örtüşüyor": "Enjekte olayla örtüşüyor"}).copy()
    alarms["Enjekte olayla örtüşüyor"] = alarms["Enjekte olayla örtüşüyor"].map({True: "Evet", False: "Hayır"})
    styled = alarms.style.format({
        "Başlangıç": "{:%d.%m.%Y %H:%M}",
        "Bitiş": "{:%d.%m.%Y %H:%M}",
        "En yüksek skor": lambda value: number(value),
        "Eşik": lambda value: number(value),
    }).hide(axis="index")
    st.dataframe(styled, width="stretch", hide_index=True)


def _controls(days):
    """Yöntem, enjeksiyon, şiddet ve tarih kontrollerini tek satırda gösterir."""
    default_end = days[min(29, len(days) - 1)]
    method_col, type_col, severity_col, start_col, end_col = st.columns([3, 2, 2, 2, 2])
    method = method_col.selectbox(
        "Yöntem", options=list(METHODS), index=list(METHODS).index(FINAL_METHOD),
        format_func=METHOD_LABELS.get, key="anomaly-method",
    )
    anomaly_type = type_col.selectbox(
        "Enjeksiyon tipi", options=list(INJECTION_LABELS),
        format_func=INJECTION_LABELS.get, key="anomaly-type",
    )
    scenario = severity_col.selectbox(
        "Şiddet", options=list(SEVERITY_LABELS), format_func=SEVERITY_LABELS.get,
        disabled=anomaly_type == NO_INJECTION, key="anomaly-severity",
    )
    start_day = start_col.date_input(
        "Başlangıç", value=days[0], min_value=days[0], max_value=days[-1],
        format="DD.MM.YYYY", key="anomaly-start",
    )
    end_day = end_col.date_input(
        "Bitiş", value=default_end, min_value=days[0], max_value=days[-1],
        format="DD.MM.YYYY", key="anomaly-end",
    )
    return method, anomaly_type, scenario, start_day, end_day


def render(dataset="household"):
    """Seçili veri seti için anomali sayfasını çizer; canlı analiz yalnız hanede vardır."""
    st.title("Anomali Tespiti")
    if dataset != "household":
        st.info(
            f"Bu demoda canlı anomali analizi yalnız {DATASET_LABELS['household']} için "
            f"hazırlanmıştır. {DATASET_LABELS[dataset]} için kayıtlı test sonuçlarını "
            "Sonuçlar sayfasında görebilirsiniz."
        )
        return

    try:
        config, data_path, models_dir, tables_dir, features = anomaly_demo_paths()
        frame = load_household_data(str(data_path))
        days = complete_test_days(frame, config)
        calibration = load_anomaly_calibration(str(tables_dir))
        models = load_f13_models(str(models_dir), features)
    except (FileNotFoundError, OSError, TypeError, ValueError, KeyError) as error:
        st.error(f"Anomali demosu gerekli veri, model veya eşik dosyalarını yükleyemedi: {error}")
        return

    method, anomaly_type, scenario, start_day, end_day = _controls(days)
    if start_day not in days or end_day not in days or end_day < start_day:
        st.error("Başlangıç ve bitiş, bitiş sonra gelecek şekilde tam 24 saatlik test günleri olmalı.")
        return

    inject = anomaly_type != NO_INJECTION
    start = pd.Timestamp(start_day)
    end = pd.Timestamp(end_day) + pd.Timedelta(hours=23)
    seed = int(config["protocol"]["scenarios"][scenario]["seeds"]["test"][0])
    if inject:
        try:
            validate_injection_range(start, end, anomaly_type, scenario_settings(config, scenario))
        except ValueError as error:
            # Servis mesajındaki iç ayrıntıyı atla; kullanıcıya gereken gün sayısı yeter.
            st.error("Bu enjeksiyon tipi için tarih aralığı çok kısa: " + str(error).split("kuralıyla ")[-1])
            return
        st.warning("Kırmızı bantlar sentetiktir, gerçek ölçüm değildir. Orijinal veri değişmez; enjeksiyon kopya üzerinde yapılır.")

    try:
        result = analyze_test_range(
            frame, config, start, end, inject,
            anomaly_type if inject else ANOMALY_TYPES[0],
            scenario, seed, calibration, checkpoint_signature(models_dir), models,
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
        st.error(f"Anomali analizi tamamlanamadı: {error}")
        return

    outcomes, false_alarms = _event_outcomes(result, method)
    _metric_cards(result, method, outcomes, false_alarms)
    show_chart(
        _make_figure(result, anomaly_type if inject else None, method),
        alt=f"{METHOD_LABELS[method]} skoru, eşik çizgisi, enjekte anomaliler ve işaretlenen saatler",
    )

    if inject:
        st.subheader("Enjekte edilen olaylar")
        _event_table(outcomes)
        with st.expander("Alarm blokları"):
            _alarm_table(result, method)
    else:
        st.subheader("Alarm blokları")
        st.caption("Enjeksiyon kapalı: gerçek anomali etiketi olmadığından alarmlar doğrulanmış anomali sayılmamalıdır.")
        _alarm_table(result, method)

    with st.expander("Detaylar"):
        scenario_note = (
            f"\n- **Bu enjeksiyon:** {SEVERITY_LABELS[scenario]}, seed {seed}. Sentetik etiketler model veya eşik seçiminde kullanılmaz."
            if inject else ""
        )
        st.markdown(
            f"""
- **Eşik:** {number(result['thresholds'][method])}. Doğrulama kümesinde seçilmiştir; test verisine göre değiştirilmez. Skor, yalnız geçmişe bakan {result['windows'][method]} saatlik yumuşatmadan geçer.
- **Yakalanan olay:** Enjekte olayın en az bir saati işaretlendiyse olay yakalanmış sayılır.
- **Yanlış alarm:** Hiçbir enjekte saatle kesişmeyen alarm bloğu. Test verisinde gerçek bir anormallik de olabileceği için bu sayı üst sınır gibi okunmalıdır.
- **Yumuşatma gecikmesi:** Skor geçmişe baktığı için kısa bir olayın etkisi olay bittikten sonra da sürer. Alarm olayın hemen ardından gelirse olay kaçırılmış, alarm da yanlış sayılır. Bu katı tanım, raporlanan olay metrikleriyle aynıdır.
- **Skorlanan saat:** Doldurulmuş saatler alarm dışı bırakılır.
- **Şiddet:** Ana senaryo eğitim kümesinin saat × gün standart sapmasına göre ölçeklenir; zor senaryo daha zayıf, genel katsayılar kullanır.
- **Tek koşu:** Buradaki sayılar seçili aralık ve tek enjeksiyon içindir. Raporlanan genel başarı Sonuçlar sayfasındadır.{scenario_note}
"""
        )
