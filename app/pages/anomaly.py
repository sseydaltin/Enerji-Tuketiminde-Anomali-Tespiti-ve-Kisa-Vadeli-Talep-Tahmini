"""Anomali Tespiti sayfası: hane test döneminde skor, eşik ve olay görünümü."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.anomaly_services import (
    CAUGHT,
    METHODS,
    MISSED,
    SEVERITY_TYPES,
    analyze_test_range,
    anomaly_demo_paths,
    checkpoint_signature,
    complete_test_days,
    event_outcomes,
    flagged_hours,
    load_anomaly_calibration,
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
from src.data.inject_anomalies import ANOMALY_TYPES
from src.utils.anomaly_metrics import runs

NO_INJECTION = "none"
DEFAULT_TYPE = "level_shift"
INJECTION_LABELS = {**ANOMALY_LABELS, NO_INJECTION: "Yok (gerçek ölçüm)"}
SCENARIO_LABELS = {"main": "Ana (σ tabanlı)", "hard": "Zor (katsayı tabanlı)"}
SEVERITY_STEPS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
EVENT_COLORS = {
    CAUGHT: "background-color: #DDF3E4; color: #14532D",
    MISSED: "background-color: #FBE0E0; color: #7F1D1D",
}


def _make_figure(result, anomaly_type, method):
    """Tüketimi ve seçili yöntemin skorunu ortak zaman ekseninde çizer.

    Girdi: analyze_test_range çıktısı, enjekte tür (yoksa None) ve yöntem kodu.
    Çıktı: 2 satırlı Plotly figürü: üstte kW, altta ham ve yumuşatılmış skor + eşik.
    Eşik yumuşatılmış skora uygulanır. Enjekte olaylar iki satırda da kırmızı yarı
    saydam bant, alarm saatleri yumuşatılmış skor üzerinde noktadır.
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
    labelled = result["labels"]["is_anomaly"].astype(bool).to_numpy()
    if injected:
        # Yalnız enjekte saatler kırmızı çizilir; gerçek seri okunur kalsın diye gerisi boştur.
        figure.add_trace(go.Scatter(
            x=index, y=result["injected_kw"].where(labelled), mode="lines+markers",
            name=f"Sentetik {ANOMALY_LABELS[anomaly_type]}",
            line={"color": ANOMALY_COLOR, "width": 1.5}, marker={"color": ANOMALY_COLOR, "size": 4},
        ), row=1, col=1)

    scores = result["scores"][method]
    threshold = result["thresholds"][method]
    figure.add_trace(go.Scatter(
        x=index, y=result["raw_scores"][method], mode="lines", name="Ham skor",
        line={"color": BASELINE_COLOR, "width": 1},
    ), row=2, col=1)
    figure.add_trace(go.Scatter(
        x=index, y=scores, mode="lines", name="Yumuşatılmış skor", line=LINE_MODEL,
    ), row=2, col=1)
    figure.add_hline(
        y=threshold, line_dash="dash", line_color=ACTUAL_COLOR, line_width=1.5, row=2, col=1,
        annotation_text=f"Eşik {number(threshold)}",
        annotation_position="top right",
    )
    flagged = flagged_hours(result, method)
    figure.add_trace(go.Scatter(
        x=index[flagged], y=scores.to_numpy()[flagged], mode="markers", name="İşaretlenen saat",
        marker={"color": ANOMALY_COLOR, "size": 7, "line": {"color": "white", "width": 1}},
    ), row=2, col=1)

    if injected:
        # Bantlar en sonda: Plotly, henüz izi olmayan alt grafiğe şekil eklemez.
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
    blocks = len(runs(flagged_hours(result, method)))
    columns = st.columns(4)
    if injected:
        caught = int((outcomes["Durum"] == CAUGHT).sum())
        columns[0].metric("Yakalanan olay / toplam olay", f"{caught} / {len(outcomes)}", border=True, height=CARD_HEIGHT)
        columns[1].metric("Yanlış alarm sayısı", false_alarms, border=True, height=CARD_HEIGHT)
    else:
        no_label = "Gerçek ölçümde anomali etiketi yoktur; değerlendirme için enjeksiyon seçin."
        columns[0].metric("Yakalanan olay / toplam olay", "—", help=no_label, border=True, height=CARD_HEIGHT)
        columns[1].metric("Yanlış alarm sayısı", "—", help=no_label, border=True, height=CARD_HEIGHT)
    columns[2].metric(
        "Toplam alarm", blocks, help="Eşiği aşan ardışık saatler tek alarm sayılır.",
        border=True, height=CARD_HEIGHT,
    )
    columns[3].metric(
        "Eşik (yumuşatılmış skor)", number(result["thresholds"][method]),
        help="Doğrulama kümesinde seçilmiştir; test verisine göre değiştirilmez.",
        border=True, height=CARD_HEIGHT,
    )


def _event_table(outcomes):
    """Enjekte olay tablosunu gösterir: yakalandı yeşil, kaçırıldı kırmızı."""
    styled = (
        outcomes.style
        .map(EVENT_COLORS.get, subset=["Durum"])
        .format({
            "Başlangıç": "{:%d.%m.%Y %H:%M}",
            "Bitiş": "{:%d.%m.%Y %H:%M}",
            "Ham skor (en yüksek)": lambda value: number(value),
            "Yumuşatılmış skor (en yüksek)": lambda value: number(value),
        }, na_rep="—")
        .hide(axis="index")
    )
    st.dataframe(styled, width="stretch", hide_index=True)


def _alarm_table(result, method):
    """Yöntemin eşik üstü alarm bloklarını kayıtlı olay tablosundan gösterir."""
    alarms = result["events"].loc[result["events"]["Yöntem"] == method].drop(columns="Yöntem")
    if alarms.empty:
        st.info("Seçilen aralıkta eşiği aşan geçerli alarm yok.")
        return
    alarms = alarms.rename(columns={
        "Sentetik olayla örtüşüyor": "Enjekte olayla örtüşüyor",
        "En yüksek skor": "Yumuşatılmış skor (en yüksek)",
    }).copy()
    alarms["Enjekte olayla örtüşüyor"] = alarms["Enjekte olayla örtüşüyor"].map({True: "Evet", False: "Hayır"})
    styled = alarms.style.format({
        "Başlangıç": "{:%d.%m.%Y %H:%M}",
        "Bitiş": "{:%d.%m.%Y %H:%M}",
        "Yumuşatılmış skor (en yüksek)": lambda value: number(value),
        "Eşik": lambda value: number(value),
    }).hide(axis="index")
    st.dataframe(styled, width="stretch", hide_index=True)


def _controls(days):
    """Yöntem, enjeksiyon, senaryo, şiddet ve tarih kontrollerini tek satırda gösterir.

    Tarih aralığı varsayılan olarak tüm test dönemidir: yakalanan/toplam kartı ve
    tablo, seçilmiş bir kesit değil, kopyanın bütün olaylarını gösterir.
    """
    method_col, type_col, scenario_col, severity_col, start_col, end_col = st.columns([3, 2, 2, 2, 2, 2])
    method = method_col.selectbox(
        "Yöntem", options=list(METHODS), index=list(METHODS).index(FINAL_METHOD),
        format_func=METHOD_LABELS.get, key="anomaly-method",
    )
    anomaly_type = type_col.selectbox(
        "Enjeksiyon tipi", options=list(INJECTION_LABELS),
        index=list(INJECTION_LABELS).index(DEFAULT_TYPE),
        format_func=INJECTION_LABELS.get, key="anomaly-type",
    )
    scenario = scenario_col.selectbox(
        "Senaryo", options=list(SCENARIO_LABELS), format_func=SCENARIO_LABELS.get,
        disabled=anomaly_type == NO_INJECTION, key="anomaly-scenario",
    )
    severity_active = anomaly_type in SEVERITY_TYPES
    severity = severity_col.select_slider(
        "Şiddet", options=SEVERITY_STEPS, value=1.0,
        format_func=lambda value: f"×{number(value, 2)}",
        disabled=not severity_active, key="anomaly-severity",
        help="×1,00 protokolün kendisidir. Düşüş ve gece örüntüsünün ayarlanacak büyüklüğü yoktur.",
    )
    start_day = start_col.date_input(
        "Başlangıç", value=days[0], min_value=days[0], max_value=days[-1],
        format="DD.MM.YYYY", key="anomaly-start",
    )
    end_day = end_col.date_input(
        "Bitiş", value=days[-1], min_value=days[0], max_value=days[-1],
        format="DD.MM.YYYY", key="anomaly-end",
    )
    # Kaydırıcı kapalıysa önceki değeri hesaba taşınmaz.
    return method, anomaly_type, scenario, severity if severity_active else 1.0, start_day, end_day


def _capture_table(tables_dir):
    """Kayıtlı tek tip yakalama tablosunu gösterir (python -m app.capture_table)."""
    path = tables_dir / "demo_capture_by_type.csv"
    if not path.is_file():
        st.caption("Tablo henüz üretilmemiş: python -m app.capture_table")
        return
    table = pd.read_csv(path)
    table["Yakalama oranı"] = [
        f"{number(mean)} ± {number(std)}"
        for mean, std in zip(table["Yakalama oranı (ort)"], table["Yakalama oranı (std)"])
    ]
    table["Yanlış alarm (ort)"] = table["Yanlış alarm (ort)"].map(lambda value: number(value, 1))
    st.table(table[[
        "Tip", "Yöntem", "Seed başına yakalanan / toplam", "Yakalama oranı", "Yanlış alarm (ort)",
    ]].style.hide(axis="index"))
    st.caption(
        "Her koşuda yalnız o tip enjekte edilir, tüm test dönemi, ana senaryo, şiddet ×1, 5 seed. "
        "Bu tablo demo denetimi içindir, final sonuç değildir; raporlanan tablolar dört tipi birlikte "
        "enjekte eder ve Sonuçlar sayfasındadır. Hiçbir seçim buna göre yapılmaz."
    )


def _details(result, method, inject, scenario, seed, severity):
    """Tanımları ve yumuşatma ödünleşimini açılır bölümde anlatır."""
    window = result["windows"][method]
    injection_note = ""
    if inject:
        injection_note = (
            f"\n- **Bu enjeksiyon:** {SCENARIO_LABELS[scenario]}, şiddet ×{number(severity, 2)}, seed {seed}. "
            "Sentetik etiketler model veya eşik seçiminde kullanılmaz."
        )
        if severity != 1.0:
            injection_note += " Şiddet ×1 dışında sonuçlar protokol dışıdır, raporla karşılaştırılmaz."
    st.markdown(
        f"""
- **Ham ve yumuşatılmış skor:** Ham skor her saatin kendi hatasıdır. Eşik, ham skora değil, yalnız geçmişe bakan {window} saatlik ortalamaya uygulanır. Bu yüzden ham skoru eşiği geçen bir olay kaçırılmış görünebilir.
- **Yumuşatma ödünleşimi:** Ortalama, tek saatlik gürültüyü ve kısa sıçramaları bastırır. Seviye kayması gibi uzun olaylarda işe yarar, 1–2 saatlik ani yükselmelerde ise skoru geciktirip zayıflatır. Bu bilinen bir ödünleşimdir.
- **Olay bitince alarm:** Skor geçmişe baktığı için alarm olay bittikten sonra gelebilir. Bu durumda olay kaçırılmış, alarm yanlış sayılır. Bu katı tanım raporlanan olay metrikleriyle aynıdır.
- **Eşik:** {number(result['thresholds'][method])}. Doğrulama kümesinde seçilmiştir; test verisine göre değiştirilmez.
- **Yakalanan olay:** Enjekte olayın en az bir saatinde yumuşatılmış skor eşiği geçtiyse olay yakalanmış sayılır.
- **Yanlış alarm:** Hiçbir enjekte saatle kesişmeyen alarm. Enjeksiyonsuz testte de bu yöntem 8 alarm verir; bunlar gerçek ölçümdeki olağandışı saatler olabilir. Bu yüzden sayı üst sınır gibi okunmalıdır.
- **Skorlanan saat:** {int(result['valid'].sum())} / {len(result['valid'])}. Doldurulmuş saatler alarm dışı bırakılır.
- **Senaryolar:** Ana senaryoda ani yükselme +3–5 eğitim σ, seviye kayması ×2–2,5, düşüş yalnız 1 kW üstünde başlar. Zor senaryoda ani yükselme ×2–4, seviye kayması ×1,5.
- **Enjeksiyon kapsamı:** Protokoldeki gibi tüm test kopyasına yapılır; tarih aralığı yalnız görünümü seçer. Tek koşudur, raporlanan genel başarı Sonuçlar sayfasındadır.{injection_note}
"""
    )


def render(dataset="household"):
    """Hane test döneminin anomali sayfasını çizer; canlı analiz yalnız hanede vardır."""
    st.title("Anomali Tespiti")
    if dataset != "household":
        st.info(
            f"Canlı anomali analizi yalnız {DATASET_LABELS['household']} için hazırlanmıştır; aşağıda hane "
            f"gösterilir. {DATASET_LABELS[dataset]} için kayıtlı test sonuçları Sonuçlar sayfasındadır."
        )

    try:
        config, data_path, models_dir, tables_dir, features = anomaly_demo_paths()
        frame = load_household_data(str(data_path))
        days = complete_test_days(frame, config)
        calibration = load_anomaly_calibration(str(tables_dir))
        models = load_f13_models(str(models_dir), features)
    except (FileNotFoundError, OSError, TypeError, ValueError, KeyError) as error:
        st.error(f"Anomali demosu gerekli veri, model veya eşik dosyalarını yükleyemedi: {error}")
        return

    method, anomaly_type, scenario, severity, start_day, end_day = _controls(days)
    if start_day not in days or end_day not in days or end_day < start_day:
        st.error("Başlangıç ve bitiş, bitiş sonra gelecek şekilde tam 24 saatlik test günleri olmalı.")
        return

    inject = anomaly_type != NO_INJECTION
    start = pd.Timestamp(start_day)
    end = pd.Timestamp(end_day) + pd.Timedelta(hours=23)
    seed = int(config["protocol"]["scenarios"][scenario]["seeds"]["test"][0])
    if inject:
        st.warning("Kırmızı bantlar sentetiktir, gerçek ölçüm değildir. Orijinal veri değişmez; enjeksiyon kopya üzerinde yapılır.")

    try:
        result = analyze_test_range(
            frame, config, start, end, inject,
            anomaly_type if inject else ANOMALY_TYPES[0],
            scenario, seed, calibration, checkpoint_signature(models_dir), models, severity,
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
        st.error(f"Anomali analizi tamamlanamadı: {error}")
        return

    outcomes, false_alarms = event_outcomes(result, method)
    _metric_cards(result, method, outcomes, false_alarms)
    show_chart(
        _make_figure(result, anomaly_type if inject else None, method),
        alt=f"{METHOD_LABELS[method]} ham ve yumuşatılmış skoru, eşik çizgisi, enjekte anomaliler ve işaretlenen saatler",
    )

    if inject:
        st.subheader("Enjekte edilen olaylar")
        if outcomes.empty:
            st.info("Seçilen tarih aralığında enjekte olay yok. Tarih aralığını genişletin.")
        else:
            _event_table(outcomes)
        with st.expander("Tüm alarmlar"):
            _alarm_table(result, method)
    else:
        st.subheader("Alarmlar")
        st.caption("Enjeksiyon kapalı: gerçek anomali etiketi olmadığından alarmlar doğrulanmış anomali sayılmamalıdır.")
        _alarm_table(result, method)

    with st.expander("Detaylar"):
        _details(result, method, inject, scenario, seed, severity)
    with st.expander("Tip bazlı yakalama oranı (5 seed)"):
        _capture_table(tables_dir)
