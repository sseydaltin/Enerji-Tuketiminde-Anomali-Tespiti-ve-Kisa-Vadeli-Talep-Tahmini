"""Household test döneminde protokole uygun anomali görselleştirmesi."""

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
from src.anomaly_protocol import scenario_settings

METHOD_COLORS = {
    "A0": "#1f77b4",
    "A3-s2s-norm": "#ff7f0e",
    "A5-mean-z": "#2ca02c",
}
ANOMALY_LABELS = {
    "spike": "Ani yükselme (spike)",
    "drop": "Ani düşüş (drop)",
    "level_shift": "Seviye kayması",
    "pattern": "Gece örüntüsü",
}
SCENARIO_LABELS = {"main": "Ana — train σ tabanlı", "hard": "Zor — genel ayarlar"}


def _make_figure(result, anomaly_type):
    """Gerçek/sentetik tüketim ve üç skor-eşik çiftini ortak zaman ekseninde çizer."""
    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        subplot_titles=(
            "Saatlik tüketim (kW)",
            "A0 — hareketli z-skoru",
            "A3-s2s-norm — F10 artık / OOF std",
            "A5-mean-z — A0 + F13 artık",
        ),
    )
    index = result["real_kw"].index
    real = result["real_kw"].where(~result["imputed"])
    imputed = result["real_kw"].where(result["imputed"])
    figure.add_trace(
        go.Scatter(x=index, y=real, name="Gerçek ölçüm", mode="lines",
                   line={"color": "#253858", "width": 1.5}),
        row=1, col=1,
    )
    if imputed.notna().any():
        figure.add_trace(
            go.Scatter(x=index, y=imputed, name="Doldurulmuş saat (skor dışı)",
                       mode="markers", marker={"color": "#9aa5b1", "size": 5}),
            row=1, col=1,
        )
    if result["seed"] is not None:
        figure.add_trace(
            go.Scatter(x=index, y=result["injected_kw"], name="Sentetik seri (enjeksiyonlu)",
                       mode="lines", line={"color": "#d62728", "dash": "dash", "width": 1.5}),
            row=1, col=1,
        )
        injected_times = result["labels"].index[result["labels"]["is_anomaly"].astype(bool)]
        figure.add_trace(
            go.Scatter(
                x=injected_times,
                y=result["injected_kw"].loc[injected_times],
                name=f"Sentetik {ANOMALY_LABELS[anomaly_type]}",
                mode="markers",
                marker={"color": "#d62728", "symbol": "x", "size": 9},
            ),
            row=1, col=1,
        )

    for row, method in enumerate(METHODS, start=2):
        scores = result["scores"][method]
        threshold = result["thresholds"][method]
        figure.add_trace(
            go.Scatter(x=scores.index, y=scores, name=method,
                       mode="lines", line={"color": METHOD_COLORS[method]}),
            row=row, col=1,
        )
        figure.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="#d62728",
            annotation_text=f"Validation eşiği {threshold:.3f}",
            annotation_position="top left",
            row=row,
            col=1,
        )
        flagged = result["valid"] & scores.ge(threshold)
        if flagged.any():
            figure.add_trace(
                go.Scatter(
                    x=scores.index[flagged],
                    y=scores.loc[flagged],
                    name=f"{method} alarmı",
                    mode="markers",
                    marker={"color": "#d62728", "size": 6, "symbol": "circle"},
                    showlegend=False,
                ),
                row=row, col=1,
            )
    figure.update_yaxes(title_text="kW", row=1, col=1)
    for row in (2, 3, 4):
        figure.update_yaxes(title_text="Skor", row=row, col=1)
    figure.update_xaxes(title_text="Saat", row=4, col=1)
    figure.update_layout(
        height=920,
        title="Household test verisi — enjeksiyon varsa sentetik seri ayrıca işaretlidir",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"t": 110},
    )
    return figure


def render():
    """Household test döneminde A0, A3-s2s-norm ve A5-mean-z skorlarını gösterir."""
    st.title("Anomali tespiti")
    st.caption(
        "Skorlar yalnız test aralığında hesaplanır. Smoothing ve alarm eşikleri "
        "kayıtlı validation seçiminden; artık ölçekleri train OOF tablosundan gelir. "
        "Test verisi eşik seçimi veya kalibrasyon için kullanılmaz."
    )
    st.info("ELD için canlı anomali analizi devre dışı: yerel ELD veri dosyası mevcut değil.")

    try:
        config, data_path, models_dir, tables_dir, features = anomaly_demo_paths()
        frame = load_household_data(str(data_path))
        days = complete_test_days(frame, config)
        calibration = load_anomaly_calibration(str(tables_dir))
        models = load_f13_models(str(models_dir), features)
    except (FileNotFoundError, OSError, TypeError, ValueError, KeyError) as error:
        st.error(f"Anomali demosu gerekli Household veri/model/kalibrasyon dosyalarını yükleyemedi: {error}")
        return

    default_end = days[min(29, len(days) - 1)]
    with st.form("anomaly-controls"):
        start_col, end_col = st.columns(2)
        start_day = start_col.date_input(
            "Başlangıç günü",
            value=days[0],
            min_value=days[0],
            max_value=days[-1],
            key="anomaly-start",
        )
        end_day = end_col.date_input(
            "Bitiş günü",
            value=default_end,
            min_value=days[0],
            max_value=days[-1],
            key="anomaly-end",
        )
        inject = st.checkbox("Sentetik anomali enjeksiyonu uygula", value=False)
        type_col, scenario_col = st.columns(2)
        anomaly_type = type_col.selectbox(
            "Anomali türü", options=list(ANOMALY_LABELS), format_func=ANOMALY_LABELS.get
        )
        scenario = scenario_col.selectbox(
            "Şiddet protokolü", options=list(SCENARIO_LABELS), format_func=SCENARIO_LABELS.get
        )
        submitted = st.form_submit_button("Analizi çalıştır", type="primary")

    if not submitted:
        st.info("Test döneminden bir tarih aralığı seçin ve analizi çalıştırın.")
        return
    if start_day not in days or end_day not in days or end_day < start_day:
        st.error("Başlangıç ve bitiş, tam 24 saatlik Household test günleri olmalı.")
        return

    start = pd.Timestamp(start_day)
    end = pd.Timestamp(end_day) + pd.Timedelta(hours=23)
    seed = int(config["protocol"]["scenarios"][scenario]["seeds"]["test"][0])
    if inject:
        settings = scenario_settings(config, scenario)
        try:
            validate_injection_range(start, end, anomaly_type, settings)
        except ValueError as error:
            st.error(str(error))
            return
        st.warning(
            f"Gösterilen anomali sentetiktir; gerçek ölçüm değildir. "
            f"Protokol: {SCENARIO_LABELS[scenario]}, seed={seed}. "
            "Sentetik etiketler model/eşik seçimi veya performans iddiası için kullanılmaz."
        )
    else:
        st.info(
            "Enjeksiyon kapalı: skorlar gerçek test ölçümleri üzerinde gösterilir. "
            "Gerçek anomali etiketleri bulunmadığından alarm işaretleri doğrulanmış "
            "anomali olarak yorumlanmamalıdır."
        )

    try:
        result = analyze_test_range(
            frame,
            config,
            start,
            end,
            inject,
            anomaly_type,
            scenario,
            seed,
            calibration,
            checkpoint_signature(models_dir),
            models,
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
        st.error(f"Anomali analizi tamamlanamadı: {error}")
        return

    st.plotly_chart(
        _make_figure(result, anomaly_type),
        width="stretch",
        alt="Household tüketimi, sentetik enjeksiyon ve A0, A3-s2s-norm, A5-mean-z skorlarıyla validation eşikleri",
    )
    valid_hours = int(result["valid"].sum())
    st.caption(
        f"Skorlanabilir test saati: {valid_hours}/{len(result['valid'])}. "
        "Doldurulmuş saatler alarm maskesinden çıkarılmıştır."
    )
    st.subheader("Tespit edilen olaylar")
    if result["events"].empty:
        st.info("Seçilen aralıkta kayıtlı validation eşiklerini aşan geçerli olay yok.")
    else:
        st.dataframe(
            result["events"],
            width="stretch",
            hide_index=True,
            column_config={
                "Başlangıç": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                "Bitiş": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                "En yüksek skor": st.column_config.NumberColumn(format="%.3f"),
                "Eşik": st.column_config.NumberColumn(format="%.3f"),
            },
        )
