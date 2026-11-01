"""Household için kayıtlı checkpoint'lerle çalışan tahmin sayfası."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.services import (
    daily_test_mae,
    demo_paths,
    load_f13_models,
    load_household_data,
    predict_test_day,
    valid_test_days,
)


def render():
    """Household 48→24 saat F13/SN168 demo sayfasını gösterir."""
    st.title("Talep Tahmini")
    st.write("Household test döneminden geçerli bir gün için kayıtlı modellerle tahmin.")
    dataset = st.selectbox("Veri kümesi", ["Household", "ELD (devre dışı)"])
    if dataset != "Household":
        st.error(
            "ELD canlı tahmini devre dışı: data/raw/eld/LD2011_2014.txt ve "
            "data/processed/eld_hourly.parquet yerelde bulunmuyor. ELD sonuç tabloları "
            "Results sayfasında statik olarak görüntülenebilir."
        )
        return

    try:
        config, data_path, models_dir, expected_features = demo_paths()
        frame = load_household_data(str(data_path))
        models = load_f13_models(str(models_dir), tuple(expected_features))
        days = valid_test_days(frame, config)
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as error:
        st.error(f"Forecast verisi veya modeli kullanılamıyor: {error}")
        return

    st.caption(
        f"Gerçek test bölmesi: {config['split']['test']['start']} – "
        f"{config['split']['test']['end']}. Geçerli gün sayısı: {len(days)}."
    )
    choice = st.radio(
        "Test örneği",
        ["Tarih seç", "Rastgele geçerli gün", "Kolay örnek", "Zor örnek"],
        horizontal=True,
    )
    if choice == "Tarih seç":
        day = st.selectbox("Test günü", days)
    elif choice == "Rastgele geçerli gün":
        if "forecast_random_day" not in st.session_state:
            st.session_state["forecast_random_day"] = str(
                np.random.default_rng().choice(days)
            )
        if st.button("Başka rastgele gün"):
            st.session_state["forecast_random_day"] = str(
                np.random.default_rng().choice(days)
            )
        day = st.session_state["forecast_random_day"]
        if day not in days:
            day = days[0]
            st.session_state["forecast_random_day"] = day
    else:
        st.warning(
            "Kolay/zor gün, test hedeflerinin F13 MAE'sine göre yalnız grafik "
            "örneği seçmek için bulunur. Bu sıralama model veya eşik seçimine "
            "bağlanmamalıdır."
        )
        try:
            errors = daily_test_mae(
                frame, tuple(days), models, config["evaluation"]["mape_min_y"]
            )
        except (ValueError, RuntimeError) as error:
            st.error(f"Test günü örnekleri hesaplanamadı: {error}")
            return
        day = min(errors, key=errors.get) if choice == "Kolay örnek" else max(errors, key=errors.get)

    try:
        result = predict_test_day(
            frame, day, models, config["evaluation"]["mape_min_y"]
        )
    except (KeyError, IndexError, TypeError, ValueError, RuntimeError) as error:
        st.error(f"{day} için tahmin üretilemedi: {error}")
        return

    target_times = result["target_times"]
    if len(target_times) != 24 or target_times[0].strftime("%Y-%m-%d") != day:
        st.error("Tahmin zaman ekseni seçilen 24 saatlik günle uyuşmuyor.")
        return
    st.subheader(f"Seçili test günü: {day}")
    st.metric("F13 örnek MAE", f"{result['metrics']['MAE']:.3f} kW")
    mape = result["metrics"]["MAPE*"]
    st.metric("F13 örnek MAPE*", "Tanımsız" if not np.isfinite(mape) else f"{mape:.1f}%")
    st.caption("MAPE*: yalnızca gerçek tüketimi > 0,1 kW olan hedef saatlerinde hesaplanır.")

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=result["history_times"], y=result["history_kw"],
        mode="lines", name="Önceki 48 saat – gerçek",
    ))
    figure.add_trace(go.Scatter(
        x=target_times, y=result["actual_kw"],
        mode="lines+markers", name="Sonraki 24 saat – gerçek",
    ))
    figure.add_trace(go.Scatter(
        x=target_times, y=result["f13_kw"],
        mode="lines+markers", name="F13 ensemble (0,8 F10 + 0,2 F5)",
    ))
    figure.add_trace(go.Scatter(
        x=target_times, y=result["sn168_kw"],
        mode="lines", name="Seasonal Naive 168",
    ))
    figure.update_layout(
        title=f"Household test – {day}: gerçek tüketim ve tahmin",
        xaxis_title="Zaman",
        yaxis_title="Aktif güç (kW)",
        hovermode="x unified",
        legend_title="Seri",
    )
    st.plotly_chart(figure, use_container_width=True)
    st.caption(
        "Tahmin ve baseline noktaları, seçilen günün gerçek hedef saatleriyle "
        "aynı zaman indeksine hizalanmıştır. Test verisi yalnız örnek gösterim/"
        "metrik içindir; model ve eşik ayarı yapılmaz."
    )
