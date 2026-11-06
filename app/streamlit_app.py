"""Enerji tahmini ve anomali tespiti için dört sayfalı Streamlit demosu."""

import streamlit as st

from app.pages import anomaly, forecast, model_info, results


def main():
    """Dört sayfa arasında gezinmeyi sağlar."""
    st.set_page_config(
        page_title="Enerji Tahmini ve Anomali Demo",
        page_icon="⚡",
        layout="wide",
    )
    st.sidebar.title("Enerji Demo")
    page = st.sidebar.radio(
        "Sayfa",
        ["Forecast", "Anomaly", "Results", "Model Info"],
    )
    pages = {
        "Forecast": forecast.render,
        "Anomaly": anomaly.render,
        "Results": results.render,
        "Model Info": model_info.render,
    }
    pages[page]()


if __name__ == "__main__":
    main()
