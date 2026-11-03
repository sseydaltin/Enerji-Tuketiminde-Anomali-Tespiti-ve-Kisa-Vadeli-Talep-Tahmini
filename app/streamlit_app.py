"""Enerji talep tahmini ve anomali tespiti için dört sayfalı Streamlit demosu."""

import streamlit as st

from app.pages import anomaly, forecast, model_info, results
from app.ui import DATASET_LABELS, apply_page_style

PAGES = {
    "Talep Tahmini": forecast.render,
    "Anomali Tespiti": anomaly.render,
    "Sonuçlar": results.render,
    "Model Bilgisi": model_info.render,
}


def _sidebar():
    """Proje bilgisini, sayfa seçimini ve ortak veri seti seçimini gösterir.

    Çıktı: (sayfa adı, veri seti anahtarı). Veri seti burada seçildiği için
    sayfalar arasında geçerken korunur.
    """
    with st.sidebar:
        st.markdown("### Enerji Tüketiminde Anomali Tespiti ve Kısa Vadeli Talep Tahmini")
        st.caption(
            "**Ekip:** Dilara Yıldırım · Şeyda Altın  \n"
            "**Ders:** YMÜ425 Uygulamalı Sinir Ağları  \n"
            "Fırat Üniversitesi, 2026–2027 Güz"
        )
        st.divider()
        page = st.radio("Sayfa", list(PAGES), key="page")
        dataset = st.radio(
            "Veri seti",
            options=list(DATASET_LABELS),
            format_func=DATASET_LABELS.get,
            key="dataset",
        )
    return page, dataset


def main():
    """Sayfa ayarlarını yapar ve seçili sayfayı veri setiyle birlikte çizer."""
    st.set_page_config(
        page_title="Enerji Talep Tahmini ve Anomali Tespiti",
        page_icon="⚡",
        layout="wide",
    )
    apply_page_style()
    page, dataset = _sidebar()
    PAGES[page](dataset)


if __name__ == "__main__":
    main()
