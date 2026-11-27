"""Enerji talep tahmini ve anomali tespiti demosu: ana sayfa + beş sayfa (st.navigation + ikonlu st.page_link menüsü).

Görünüm: .streamlit/config.toml (tema), app/theme.py (renk ve Plotly şablonu), app/style.css (tek CSS dosyası).
Veri seti seçimi her sayfanın üstündeki yatay şerittedir (app.ui.dataset_strip); kenar çubuğunda yalnız gezinme var.
"""

import streamlit as st

from app.views import anomaly, forecast, home, lead, model_info, results
from app.ui import apply_page_style


def main():
    """Sayfa ayarları, stil ve gezinme. Her sayfa kendi render fonksiyonuyla çizilir."""
    st.set_page_config(page_title="Enerji Talep Tahmini ve Anomali Tespiti", page_icon="⚡", layout="wide")
    apply_page_style()
    pages = {
        "forecast": st.Page(forecast.render, title="Talep Tahmini", icon=":material/show_chart:",
                            url_path="talep-tahmini"),
        "anomaly": st.Page(anomaly.render, title="Anomali Tespiti", icon=":material/troubleshoot:",
                           url_path="anomali-tespiti"),
        "lead": st.Page(lead.render, title="Gerçek Etiketli Veri (LEAD)", icon=":material/apartment:",
                        url_path="gercek-etiketli-veri"),
        "results": st.Page(results.render, title="Sonuçlar", icon=":material/bar_chart:", url_path="sonuclar"),
        "model_info": st.Page(model_info.render, title="Model Bilgisi", icon=":material/schema:",
                              url_path="model-bilgisi"),
    }
    home_page = st.Page(lambda: home.render(pages), title="Ana Sayfa", icon=":material/home:", url_path="ana-sayfa",
                        default=True)
    # Gezinme gizli modda; menü aşağıda st.page_link ile ikonlu ve sade çizilir. Sayfa modülleri app/views/ altındadır
    # (app/pages/ adı Streamlit'in eski otomatik sayfa algılamasını tetikleyip "Page not found" iletisi veriyordu).
    navigation = st.navigation([home_page, *pages.values()], position="hidden")
    with st.sidebar:
        st.markdown("**Enerji Talep Tahmini ve Anomali Tespiti**")
        for page in (home_page, *pages.values()):
            st.page_link(page, label=page.title, icon=page.icon)
        st.divider()
        st.caption("YMÜ425 Uygulamalı Sinir Ağları · Fırat Üniversitesi · 2026–2027 Güz  \n"
                   "Dilara YILDIRIM · Şeyda ALTIN")
    navigation.run()


if __name__ == "__main__":
    main()
