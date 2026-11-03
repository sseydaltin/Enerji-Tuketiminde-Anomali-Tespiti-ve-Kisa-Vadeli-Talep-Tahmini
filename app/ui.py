"""Demo sayfalarının ortak görünümü: Türkçe etiketler, renkler ve Plotly şablonu.

Bu modül yalnız sunum katmanıdır; hiçbir metrik hesaplamaz. Etiketlerin ve
renklerin tek yerde durması, dört sayfanın aynı dili konuşmasını sağlar.
"""

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Renklerin anlamı sabittir: aynı renk her grafikte aynı şeyi gösterir.
ACTUAL_COLOR = "#374151"     # gerçek değer: koyu gri
MODEL_COLOR = "#1F4E9E"      # final model: mavi
BASELINE_COLOR = "#A9B1BC"   # baseline: açık gri (çizgide kesikli)
ANOMALY_COLOR = "#D32F2F"    # anomali: kırmızı
SECOND_COLOR = "#0B9E8F"     # final olmayan karşılaştırma modeli: camgöbeği
HIGHLIGHT_ROW = "#E3EBF7"    # tablolarda final model satırı
CARD_HEIGHT = 112            # delta olan ve olmayan kartlar aynı boyda dursun

LINE_ACTUAL = {"color": ACTUAL_COLOR, "width": 2}
LINE_MODEL = {"color": MODEL_COLOR, "width": 2.5}
LINE_BASELINE = {"color": BASELINE_COLOR, "width": 2, "dash": "dash"}

DATASET_LABELS = {"household": "Hane (tek ev)", "eld": "Şebeke (160 müşteri)"}

# İç kodlar arayüze yansımaz; kod görünen tek yer Model Bilgisi sayfasıdır.
FINAL_MODEL = "F13-ens"
FORECAST_LABELS = {
    "F13-ens": "Topluluk modeli (Seq2Seq + LSTM)",
    "XGBoost": "XGBoost",
    "SN168": "Mevsimsel naif (SN168)",
}
FORECAST_COLORS = {"F13-ens": MODEL_COLOR, "XGBoost": SECOND_COLOR, "SN168": BASELINE_COLOR}

FINAL_METHOD = "A3-s2s-norm"
METHOD_LABELS = {
    "A0": "Hareketli z-skoru",
    "A3-s2s-norm": "Seq2Seq artık skoru",
    "A5-mean-z": "Birleşik skor (z-skoru + Topluluk artığı)",
    "A6": "Geliştirilmiş (deneysel)",
}
METHOD_COLORS = {"A0": BASELINE_COLOR, "A3-s2s-norm": MODEL_COLOR, "A5-mean-z": SECOND_COLOR}

ANOMALY_LABELS = {
    "spike": "Ani yükselme",
    "drop": "Ani düşüş",
    "level_shift": "Seviye kayması",
    "pattern": "Gece örüntüsü",
}

# separators=",." → grafiklerde Türkçe ondalık virgülü.
pio.templates["enerji"] = go.layout.Template(layout={
    "font": {"family": "Arial, Helvetica, sans-serif", "size": 13, "color": ACTUAL_COLOR},
    "title": {"font": {"size": 15}, "x": 0, "xanchor": "left"},
    "paper_bgcolor": "white",
    "plot_bgcolor": "white",
    "separators": ",.",
    "colorway": [MODEL_COLOR, ACTUAL_COLOR, SECOND_COLOR, BASELINE_COLOR],
    "hovermode": "x unified",
    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0, "title": {"text": ""}},
    "margin": {"l": 60, "r": 20, "t": 70, "b": 50},
    "xaxis": {"showgrid": False, "zeroline": False, "linecolor": "#D5D9E0", "ticks": "outside", "tickcolor": "#D5D9E0"},
    "yaxis": {"gridcolor": "#EEF0F3", "zeroline": False, "rangemode": "tozero"},
})
pio.templates.default = "enerji"


def apply_page_style():
    """Üst boşluğu daraltır; böylece grafik ilk ekrana sığar."""
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
        h1 {font-size: 2rem !important; padding-bottom: 0.4rem !important;}
        [data-testid="stMetricValue"] {font-size: 1.65rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def number(value, digits=3):
    """Sayıyı Türkçe ondalık virgülüyle metne çevirir; yalnız görünümü değiştirir."""
    return f"{value:.{digits}f}".replace(".", ",")


def decimal_comma(text):
    """Hazır biçimlenmiş metin sayıdaki ondalık noktayı virgüle çevirir."""
    return text.replace(".", ",")


def show_chart(figure, alt):
    """Grafiği tam genişlikte, projenin şablonuyla ve araç çubuğu olmadan gösterir.

    theme=None: Streamlit'in kendi teması şablonumuzu ezmesin diye.
    """
    # Streamlit zemini ikincil tema rengiyle boyamasın diye beyaz açıkça verilir.
    figure.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(
        figure,
        width="stretch",
        theme=None,
        config={"displayModeBar": False},
        alt=alt,
    )


def highlighted_table(table, column, final_label):
    """Final model satırı vurgulu, indekssiz bir pandas Styler döndürür.

    Girdi: görüntülenecek DataFrame, satırı belirleyen sütun, final model etiketi.
    Çıktı: st.table'a verilecek Styler; veriler değişmez, yalnız biçim eklenir.
    """
    def row_style(row):
        """Final satırına arka plan ve kalın yazı, diğerlerine stil yok."""
        is_final = row[column] == final_label
        style = f"background-color: {HIGHLIGHT_ROW}; font-weight: 600" if is_final else ""
        return [style] * len(row)

    return table.style.apply(row_style, axis=1).hide(axis="index")
