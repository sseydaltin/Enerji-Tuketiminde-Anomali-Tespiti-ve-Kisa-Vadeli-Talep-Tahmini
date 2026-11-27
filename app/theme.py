"""Demonun tek görsel teması: renkler ve Plotly şablonu (yalnız görünüm; hiçbir metrik hesaplanmaz).

Palet renk körlüğüne uygun Okabe–Ito paletinden türetildi. Kural: aynı model ve aynı alt tür her sayfada aynı
renktedir; doygun kırmızı-turuncu (vermilyon) yalnız anomali vurgusu içindir.
"""

import plotly.graph_objects as go
import plotly.io as pio

# --- Zemin ve metin (.streamlit/config.toml ile aynı) --------------------------------------------
BACKGROUND = "#FAFAF7"
SURFACE = "#FFFFFF"
TEXT = "#2F3437"
MUTED = "#6B7075"
GRID = "#ECECE6"
BORDER = "#E3E3DC"
ACCENT = "#3D5A80"            # tek vurgu rengi (yumuşak lacivert): final model ve etkileşim

# --- Anlamlı renkler (Okabe–Ito tabanlı) ---------------------------------------------------------
ACTUAL = "#4A4F55"            # gerçek değer: koyu gri
FINAL = ACCENT                # final model: yumuşak lacivert
COMPARISON = "#3E9C86"        # karşılaştırma modeli: yumuşak mavi-yeşil (Okabe–Ito #009E73'ün açığı)
THIRD = "#B07AA1"             # üçüncü model: yumuşak mor (Okabe–Ito #CC79A7'nin açığı)
BASELINE = "#A3A9B0"          # referans / naif: açık gri, kesikli
ANOMALY = "#D55E00"           # anomali vurgusu: vermilyon (yalnız burada doygun renk)
SUBTYPES = {"gap_edge": "#E69F00",   # boşluk kenarı: turuncu
            "frozen": "#56B4E9",     # donmuş okuma: gök mavisi
            "other": ANOMALY}        # davranışsal: vermilyon

FONT = "Inter, 'Segoe UI', 'Source Sans Pro', Arial, sans-serif"

pio.templates["demo"] = go.layout.Template(layout={
    "font": {"family": FONT, "size": 14, "color": TEXT},          # projeksiyon için okunaklı
    "title": {"font": {"size": 15, "color": TEXT}, "x": 0, "xanchor": "left", "yref": "container", "y": 0.98,
              "yanchor": "top"},          # başlık en üstte; lejant onun altında, çizim alanının üstünde
    "paper_bgcolor": SURFACE,
    "plot_bgcolor": SURFACE,
    "separators": ",.",                                   # Türkçe ondalık virgülü
    "colorway": [FINAL, COMPARISON, THIRD, BASELINE, ACTUAL],
    "hovermode": "x unified",
    "hoverlabel": {"bgcolor": SURFACE, "bordercolor": BORDER, "font": {"family": FONT, "size": 13}},
    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "title": {"text": ""},
               "font": {"size": 13}},
    "margin": {"l": 56, "r": 16, "t": 96, "b": 60},
    # Eksen etiketleri en az 13 px, eksen başlıkları 14 px (projeksiyon okunurluğu).
    "xaxis": {"showgrid": False, "zeroline": False, "linecolor": BORDER, "ticks": "outside", "tickcolor": BORDER,
              "tickfont": {"color": MUTED, "size": 13}, "title": {"font": {"size": 14}}},
    "yaxis": {"gridcolor": GRID, "gridwidth": 1, "zeroline": False, "rangemode": "tozero",
              "tickfont": {"color": MUTED, "size": 13}, "title": {"font": {"size": 14}}},
}, data={"scatter": [go.Scatter(line={"width": 2.2})]})   # ince çizgiler: bütün çizgi grafiklerinin varsayılanı
pio.templates.default = "demo"
