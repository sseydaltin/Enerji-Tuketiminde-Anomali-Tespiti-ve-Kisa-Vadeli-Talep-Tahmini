"""Demo sayfalarının ortak görünümü: Türkçe etiketler, renkler, sayfa düzeni yardımcıları.

Bu modül yalnız sunum katmanıdır; hiçbir metrik hesaplamaz. Renkler ve Plotly şablonu `app/theme.py`'de,
özel CSS `app/style.css`'tedir; burada aynı adlarla sayfalara verilir (eski adlar korunur).
"""

from html import escape
from pathlib import Path

import streamlit as st

from app import theme

# Renklerin anlamı sabittir: aynı renk her grafikte aynı şeyi gösterir (app/theme.py).
ACTUAL_COLOR = theme.ACTUAL          # gerçek değer: koyu gri
MODEL_COLOR = theme.FINAL            # final model: yumuşak lacivert (tek vurgu rengi)
BASELINE_COLOR = theme.BASELINE      # baseline: açık gri (çizgide kesikli)
ANOMALY_COLOR = theme.ANOMALY        # anomali: vermilyon (yalnız anomali vurgusu)
SECOND_COLOR = theme.COMPARISON      # final olmayan karşılaştırma modeli: mavi-yeşil
THIRD_COLOR = theme.THIRD            # üçüncü model: yumuşak mor
SUBTYPE_COLORS = theme.SUBTYPES      # LEAD alt türleri
HIGHLIGHT_ROW = "#E6ECF3"            # tablolarda final model satırı (vurgu renginin açığı)
CARD_HEIGHT = 112                    # delta olan ve olmayan kartlar aynı boyda dursun

LINE_ACTUAL = {"color": ACTUAL_COLOR, "width": 1.8}
LINE_MODEL = {"color": MODEL_COLOR, "width": 2.2}
LINE_BASELINE = {"color": BASELINE_COLOR, "width": 1.6, "dash": "dash"}

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
    "A6-dual": "Geliştirilmiş: kısa sıçramalar (deneysel)",
    "A6-dual-floor": "Geliştirilmiş: sıçrama + taban yük (deneysel)",
}
METHOD_COLORS = {"A0": BASELINE_COLOR, "A3-s2s-norm": MODEL_COLOR, "A5-mean-z": SECOND_COLOR}

ANOMALY_LABELS = {
    "spike": "Ani yükselme",
    "drop": "Ani düşüş",
    "level_shift": "Seviye kayması",
    "pattern": "Gece örüntüsü",
}

STYLE_FILE = Path(__file__).with_name("style.css")


def apply_page_style():
    """Tek CSS dosyasını (app/style.css) sayfaya ekler: kart kenarları, yazı boyutları, sayfa genişliği."""
    st.markdown(f"<style>{STYLE_FILE.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def page_intro(title, text):
    """Sayfa başlığı ve altında 1–2 cümlelik "Bu sayfada ne görüyorsunuz?" kutusu."""
    st.title(title)
    st.markdown(f"<div class='page-intro'><b>Bu sayfada ne görüyorsunuz?</b> {text}</div>", unsafe_allow_html=True)


def dataset_strip(key_suffix, label="Veri seti"):
    """Sayfanın üstünde yatay veri seti seçimi. Seçim sayfalar arasında korunur (session_state['dataset']).

    Widget anahtarı sayfaya özeldir; ortak değer ayrı tutulur. Neden: başka sayfada çizilmeyen widget'ın durumu
    Streamlit tarafından silinir, ortak anahtar bu yüzden widget'a bağlanmaz.
    """
    st.session_state.setdefault("dataset", "household")
    widget = f"dataset_{key_suffix}"

    def sync():
        st.session_state["dataset"] = st.session_state[widget]
    choice = st.segmented_control(label, options=list(DATASET_LABELS), format_func=DATASET_LABELS.get,
                                  default=st.session_state["dataset"], key=widget, on_change=sync)
    return choice or st.session_state["dataset"]


def kpi_cards(cards):
    """En çok 3 KPI kartı: cards = [(etiket, değer metni, tek satırlık açıklama), ...].

    İç içe sütun kullanılmaz: kartlar CSS ızgarasıyla dizilir (app/style.css, .kpi-grid); dar ekranda kendiliğinden
    alt alta geçer, metin harf harf kırılmaz. Değerler çağıran sayfada kayıtlı dosyalardan hazırlanır.
    """
    if len(cards) > 3:
        raise ValueError("Bir sayfada en çok 3 KPI kartı gösterilir.")
    items = "".join(f"<div class='kpi-card'><div class='kpi-label'>{escape(label)}</div>"
                    f"<div class='kpi-value'>{escape(value)}</div><div class='kpi-desc'>{escape(desc)}</div></div>"
                    for label, value, desc in cards)
    st.markdown(f"<div class='kpi-grid'>{items}</div>", unsafe_allow_html=True)


def kpi_row(cards):
    """Geriye uyumlu ad: kpi_cards ile aynı (etiket, değer, açıklama)."""
    kpi_cards(cards)


def compact_table(frame, columns, styler=None, key=None, selectable=False):
    """Yatay kaydırma gerektirmeyen tablo: uzun sütun adları kısaltılır, tam ad sütun ipucunda (help) durur.

    columns: {asıl ad: (kısa ad, ipucu)}; listede olmayan sütunlar adıyla kalır. styler: kısa adlı tabloya
    uygulanacak biçimlendirme fonksiyonu (DataFrame → Styler). selectable: tek satır seçimi (olay tablosu).
    Çıktı: st.dataframe dönüşü (seçim bilgisi için). Değerler değişmez; yalnız başlıklar değişir.
    """
    shown = frame.rename(columns={name: short for name, (short, _) in columns.items()})
    config = {short: st.column_config.Column(short, help=hint) for short, hint in columns.values() if hint}
    data = styler(shown) if styler else shown
    options = {"on_select": "rerun", "selection_mode": "single-row", "key": key} if selectable else {}
    height = 36 * (len(frame) + 1) + 4          # bütün satırlar görünsün: tablo içinde dikey kaydırma olmasın
    return st.dataframe(data, hide_index=True, width="stretch", height=height, column_config=config, **options)


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
    # Grafik zemini tema yüzeyi (beyaz kart); Streamlit teması şablonu ezmesin diye theme=None.
    figure.update_layout(plot_bgcolor=theme.SURFACE, paper_bgcolor=theme.SURFACE)
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
