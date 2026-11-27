"""Ana sayfa: projenin tek ekranlık özeti. Üç veri seti üç kartta, her birinin ana sonucu ve ilgili sayfaya bağlantı.

Hesap yok: sayılar kayıtlı final test dosyalarından okunur (hane/şebeke: app.results_services; LEAD: app.lead_services).
"""

from html import escape

import streamlit as st

from app.lead_services import FINAL_LEAD_MODEL, LEAD_MODELS, slice_results
from app.results_services import load_results_bundle
from app.ui import FINAL_METHOD, number, page_intro


def household_eld_numbers(dataset):
    """(final tahmin satırı, final anomali yönteminin test ana senaryo satırı) — kayıtlı final test tabloları."""
    bundle = load_results_bundle(dataset)
    forecast = bundle["forecast"].set_index("model_key").loc["F13-ens"]
    anomaly = bundle["anomaly"]
    anomaly = anomaly[(anomaly["split"] == "test") & (anomaly["scenario"] == "main")
                      & (anomaly["method"] == FINAL_METHOD)].iloc[0]
    return forecast, anomaly


def lead_numbers():
    """Final LEAD modelinin test satırları: {dilim etiketi: satır} (slice_results, 3 seed ort.)."""
    table = slice_results()
    return table[table["model"] == LEAD_MODELS[FINAL_LEAD_MODEL]].set_index("dilim")


def _card_html(title, subtitle, metrics, note, url, link_label):
    """Yumuşak kenarlı veri seti kartı (HTML; iç içe sütun yok, dar ekranda kartlar alt alta dizilir)."""
    numbers = "".join(f"<div><b>{escape(value)}</b><span>{escape(label)}</span></div>" for label, value, _ in metrics)
    link = f"<a href='{url}' target='_self'>→ {escape(link_label)}</a>" if url else ""
    return (f"<div class='ds-card'><h3>{escape(title)}</h3><div class='ds-sub'>{escape(subtitle)}</div>"
            f"<div class='ds-nums'>{numbers}</div><div class='kpi-desc'>{escape(note)}</div>"
            f"<div style='margin-top:0.6rem'>{link}</div></div>")


def render(links=None):
    """links: {"forecast", "anomaly", "lead", "results"} → st.Page (adresleri kart bağlantılarında kullanılır)."""
    links = links or {}

    def url(name):
        page = links.get(name)
        return f"./{page.url_path}" if page is not None else None
    page_intro("Enerji Tüketiminde Anomali Tespiti ve Kısa Vadeli Talep Tahmini",
               "Projenin iki sorusu: yarın ne kadar enerji tüketilecek (24 saatlik tahmin) ve hangi tüketim olağandışı "
               "(anomali tespiti)? Aşağıda üç veri setinin kayıtlı final test sonuçları; ayrıntı için kartlardaki "
               "bağlantıları kullanın.")
    cards = []
    try:
        f, a = household_eld_numbers("household")
        cards.append(_card_html("Hane (tek ev)", "UCI IHEPC · saatlik · sentetik anomali",
                                [("MAE kazancı (SN168'e göre)", f"%{number(f['gain_vs_sn168_pct'], 1)}", ""),
                                 ("Olay F1 (anomali)", number(a["event_f1_mean"]), "")],
                                "Tahmin haftalık naif yöntemi geçiyor; anomali olay F1 hedefi (0,70) karşılanmadı.",
                                url("forecast"), "Talep Tahmini"))
        f, a = household_eld_numbers("eld")
        cards.append(_card_html("Şebeke (160 müşteri)", "UCI ELD · saatlik · aynı yöntemler",
                                [("MAPE", f"%{number(f['MAPE*'], 2)}", ""), ("Olay F1 (anomali)", number(a["event_f1_mean"]), "")],
                                f"R² {number(f['R2'])}. Toplam yük daha düzenli olduğu için hata hane'den çok düşük.",
                                url("results"), "Sonuçlar"))
    except (FileNotFoundError, OSError, ValueError, KeyError) as error:
        st.error(f"Kayıtlı hane/şebeke sonuçları okunamadı: {error}")
    try:
        lead = lead_numbers()
        cards.append(_card_html("LEAD (200 bina)", "Gerçek, elle konmuş etiketler · ek doğrulama",
                                [("Davranışsal PR-AUC", number(lead.loc["Davranışsal (other)", "pr_auc_mean"]), ""),
                                 ("Genel ROC-AUC", number(lead.loc["Genel", "roc_auc_mean"]), "")],
                                f"Final model {LEAD_MODELS[FINAL_LEAD_MODEL]}, test. Kolay alt türler (boşluk, donmuş "
                                "sayaç) yüksek; davranışsal anomaliler zor.", url("lead"), "Gerçek Etiketli Veri"))
    except (FileNotFoundError, OSError, ValueError, KeyError) as error:
        st.error(f"Kayıtlı LEAD sonuçları okunamadı: {error}")
    st.markdown(f"<div class='kpi-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)
    st.caption("Ekip: Şeyda ALTIN · Dilara YILDIRIM — YMÜ425 Uygulamalı Sinir Ağları, Fırat Üniversitesi, 2026–2027 Güz. "
               "Demo eğitim yapmaz; bütün sayılar kayıtlı final dosyalarından gelir.")
