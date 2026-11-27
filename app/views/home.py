"""Ana sayfa: projenin tek ekranlık özeti. Üç veri seti üç kartta, her birinin ana sonucu ve ilgili sayfaya bağlantı.

Hesap yok: sayılar kayıtlı final test dosyalarından okunur (hane/şebeke: app.results_services; LEAD: app.lead_services).
"""

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


def _card(column, title, subtitle, metrics, note, page, link_label):
    """Yumuşak kenarlı kart: başlık, iki ana sayı, kısa not, sayfa bağlantısı."""
    with column.container(border=True):
        st.markdown(f"### {title}")
        st.caption(subtitle)
        left, right = st.columns(2)
        for target, (label, value, hint) in zip((left, right), metrics):
            target.metric(label, value, help=hint)
        st.markdown(f"<span style='color:#6B7075'>{note}</span>", unsafe_allow_html=True)
        if page is not None:
            st.page_link(page, label=link_label, icon=":material/arrow_forward:")


def render(links=None):
    """links: {"forecast", "anomaly", "lead", "results"} → st.Page (gezinme streamlit_app.py'de kurulur)."""
    links = links or {}
    page_intro("Enerji Tüketiminde Anomali Tespiti ve Kısa Vadeli Talep Tahmini",
               "Projenin iki sorusu: yarın ne kadar enerji tüketilecek (24 saatlik tahmin) ve hangi tüketim olağandışı "
               "(anomali tespiti)? Aşağıda üç veri setinin kayıtlı final test sonuçları; ayrıntı için kartlardaki "
               "bağlantıları kullanın.")
    columns = st.columns(3)
    try:
        f, a = household_eld_numbers("household")
        _card(columns[0], "Hane (tek ev)", "UCI IHEPC · saatlik · sentetik anomali",
              [("MAE kazancı", f"%{number(f['gain_vs_sn168_pct'], 1)}", "Topluluk modelinin mevsimsel naife (SN168) göre MAE kazancı; 3 seed, test."),
               ("Olay F1", number(a["event_f1_mean"]), "Anomali: Seq2Seq artık skoru, test, ana senaryo.")],
              "Tahmin referansı geçiyor; anomali olay F1 hedefi (0,70) karşılanmadı.",
              links.get("forecast"), "Talep Tahmini sayfası")
        f, a = household_eld_numbers("eld")
        _card(columns[1], "Şebeke (160 müşteri)", "UCI ELD · saatlik · aynı yöntemler",
              [("MAPE", f"%{number(f['MAPE*'], 2)}", "Topluluk modeli, 3 seed, test."),
               ("Olay F1", number(a["event_f1_mean"]), "Anomali: Seq2Seq artık skoru, test, ana senaryo.")],
              f"R² {number(f['R2'])}. Toplam yük daha düzenli olduğu için hata hane'den çok düşük.",
              links.get("results"), "Sonuçlar sayfası")
    except (FileNotFoundError, OSError, ValueError, KeyError) as error:
        columns[0].error(f"Kayıtlı hane/şebeke sonuçları okunamadı: {error}")
    try:
        lead = lead_numbers()
        _card(columns[2], "LEAD (200 bina)", "Gerçek, elle konmuş etiketler · ek doğrulama",
              [("Other PR-AUC", number(lead.loc["Davranışsal (other)", "pr_auc_mean"]),
                f"Davranışsal (other) anomaliler; final model ({LEAD_MODELS[FINAL_LEAD_MODEL]}), test. Taban ≈ "
                f"{number(lead.loc['Davranışsal (other)', 'taban'], 4)}."),
               ("ROC-AUC", number(lead.loc["Genel", "roc_auc_mean"]), "Genel ROC-AUC; final model, test, 3 seed.")],
              "Kolay alt türler (boşluk, donmuş sayaç) yüksek; davranışsal anomaliler zor.",
              links.get("lead"), "Gerçek Etiketli Veri sayfası")
    except (FileNotFoundError, OSError, ValueError, KeyError) as error:
        columns[2].error(f"Kayıtlı LEAD sonuçları okunamadı: {error}")
    st.caption("Ekip: Şeyda ALTIN · Dilara YILDIRIM — YMÜ425 Uygulamalı Sinir Ağları, Fırat Üniversitesi, 2026–2027 Güz. "
               "Demo eğitim yapmaz; bütün sayılar kayıtlı final dosyalarından gelir.")
