"""Talep Tahmini sayfası: kayıtlı modellerle seçili test günü için 24 saatlik tahmin."""

import numpy as np
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
from app.ui import (
    ACTUAL_COLOR,
    CARD_HEIGHT,
    DATASET_LABELS,
    FINAL_MODEL,
    FORECAST_LABELS,
    LINE_ACTUAL,
    LINE_BASELINE,
    LINE_MODEL,
    number,
    show_chart,
)
from src.utils.metrics import calculate_metrics


def _set_random_day(days):
    """'Rastgele gün' düğmesi: geçerli günlerden birini seçili güne yazar."""
    st.session_state["forecast_day"] = str(np.random.default_rng().choice(days))
    st.session_state["forecast_note"] = "Rastgele seçilmiş bir test günü gösteriliyor."


def _set_ranked_day(frame, days, models, mape_min_y, hardest):
    """'Kolay/Zor örnek' düğmesi: en düşük veya en yüksek hatalı günü seçer.

    Sıralama yalnız grafik örneği içindir; model veya eşik seçimine bağlanmaz.
    """
    errors = daily_test_mae(frame, tuple(days), models, mape_min_y)
    if hardest:
        st.session_state["forecast_day"] = max(errors, key=errors.get)
        st.session_state["forecast_note"] = "Zor örnek: Topluluk modelinin test kümesinde en yüksek MAE aldığı gün."
    else:
        st.session_state["forecast_day"] = min(errors, key=errors.get)
        st.session_state["forecast_note"] = "Kolay örnek: Topluluk modelinin test kümesinde en düşük MAE aldığı gün."


def _clear_note():
    """Tarih elle değiştirilince önceki örnek notunu kaldırır."""
    st.session_state["forecast_note"] = ""


def _controls(frame, days, models, mape_min_y):
    """Tarih seçimi ve üç düğmeyi tek satırda gösterir, seçili günü döndürür.

    Düğmeler on_click ile çalışır: Streamlit, bir widget çizildikten sonra onun
    değerinin değiştirilmesine izin vermediği için gün, kutu çizilmeden önce yazılır.
    """
    if st.session_state.get("forecast_day") not in days:
        st.session_state["forecast_day"] = days[0]
    day_col, random_col, easy_col, hard_col = st.columns([3, 1, 1, 1], vertical_alignment="bottom")
    # Takvim yerine liste: yalnız 48+24 saati eksiksiz günler seçilebilsin diye.
    day = day_col.selectbox("Test günü", days, key="forecast_day", on_change=_clear_note)
    random_col.button("Rastgele gün", width="stretch", on_click=_set_random_day, args=(days,))
    easy_col.button(
        "Kolay örnek", width="stretch", on_click=_set_ranked_day,
        args=(frame, days, models, mape_min_y, False),
    )
    hard_col.button(
        "Zor örnek", width="stretch", on_click=_set_ranked_day,
        args=(frame, days, models, mape_min_y, True),
    )
    return day


def _metric_cards(result, baseline_mae):
    """Dört gösterge kartını çizer; iyileşme kartında delta yeşil/kırmızıdır.

    İyileşme = 100 × (SN168 MAE − Topluluk MAE) / SN168 MAE; pozitifse model daha iyidir.
    """
    model_mae = result["metrics"]["MAE"]
    mape = result["metrics"]["MAPE*"]
    columns = st.columns(4)
    columns[0].metric("Topluluk MAE", f"{number(model_mae)} kW", border=True, height=CARD_HEIGHT)
    columns[1].metric(
        "MAPE",
        "Tanımsız" if not np.isfinite(mape) else f"%{number(mape, 1)}",
        border=True,
        height=CARD_HEIGHT,
    )
    columns[2].metric("SN168 MAE", f"{number(baseline_mae)} kW", border=True, height=CARD_HEIGHT)
    if baseline_mae > 0:
        gain = 100.0 * (baseline_mae - model_mae) / baseline_mae
        difference = baseline_mae - model_mae
        # Streamlit, "-" ile başlayan deltayı kırmızı ve aşağı oklu gösterir.
        delta = (
            f"{number(difference)} kW daha az hata" if difference >= 0
            else f"-{number(-difference)} kW daha fazla hata"
        )
        columns[3].metric("SN168'e göre iyileşme", f"{'-' if gain < 0 else ''}%{number(abs(gain), 1)}", delta=delta, border=True, height=CARD_HEIGHT)
    else:
        columns[3].metric("SN168'e göre iyileşme", "Tanımsız", border=True, height=CARD_HEIGHT)


def _make_figure(result, day, dataset_label):
    """Gerçek seri, model ve baseline tahminini ortak zaman ekseninde çizer.

    Girdi: predict_test_day çıktısı (48 saat geçmiş + 24 saat hedef/tahmin), gün.
    Çıktı: tahmin başlangıcı dikey çizgiyle işaretli Plotly figürü.
    """
    figure = go.Figure()
    # Geçmiş ve hedef aynı gerçek seridir; tek çizgi olarak çizilir.
    actual_times = result["history_times"].append(result["target_times"])
    actual_values = np.concatenate((result["history_kw"], result["actual_kw"]))
    figure.add_trace(go.Scatter(
        x=actual_times, y=actual_values, mode="lines", name="Gerçek tüketim", line=LINE_ACTUAL,
    ))
    figure.add_trace(go.Scatter(
        x=result["target_times"], y=result["f13_kw"], mode="lines+markers",
        name=FORECAST_LABELS[FINAL_MODEL], line=LINE_MODEL, marker={"size": 6},
    ))
    figure.add_trace(go.Scatter(
        x=result["target_times"], y=result["sn168_kw"], mode="lines",
        name=FORECAST_LABELS["SN168"], line=LINE_BASELINE,
    ))
    # add_vline yerine shape + annotation: tarih ekseninde metinli vline hata verebiliyor.
    start = result["target_times"][0]
    figure.add_shape(
        type="line", x0=start, x1=start, xref="x", y0=0, y1=1, yref="paper",
        line={"color": ACTUAL_COLOR, "width": 1.5, "dash": "dot"},
    )
    figure.add_annotation(
        x=start, y=1, xref="x", yref="paper", text="Tahmin başlangıcı",
        showarrow=False, xanchor="left", yanchor="top", xshift=6,
    )
    figure.update_xaxes(title_text="Zaman (gün.ay saat)", tickformat="%d.%m %H:%M", hoverformat="%d.%m.%Y %H:%M")
    figure.update_yaxes(title_text="Aktif güç (kW)", hoverformat=".3f")
    figure.update_layout(
        title=f"{dataset_label}, test kümesi, {day}: gerçek tüketim ve 24 saatlik tahmin",
        height=420,
    )
    return figure


def _details(config, days):
    """Teknik açıklamaları ana görünümü kalabalıklaştırmadan açılır bölümde verir."""
    with st.expander("Detaylar"):
        st.markdown(
            f"""
- **Test dönemi:** {config['split']['test']['start']} – {config['split']['test']['end']}. Seçilebilir gün sayısı: {len(days)}. Yalnız önceki 48 saati ve hedef 24 saati doldurulmamış, tam ölçüm içeren günler listelenir.
- **Topluluk modeli:** 0,8 × Seq2Seq + 0,2 × LSTM. Girdi 48 saat, çıktı 24 saat. Ağırlıklar kayıtlıdır; demoda eğitim yapılmaz.
- **SN168 (mevsimsel naif):** Bir önceki haftanın aynı saatlerini tahmin olarak kopyalar.
- **MAE:** Seçili günün 24 saati üzerinden ortalama mutlak hata (kW).
- **MAPE:** Yalnız gerçek tüketimin {config['evaluation']['mape_min_y']} kW üstünde olduğu saatlerde hesaplanır; hane verisinde değer yüksek çıkabilir çünkü düşük tüketimde küçük hatalar büyük yüzdelere dönüşür.
- **İyileşme:** 100 × (SN168 MAE − Topluluk MAE) / SN168 MAE. Bu değer yalnız seçili güne aittir; genel başarı için Sonuçlar sayfasına bakın.
- **Kolay/zor örnek:** Günler test hatasına göre sıralanır ama bu yalnız görsel örnek seçmek içindir. Model veya eşik seçimi test kümesinden yapılmaz.
"""
        )


def render(dataset="household"):
    """Seçili veri seti için tahmin sayfasını çizer; canlı tahmin yalnız hanede vardır."""
    st.title("Talep Tahmini")
    dataset_label = DATASET_LABELS[dataset]
    if dataset != "household":
        st.info(
            f"Bu demoda canlı tahmin yalnız {DATASET_LABELS['household']} için hazırlanmıştır. "
            f"{dataset_label} için kayıtlı test sonuçlarını Sonuçlar sayfasında görebilirsiniz."
        )
        return

    try:
        config, data_path, models_dir, expected_features = demo_paths()
        frame = load_household_data(str(data_path))
        models = load_f13_models(str(models_dir), tuple(expected_features))
        days = valid_test_days(frame, config)
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as error:
        st.error(f"Tahmin verisi veya modeli kullanılamıyor: {error}")
        return

    mape_min_y = config["evaluation"]["mape_min_y"]
    day = _controls(frame, days, models, mape_min_y)
    note = st.session_state.get("forecast_note")
    if note:
        st.caption(note)

    try:
        result = predict_test_day(frame, day, models, mape_min_y)
    except (KeyError, IndexError, TypeError, ValueError, RuntimeError) as error:
        st.error(f"{day} için tahmin üretilemedi: {error}")
        return
    target_times = result["target_times"]
    if len(target_times) != 24 or target_times[0].strftime("%Y-%m-%d") != day:
        st.error("Tahmin zaman ekseni seçilen 24 saatlik günle uyuşmuyor.")
        return

    # SN168'in günlük MAE'si, modelinkiyle aynı metrik fonksiyonundan gelir.
    baseline = calculate_metrics(result["actual_kw"], result["sn168_kw"], mape_min_y=mape_min_y)
    _metric_cards(result, baseline["MAE"])
    show_chart(
        _make_figure(result, day, dataset_label),
        alt=f"{day} test günü için gerçek tüketim, Topluluk modeli ve SN168 tahmini",
    )
    _details(config, days)
