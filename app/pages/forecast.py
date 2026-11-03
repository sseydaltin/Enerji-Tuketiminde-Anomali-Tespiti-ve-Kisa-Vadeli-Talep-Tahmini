"""Talep Tahmini sayfası: kayıtlı ağırlıklarla seçili test günü için 24 saatlik tahmin."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.eld_services import HARD_DAY, load_eld_forecasts, predict_eld_day
from app.results_services import load_results_bundle
from app.services import (
    best_decile_days,
    daily_total_error_pct,
    demo_paths,
    household_comparison,
    load_f13_models,
    load_household_data,
    median_day,
    predict_test_day,
    valid_test_days,
    worst_decile_days,
)
from app.ui import (
    ACTUAL_COLOR,
    BASELINE_COLOR,
    CARD_HEIGHT,
    DATASET_LABELS,
    FINAL_MODEL,
    FORECAST_LABELS,
    LINE_ACTUAL,
    LINE_BASELINE,
    LINE_MODEL,
    MODEL_COLOR,
    number,
    show_chart,
)
from src.utils.metrics import calculate_metrics

# Şebeke yükü ~100.000 kW olduğundan 3 ondalık anlamsızdır.
DIGITS = {"household": 3, "eld": 1}
# Dağılım grafiğinde uç değerler bu aralığa kırpılır; yalnız görünümü etkiler.
GAIN_AXIS_LIMIT = 100.0


def _display_day(day):
    """YYYY-MM-DD metnini arayüzdeki gün.ay.yıl biçimine çevirir."""
    return pd.Timestamp(day).strftime("%d.%m.%Y")


def _pick_day(dataset, day, note):
    """Düğme geri çağrısı: günü ve neden seçildiğini oturuma yazar."""
    st.session_state[f"forecast_day_{dataset}"] = day
    st.session_state[f"forecast_note_{dataset}"] = note


def _pick_from(dataset, candidates, note):
    """Düğme geri çağrısı: verilen günlerden birini rastgele seçer."""
    _pick_day(dataset, str(np.random.default_rng().choice(candidates)), note)


def _clear_note(dataset):
    """Tarih elle değiştirilince önceki örnek notunu kaldırır."""
    st.session_state[f"forecast_note_{dataset}"] = ""


def _controls(dataset, days, table):
    """Tarih seçimi ve üç düğmeyi tek satırda gösterir, seçili günü döndürür.

    Düğmeler on_click ile çalışır: Streamlit, bir widget çizildikten sonra onun
    değerinin değiştirilmesine izin vermediği için gün, kutu çizilmeden önce yazılır.
    """
    key = f"forecast_day_{dataset}"
    if st.session_state.get(key) not in days:
        st.session_state[key] = median_day(table)
        st.session_state[f"forecast_note_{dataset}"] = "Varsayılan gün: iyileşmesi medyana en yakın test günü."
    day_col, random_col, easy_col, hard_col = st.columns([3, 1, 1, 1], vertical_alignment="bottom")
    # Takvim yerine liste: yalnız geçmişi ve hedefi eksiksiz günler seçilebilsin diye.
    day = day_col.selectbox(
        "Test günü", days, key=key, format_func=_display_day, on_change=_clear_note, args=(dataset,),
    )
    random_col.button(
        "Rastgele gün", width="stretch", on_click=_pick_from,
        args=(dataset, days, "Rastgele seçilmiş bir test günü."),
    )
    easy_col.button(
        "Kolay örnek", width="stretch", on_click=_pick_from,
        args=(dataset, best_decile_days(table), "Kolay örnek: Topluluk MAE'sinin en iyi %10'luk diliminden."),
    )
    if dataset == "eld":
        hard_col.button(
            "Zor örnek", width="stretch", on_click=_pick_day,
            args=(dataset, HARD_DAY, "Zor örnek: 25 Aralık 2014, Noel. Yük komşu günlerin yaklaşık yarısına düşer."),
        )
    else:
        hard_col.button(
            "Zor örnek", width="stretch", on_click=_pick_from,
            args=(dataset, worst_decile_days(table), "Zor örnek: Topluluk MAE'sinin en kötü %10'luk diliminden."),
        )
    return day


def overall_result_text(forecast, digits):
    """Kayıtlı final sonuçtan test geneli cümlesini kurar.

    Girdi: load_results_bundle(...)["forecast"] tablosu, ondalık sayısı.
    Çıktı: "Test geneli: MAE 0,406 kW, SN168'den %26,0 daha iyi" biçiminde metin.
    Neden: Tek gün izleyiciyi yanıltmasın; asıl sonuç her zaman görünür kalır.
    """
    row = forecast.loc[forecast["model_key"] == FINAL_MODEL].iloc[0]
    return (
        f"**Test geneli:** MAE {number(row['MAE'], digits)} kW, "
        f"SN168'den %{number(row['gain_vs_sn168_pct'], 1)} daha iyi"
    )


def _metric_cards(result, baseline_mae, digits):
    """Seçili günün dört kartını çizer; iyileşme kartında delta yeşil/kırmızıdır.

    İyileşme = 100 × (SN168 MAE − Topluluk MAE) / SN168 MAE; pozitifse model daha iyidir.
    """
    model_mae = result["metrics"]["MAE"]
    total_error = daily_total_error_pct(result["actual_kw"], result["f13_kw"])
    columns = st.columns(4)
    columns[0].metric("Seçili gün: Topluluk MAE", f"{number(model_mae, digits)} kW", border=True, height=CARD_HEIGHT)
    columns[1].metric(
        "Seçili gün: Günlük toplam tüketim hatası",
        "Tanımsız" if not np.isfinite(total_error) else f"%{number(total_error, 1)}",
        border=True,
        height=CARD_HEIGHT,
    )
    columns[2].metric("Seçili gün: SN168 MAE", f"{number(baseline_mae, digits)} kW", border=True, height=CARD_HEIGHT)
    if baseline_mae > 0:
        gain = 100.0 * (baseline_mae - model_mae) / baseline_mae
        difference = baseline_mae - model_mae
        # Streamlit, "-" ile başlayan deltayı kırmızı ve aşağı oklu gösterir.
        delta = (
            f"{number(difference, digits)} kW daha az hata" if difference >= 0
            else f"-{number(-difference, digits)} kW daha fazla hata"
        )
        columns[3].metric(
            "Seçili gün: SN168'e göre iyileşme",
            f"{'-' if gain < 0 else ''}%{number(abs(gain), 1)}",
            delta=delta, border=True, height=CARD_HEIGHT,
        )
    else:
        columns[3].metric("Seçili gün: SN168'e göre iyileşme", "Tanımsız", border=True, height=CARD_HEIGHT)


def _make_figure(result, day, dataset_label, digits):
    """Gerçek seri, model ve baseline tahminini ortak zaman ekseninde çizer.

    Girdi: tahmin sözlüğü (48 saat geçmiş + 24 saat hedef/tahmin), gün, veri seti etiketi.
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
    figure.update_yaxes(title_text="Aktif güç (kW)", hoverformat=f".{digits}f")
    if digits == 1:
        figure.update_yaxes(tickformat=",.0f")   # Şebeke: "120k" yerine 120.000
    figure.update_layout(
        title=f"{dataset_label}, test kümesi, {_display_day(day)}: gerçek tüketim ve 24 saatlik tahmin",
        height=350,
    )
    return figure


def _distribution_figure(table, day, dataset_label):
    """Günlük iyileşme dağılımını, medyan ve seçili gün çizgileriyle çizer.

    Girdi: comparison_table çıktısı, seçili gün, veri seti etiketi.
    Çıktı: model önde (mavi) / SN168 önde (gri) histogram. ±100'ün dışındaki günler
    okunabilirlik için kenar çubuğuna kırpılır; özet sayılar kırpılmamış veriden gelir.
    """
    gains = table["gain_pct"].dropna().clip(-GAIN_AXIS_LIMIT, GAIN_AXIS_LIMIT)
    bins = {"start": -GAIN_AXIS_LIMIT, "end": GAIN_AXIS_LIMIT + 10, "size": 10}
    figure = go.Figure()
    figure.add_trace(go.Histogram(
        x=gains[gains >= 0], xbins=bins, name="Model önde", marker_color=MODEL_COLOR,
        hovertemplate="%{y} gün<extra>Model önde</extra>",
    ))
    figure.add_trace(go.Histogram(
        x=gains[gains < 0], xbins=bins, name="SN168 önde", marker_color=BASELINE_COLOR,
        hovertemplate="%{y} gün<extra>SN168 önde</extra>",
    ))
    selected = float(table.loc[table["day"] == day, "gain_pct"].iloc[0])
    selected = float(np.clip(selected, -GAIN_AXIS_LIMIT, GAIN_AXIS_LIMIT))
    median = float(table["gain_pct"].median())
    for x, label, dash, y in (
        (median, f"Medyan %{number(median, 1)}", "dash", 1.0),
        (selected, "Seçili gün", "dot", 0.86),
    ):
        figure.add_shape(
            type="line", x0=x, x1=x, xref="x", y0=0, y1=1, yref="paper",
            line={"color": ACTUAL_COLOR, "width": 1.5, "dash": dash},
        )
        figure.add_annotation(
            x=x, y=y, xref="x", yref="paper", text=label, showarrow=False,
            xanchor="left", yanchor="top", xshift=5, bgcolor="white",
        )
    figure.update_xaxes(
        title_text="SN168'e göre günlük MAE iyileşmesi (%)",
        range=[-GAIN_AXIS_LIMIT - 5, GAIN_AXIS_LIMIT + 15],
    )
    figure.update_yaxes(title_text="Gün sayısı")
    figure.update_layout(
        title=f"{dataset_label}, test kümesi: günlük iyileşme dağılımı",
        barmode="stack", bargap=0.05, height=320, hovermode="closest",
    )
    return figure


def _distribution_section(table, day, dataset_label):
    """Tek günün tüm testi temsil edip etmediğini gösteren özeti ve dağılımı çizer."""
    wins = int((table["model_mae"] < table["baseline_mae"]).sum())
    st.subheader("Test günlerinin tamamı")
    columns = st.columns(3)
    columns[0].metric("Modelin önde olduğu gün", f"{wins} / {len(table)}", border=True)
    columns[1].metric("SN168'in önde olduğu gün", f"{len(table) - wins} / {len(table)}", border=True)
    columns[2].metric("Medyan günlük iyileşme", f"%{number(table['gain_pct'].median(), 1)}", border=True)
    show_chart(
        _distribution_figure(table, day, dataset_label),
        alt=f"{dataset_label} test günlerinde SN168'e göre günlük MAE iyileşmesinin dağılımı",
    )
    clipped = int((table["gain_pct"].abs() > GAIN_AXIS_LIMIT).sum())
    st.caption(
        f"Her gün 00:00'da başlayan pencere, seed 42. Grafikte ±%{GAIN_AXIS_LIMIT:.0f} dışındaki "
        f"{clipped} gün en uç sütunda gösterilir; sayılar kırpılmamış veriden gelir."
    )


def _details(dataset, config, days, result):
    """Teknik açıklamaları ana görünümü kalabalıklaştırmadan açılır bölümde verir."""
    mape = result["metrics"]["MAPE*"]
    mape_value = "tanımsız" if not np.isfinite(mape) else f"%{number(mape, 1)}"
    if dataset == "household":
        mape_note = (
            f"Seçili günde {mape_value}. Yalnız gerçek tüketimin {config['evaluation']['mape_min_y']} kW üstünde "
            "olduğu saatlerde hesaplanır; düşük tüketimli saatlerde payda çok küçük olduğundan küçük bir kW hatası "
            "bile çok büyük yüzdeye dönüşür, bu yüzden kartlarda gösterilmez."
        )
        days_note = (
            f"Seçilebilir {len(days)} gün, önceki 48 saati ve hedef 24 saati doldurulmamış, tam ölçüm içeren "
            "günlerdir. Raporun günlük toplam satırındaki 229 gün ise yalnız hedefi doldurulmamış günleri sayar."
        )
        hard_note = "**Kolay/zor örnek:** Topluluk MAE'sinin en iyi/en kötü %10'luk diliminden rastgele seçilir."
    else:
        mape_note = (
            f"Seçili günde {mape_value}. Şebeke yükü hep yüksek olduğu için yanıltıcı değildir; "
            "iki veri setinde aynı kart kullanılsın diye burada verilir."
        )
        days_note = (
            f"Seçilebilir {len(days)} gün, önceki 48 saati ve hedef 24 saati doldurulmamış, tam ölçüm içeren "
            "günlerdir. Yaz saati geçişinden etkilenen günler dışarıda kalır."
        )
        hard_note = (
            "**Kolay/zor örnek:** Kolay örnek en iyi %10'luk dilimden seçilir. Zor örnek sabittir: 25 Aralık 2014 "
            "(Noel), testteki en yüksek MAE'li gün. Eğitimde yalnız iki Noel vardır."
        )
    with st.expander("Detaylar"):
        st.markdown(
            f"""
- **Varsayılan gün:** SN168'e göre iyileşmesi test günlerinin medyanına en yakın gün gösterilir. Böylece ilk görülen gün ne en iyi ne en kötü örnektir.
- **Test geneli ve seçili gün:** Üstteki test geneli cümlesi kayıtlı final sonuçtur, 3 seed ortalamasıdır ve tüm saatlik pencereleri kapsar. Kartlar ve dağılım yalnız 00:00'da başlayan günlük pencerelere ve seed 42 ağırlıklarına aittir; bu yüzden ikisi birebir aynı olmak zorunda değildir.
- **Test dönemi:** {_display_day(config['split']['test']['start'])} – {_display_day(config['split']['test']['end'])}. {days_note}
- **Topluluk modeli:** 0,8 × Seq2Seq + 0,2 × LSTM. Girdi 48 saat, çıktı 24 saat. Ağırlıklar kayıtlıdır; demoda eğitim yapılmaz.
- **SN168 (mevsimsel naif):** Bir önceki haftanın aynı saatlerini tahmin olarak kopyalar.
- **MAE:** Seçili günün 24 saati üzerinden ortalama mutlak hata (kW).
- **Günlük toplam tüketim hatası:** 100 × |Σtahmin − Σgerçek| / Σgerçek. Saatlik hatalar toplamda birbirini götürebilir; bu yüzden MAE ile birlikte okunmalıdır.
- **MAPE:** {mape_note}
- **İyileşme:** 100 × (SN168 MAE − Topluluk MAE) / SN168 MAE.
- {hard_note} Bu seçim yalnız görsel örnek içindir; model veya eşik seçimi test kümesinden yapılmaz.
"""
        )


def _household_source():
    """Hane için günleri, gün tahmin fonksiyonunu ve karşılaştırma tablosunu hazırlar."""
    config, data_path, models_dir, expected_features = demo_paths()
    frame = load_household_data(str(data_path))
    models = load_f13_models(str(models_dir), tuple(expected_features))
    days = valid_test_days(frame, config)
    mape_min_y = config["evaluation"]["mape_min_y"]
    table = household_comparison(frame, days, models, mape_min_y)
    return config, days, table, lambda day: predict_test_day(frame, day, models, mape_min_y)


def _eld_source():
    """Şebeke için aynı parçaları kayıtlı ağırlıkların tahminlerinden hazırlar."""
    frame, config, results, days, table = load_eld_forecasts()
    return config, days, table, lambda day: predict_eld_day(frame, config, results, day)


def render(dataset="household"):
    """Seçili veri seti için tahmin sayfasını çizer."""
    st.title("Talep Tahmini")
    dataset_label = DATASET_LABELS[dataset]
    digits = DIGITS[dataset]
    try:
        config, days, table, predict = _household_source() if dataset == "household" else _eld_source()
    except FileNotFoundError as error:
        if dataset == "eld":
            st.info(
                f"{dataset_label} için canlı tahmin dosyaları bu bilgisayarda yok ({error}). "
                "Kayıtlı test sonuçlarını Sonuçlar sayfasında görebilirsiniz."
            )
        else:
            st.error(f"Tahmin verisi veya modeli kullanılamıyor: {error}")
        return
    except (TypeError, ValueError, RuntimeError) as error:
        st.error(f"Tahmin verisi veya modeli kullanılamıyor: {error}")
        return

    day = _controls(dataset, days, table)
    # Test geneli her zaman seçili günün kartlarının üstünde görünür.
    note = st.session_state.get(f"forecast_note_{dataset}") or ""
    try:
        overall = overall_result_text(load_results_bundle(dataset)["forecast"], digits)
        st.markdown(f"{overall} &nbsp; <span style='color:#6B7280'>{note}</span>", unsafe_allow_html=True)
    except (FileNotFoundError, OSError, ValueError, KeyError) as error:
        st.error(f"Kayıtlı test geneli sonucu okunamadı: {error}")

    try:
        result = predict(day)
    except (KeyError, IndexError, TypeError, ValueError, RuntimeError) as error:
        st.error(f"{_display_day(day)} için tahmin üretilemedi: {error}")
        return
    target_times = result["target_times"]
    if len(target_times) != 24 or target_times[0].strftime("%Y-%m-%d") != day:
        st.error("Tahmin zaman ekseni seçilen 24 saatlik günle uyuşmuyor.")
        return

    # SN168'in günlük MAE'si, modelinkiyle aynı metrik fonksiyonundan gelir.
    baseline = calculate_metrics(
        result["actual_kw"], result["sn168_kw"], mape_min_y=config["evaluation"]["mape_min_y"]
    )
    _metric_cards(result, baseline["MAE"], digits)
    show_chart(
        _make_figure(result, day, dataset_label, digits),
        alt=f"{_display_day(day)} test günü için gerçek tüketim, Topluluk modeli ve SN168 tahmini",
    )
    _distribution_section(table, day, dataset_label)
    _details(dataset, config, days, result)
