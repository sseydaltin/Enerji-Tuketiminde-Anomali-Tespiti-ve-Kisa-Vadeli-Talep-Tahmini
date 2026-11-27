"""Anomali Tespiti sayfası: hane test döneminde skor, eşik ve olay görünümü."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.anomaly_services import (
    CAUGHT,
    EXPERIMENT_SPLIT,
    EXPERIMENTAL_METHODS,
    METHODS,
    MISSED,
    SEVERITY_TYPES,
    analyze_test_range,
    anomaly_demo_paths,
    checkpoint_signature,
    complete_test_days,
    event_outcomes,
    flagged_hours,
    load_anomaly_calibration,
)
from app.services import load_f13_models, load_household_data
from app.ui import (
    compact_table,
    kpi_cards,
    page_intro,
    ACTUAL_COLOR,
    ANOMALY_COLOR,
    ANOMALY_LABELS,
    BASELINE_COLOR,
    CARD_HEIGHT,
    DATASET_LABELS,
    FINAL_METHOD,
    LINE_ACTUAL,
    LINE_MODEL,
    METHOD_LABELS,
    number,
    show_chart,
)
from src.core.inject_anomalies import ANOMALY_TYPES
from src.core.anomaly_metrics import runs

NO_INJECTION = "none"
EVENT_TABLE_KEY = "anomaly-events"     # olay tablosundaki seçim (grafik o olaya gider)
WINDOW_DAYS = 7                        # varsayılan görünüm: seçili olayın çevresindeki 7 gün
DEFAULT_TYPE = "level_shift"
INJECTION_LABELS = {**ANOMALY_LABELS, NO_INJECTION: "Yok (gerçek ölçüm)"}
SCENARIO_LABELS = {"main": "Ana (σ tabanlı)", "hard": "Zor (katsayı tabanlı)"}
SEVERITY_STEPS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
SPLIT_LABELS = {"test": "test kümesi", EXPERIMENT_SPLIT: "doğrulama kümesi (val-hold)"}
EVENT_COLORS = {
    CAUGHT: "background-color: #DDF3E4; color: #14532D",
    MISSED: "background-color: #FBE0E0; color: #7F1D1D",
}


def _make_figure(result, anomaly_type, method, show_extra=False, x_range=None):
    """Tüketimi ve seçili yöntemin skorunu ortak zaman ekseninde çizer.

    Girdi: analyze_test_range çıktısı, enjekte tür (yoksa None), yöntem kodu; show_extra = ham skor ve
    doldurulmuş saatler de çizilsin mi (varsayılan gizli); x_range = (başlangıç, bitiş) görünüm aralığı (None: tüm dönem).
    Çıktı: 2 satırlı Plotly figürü: üstte kW, altta yumuşatılmış skor + eşik. Lejant en çok 4 öğe: gerçek ölçüm,
    enjekte anomali (bant), alarm verilen saat, eşik. Eşik yumuşatılmış skora uygulanır.
    """
    figure = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.42, 0.58],
    )
    index = result["real_kw"].index
    figure.add_trace(go.Scatter(
        x=index, y=result["real_kw"].where(~result["imputed"]), mode="lines",
        name="Gerçek ölçüm", line=LINE_ACTUAL,
    ), row=1, col=1)
    imputed = result["real_kw"].where(result["imputed"])
    if show_extra and imputed.notna().any():
        figure.add_trace(go.Scatter(
            x=index, y=imputed, mode="markers", name="Doldurulmuş saat (skor dışı)",
            marker={"color": BASELINE_COLOR, "size": 6},
        ), row=1, col=1)

    injected = result["seed"] is not None
    labelled = result["labels"]["is_anomaly"].astype(bool).to_numpy()
    if injected:
        # Yalnız enjekte saatler kırmızı çizilir; gerçek seri okunur kalsın diye gerisi boştur.
        figure.add_trace(go.Scatter(
            x=index, y=result["injected_kw"].where(labelled), mode="lines+markers",
            name=f"Sentetik {ANOMALY_LABELS[anomaly_type]}", showlegend=False,   # bant lejantta anlatılır
            line={"color": ANOMALY_COLOR, "width": 1.5}, marker={"color": ANOMALY_COLOR, "size": 4},
        ), row=1, col=1)

    scores = result["scores"][method]
    threshold = result["thresholds"][method]
    if show_extra:
        figure.add_trace(go.Scatter(
            x=index, y=result["raw_scores"][method], mode="lines", name="Ham skor", showlegend=False,
            line={"color": BASELINE_COLOR, "width": 1},
        ), row=2, col=1)
    figure.add_trace(go.Scatter(
        x=index, y=scores, mode="lines", name="Yumuşatılmış skor", line=LINE_MODEL, showlegend=False,
    ), row=2, col=1)
    # Eşik bir şekildir (lejantta görünmez); anlamı boş bir izle lejanda yazılır.
    figure.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines", name="Eşik", line={"color": ACTUAL_COLOR, "dash": "dash", "width": 1.5},
    ), row=2, col=1)
    figure.add_hline(
        y=threshold, line_dash="dash", line_color=ACTUAL_COLOR, line_width=1.5, row=2, col=1,
        annotation_text=f"Eşik {number(threshold)}",
        annotation_position="top right",
    )
    experimental = result["experimental"][method] if method in EXPERIMENTAL_METHODS else None
    marker_scores = scores
    if experimental:
        # Yumuşatılmış skorla tetiklenmeyen alarm, eşiğin altında görünmesin diye ham skorda çizilir.
        marker_scores = scores.where(scores >= experimental["t1"], result["raw_scores"][method])
        figure.add_hline(
            y=experimental["t2"], line_dash="dot", line_color=BASELINE_COLOR, line_width=1.5, row=2, col=1,
            annotation_text=f"Ham skor eşiği {number(experimental['t2'])}", annotation_position="top right",
        )
        if experimental["k"] is not None:
            figure.add_hline(
                y=experimental["floor_kw"], line_dash="dot", line_color=ACTUAL_COLOR, line_width=1, row=1, col=1,
                annotation_text=f"Taban {number(experimental['floor_kw'])} kW", annotation_position="top right",
            )
    flagged = flagged_hours(result, method)
    figure.add_trace(go.Scatter(
        x=index[flagged], y=marker_scores.to_numpy()[flagged], mode="markers", name="İşaretlenen saat",   # alarm
        marker={"color": ANOMALY_COLOR, "size": 7, "line": {"color": "white", "width": 1}},
    ), row=2, col=1)

    if injected:
        # Bantlar en sonda: Plotly, henüz izi olmayan alt grafiğe şekil eklemez.
        half_hour = pd.Timedelta(minutes=30)
        for start, end in runs(labelled):
            for row in (1, 2):  # bant hem tüketim hem skor satırında görünsün
                figure.add_vrect(
                    x0=index[start] - half_hour, x1=index[end - 1] + half_hour,
                    fillcolor=ANOMALY_COLOR, opacity=0.2, line_width=0, row=row, col=1,
                )
        # add_vrect lejanda görünmez; bandın anlamı boş bir izle lejanda yazılır.
        figure.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name="Enjekte edilen anomali",   # bant
            marker={"color": ANOMALY_COLOR, "symbol": "square", "size": 12, "opacity": 0.3},
        ))

    figure.update_yaxes(title_text="Aktif güç (kW)", hoverformat=".3f", row=1, col=1)
    figure.update_yaxes(title_text="Skor (birimsiz)", hoverformat=".3f", row=2, col=1)
    figure.update_xaxes(tickformat="%d.%m", hoverformat="%d.%m.%Y %H:%M")
    figure.update_xaxes(title_text="Zaman (gün.ay)", row=2, col=1)
    if x_range is not None:
        figure.update_xaxes(range=list(x_range))
    figure.update_layout(
        title=f"{DATASET_LABELS['household']}, {SPLIT_LABELS[result.get('split', 'test')]}: {METHOD_LABELS[method]} ve eşik",
        height=560,
        margin={"t": 95},
    )
    return figure


def _metric_cards(result, method, outcomes, false_alarms):
    """Üç KPI kartı; gerçek etiket yokken (enjeksiyon kapalı) başarı sayısı iddia edilmez ("—")."""
    injected = result["seed"] is not None
    blocks = len(runs(flagged_hours(result, method)))
    caught = f"{int((outcomes['Durum'] == CAUGHT).sum())} / {len(outcomes)}" if injected else "—"
    kpi_cards([
        ("Yakalanan olay", caught,
         "Enjekte olayların kaçı alarmla yakalandı." if injected else "Gerçek ölçümde etiket yok; enjeksiyon seçin."),
        ("Yanlış alarm", str(false_alarms) if injected else "—",
         f"Hiçbir olayla örtüşmeyen alarm (toplam {blocks} alarm)."),
        ("Eşik", number(result["thresholds"][method]), "Doğrulama kümesinde seçildi; testte değişmez."),
    ])


EVENT_COLUMNS = {   # olay ve alarm tablolarında kısa başlık → ipucu
    "Süre (saat)": ("Süre (sa)", "Olayın süresi (saat)."),
    "Ham skor (en yüksek)": ("Ham maks.", "Olay boyunca ham skorun en yüksek değeri."),
    "Yumuşatılmış skor (en yüksek)": ("Yumuş. maks.", "Olay boyunca yumuşatılmış skorun en yüksek değeri; eşik buna uygulanır."),
    "Olay bitince alarm": ("Geç alarm", "Alarm olay bittikten sonra mı geldi (katı tanımda kaçırılmış sayılır)."),
    "Enjekte olayla örtüşüyor": ("Olayla örtüşme", "Alarm bir enjekte olayla kesişiyor mu."),
}


def _event_table(outcomes):
    """Enjekte olay tablosu (yakalandı yeşil, kaçırıldı kırmızı); bir satır seçilince grafik o olaya gider."""
    def style(frame):
        return (frame.style.map(EVENT_COLORS.get, subset=["Durum"])
                .format({"Başlangıç": "{:%d.%m %H:%M}", "Yumuş. maks.": lambda value: number(value)},
                        na_rep="—").hide(axis="index"))
    # Dar ekranda yatay kaydırma olmasın: Bitiş (= başlangıç + süre) ve ham skor tepesi ("Tüm alarmlar") burada yok.
    shown = outcomes[["Başlangıç", "Tür", "Süre (saat)", "Yumuşatılmış skor (en yüksek)", "Durum", "Olay bitince alarm"]]
    compact_table(shown, EVENT_COLUMNS, styler=style, key=EVENT_TABLE_KEY, selectable=True)
    st.caption("Bitiş = başlangıç + süre. Ham skorun en yüksek değeri \"Tüm alarmlar\" tablosundadır.")


def _alarm_table(result, method):
    """Yöntemin eşik üstü alarm bloklarını kayıtlı olay tablosundan gösterir."""
    alarms = result["events"].loc[result["events"]["Yöntem"] == method].drop(columns="Yöntem")
    if alarms.empty:
        st.info("Seçilen aralıkta eşiği aşan geçerli alarm yok.")
        return
    alarms = alarms.rename(columns={
        "Sentetik olayla örtüşüyor": "Enjekte olayla örtüşüyor",
        "En yüksek skor": "Yumuşatılmış skor (en yüksek)",
    }).copy()
    raw = result["raw_scores"][method]
    alarms.insert(2, "Ham skor (en yüksek)", [
        raw.loc[start:end].max() for start, end in zip(alarms["Başlangıç"], alarms["Bitiş"])
    ])
    alarms["Enjekte olayla örtüşüyor"] = alarms["Enjekte olayla örtüşüyor"].map({True: "Evet", False: "Hayır"})
    def style(frame):
        return frame.style.format({"Başlangıç": "{:%d.%m %H:%M}", "Bitiş": "{:%d.%m %H:%M}",
                                   "Ham maks.": lambda value: number(value),
                                   "Yumuş. maks.": lambda value: number(value)}).hide(axis="index")
    # Eşik sütunu sabit olduğu için gösterilmez (KPI kartında); dar ekranda yatay kaydırma olmasın.
    compact_table(alarms.drop(columns=["Eşik"], errors="ignore"), EVENT_COLUMNS, styler=style)


def _controls(frame, config, calibration):
    """Üstte yöntem ve anomali türü; senaryo, şiddet ve tarihler "Gelişmiş ayarlar" açılır bölümünde.

    Çıktı: yöntem, tip, senaryo, şiddet, başlangıç, bitiş, küme, o kümenin günleri.
    Tarih aralığı varsayılan olarak tüm dönemdir: yakalanan/toplam kartı ve tablo, seçilmiş
    bir kesit değil, kopyanın bütün olaylarını gösterir. Deneysel yöntem test kümesinde
    çalıştırılmadığı için seçilince dönem val-hold olur; bu yüzden tarih anahtarları kümeye bağlıdır.
    """
    method_col, type_col = st.columns([3, 2])
    options = list(METHODS) + (list(EXPERIMENTAL_METHODS) if calibration.get("experimental") else [])
    method = method_col.selectbox(
        "Yöntem", options=options, index=options.index(FINAL_METHOD),
        format_func=METHOD_LABELS.get, key="anomaly-method",
    )
    split = EXPERIMENT_SPLIT if method in EXPERIMENTAL_METHODS else "test"
    days = complete_test_days(frame, config, split)
    anomaly_type = type_col.selectbox(
        "Anomali türü", options=list(INJECTION_LABELS),
        index=list(INJECTION_LABELS).index(DEFAULT_TYPE),
        format_func=INJECTION_LABELS.get, key="anomaly-type",
    )
    advanced = st.expander("Gelişmiş ayarlar: senaryo, şiddet, tarih aralığı")
    scenario_col, severity_col, start_col, end_col = advanced.columns(4)
    scenario = scenario_col.selectbox(
        "Senaryo", options=list(SCENARIO_LABELS), format_func=SCENARIO_LABELS.get,
        disabled=anomaly_type == NO_INJECTION, key="anomaly-scenario",
    )
    severity_active = anomaly_type in SEVERITY_TYPES
    severity = severity_col.select_slider(
        "Şiddet", options=SEVERITY_STEPS, value=1.0,
        format_func=lambda value: f"×{number(value, 2)}",
        disabled=not severity_active, key="anomaly-severity",
        help="×1,00 protokolün kendisidir. Düşüş ve gece örüntüsünün ayarlanacak büyüklüğü yoktur.",
    )
    start_day = start_col.date_input(
        "Başlangıç", value=days[0], min_value=days[0], max_value=days[-1],
        format="DD.MM.YYYY", key=f"anomaly-start-{split}",
    )
    end_day = end_col.date_input(
        "Bitiş", value=days[-1], min_value=days[0], max_value=days[-1],
        format="DD.MM.YYYY", key=f"anomaly-end-{split}",
    )
    # Kaydırıcı kapalıysa önceki değeri hesaba taşınmaz.
    return method, anomaly_type, scenario, severity if severity_active else 1.0, start_day, end_day, split, days


def _capture_table(tables_dir):
    """Kayıtlı tek tip yakalama tablosunu gösterir (python -m app.capture_table)."""
    path = tables_dir / "demo_capture_by_type.csv"
    if not path.is_file():
        st.caption("Tablo henüz üretilmemiş: python -m app.capture_table")
        return
    table = pd.read_csv(path)
    table["Yakalama oranı"] = [
        f"{number(mean)} ± {number(std)}"
        for mean, std in zip(table["Yakalama oranı (ort)"], table["Yakalama oranı (std)"])
    ]
    table["Yanlış alarm (ort)"] = table["Yanlış alarm (ort)"].map(lambda value: number(value, 1))
    compact_table(table[["Tip", "Yöntem", "Seed başına yakalanan / toplam", "Yakalama oranı", "Yanlış alarm (ort)"]],
                  {"Seed başına yakalanan / toplam": ("Yakalanan", "Seed başına yakalanan / toplam olay."),
                   "Yakalama oranı": ("Oran", "Yakalama oranı (5 seed ort. ± std)."),
                   "Yanlış alarm (ort)": ("Yanlış alarm", "Seed başına ortalama yanlış alarm.")})
    st.caption(
        "Her koşuda yalnız o tip enjekte edilir, tüm test dönemi, ana senaryo, şiddet ×1, 5 seed. "
        "Bu tablo demo denetimi içindir, final sonuç değildir; raporlanan tablolar dört tipi birlikte "
        "enjekte eder ve Sonuçlar sayfasındadır. Hiçbir seçim buna göre yapılmaz."
    )


def _experimental_note(method, settings):
    """Seçili "Geliştirilmiş" varyantını ve ödünleşimini Detaylar için açıklar."""
    if settings["k"] is None:
        behaviour = (
            "Kısa sıçramaları final yöntemden daha çok yakalar, karşılığında yanlış alarm artar. "
            "Ani düşüşleri görmez; taban kuralı yalnız diğer seçenektedir."
        )
    else:
        behaviour = (
            f"Ayrıca tüketim {number(settings['floor_kw'])} kW tabanının (eğitimin 0,5. yüzdeliği) altında "
            f"en az {settings['k']} saat kalırsa alarm verir; bu alarmlar skordan gelmez. Ham skor eşiği yüksek seçildiği için "
            "kısa sıçramalarda final yöntemden belirgin farkı yoktur. Enjekte düşüş hep 0,05 kW olduğundan taban kuralı "
            "o olayları neredeyse tanım gereği yakalar; gerçek düşüşlerde bu kadar iyi olacağı gösterilmemiştir."
        )
    return (
        f"- **{METHOD_LABELS[method]}:** Final sonuç değildir. Test sonrası denenen bir kuraldır: yumuşatılmış skor "
        f"{number(settings['t1'])} eşiğini veya ham skor {number(settings['t2'])} eşiğini geçerse alarm verir. {behaviour} "
        "Ayarları yalnız val-dev'de seçildi, val-hold'da bir kez ölçüldü. Test kümesinde çalıştırılmaz; bu yüzden bu seçenekte "
        "val-hold dönemi gösterilir ve testte daha iyi olduğu iddia edilemez. Karşılaştırma tablosu Model Bilgisi sayfasındadır.\n"
    )


def _details(result, method, inject, scenario, seed, severity):
    """Tanımları ve yumuşatma ödünleşimini açılır bölümde anlatır."""
    window = result["windows"][method]
    is_experimental = method in EXPERIMENTAL_METHODS
    experimental = _experimental_note(method, result["experimental"][method]) if is_experimental else ""
    threshold_rule = (
        "Bu kuralda ham skorun da ayrı, daha yüksek bir eşiği vardır; bu yüzden tek saatlik sıçrama doğrudan alarm verebilir."
        if is_experimental else
        "Bu yüzden ham skoru eşiği geçen bir olay kaçırılmış görünebilir."
    )
    caught_rule = "bu kural alarm verdiyse" if is_experimental else "yumuşatılmış skor eşiği geçtiyse"
    split_label = SPLIT_LABELS[result["split"]]
    injection_note = ""
    if inject:
        injection_note = (
            f"\n- **Bu enjeksiyon:** {SCENARIO_LABELS[scenario]}, şiddet ×{number(severity, 2)}, seed {seed}. "
            "Sentetik etiketler model veya eşik seçiminde kullanılmaz."
        )
        if severity != 1.0:
            injection_note += " Şiddet ×1 dışında sonuçlar protokol dışıdır, raporla karşılaştırılmaz."
    st.markdown(
        f"""
{experimental}- **Ham ve yumuşatılmış skor:** Ham skor her saatin kendi hatasıdır. Ana eşik, yalnız geçmişe bakan {window} saatlik ortalamaya uygulanır. {threshold_rule}
- **Yumuşatma ödünleşimi:** Ortalama, tek saatlik gürültüyü ve kısa sıçramaları bastırır. Seviye kayması gibi uzun olaylarda işe yarar, 1–2 saatlik ani yükselmelerde ise skoru geciktirip zayıflatır. Bu bilinen bir ödünleşimdir.
- **Olay bitince alarm:** Skor geçmişe baktığı için alarm olay bittikten sonra gelebilir. Bu durumda olay kaçırılmış, alarm yanlış sayılır. Bu katı tanım raporlanan olay metrikleriyle aynıdır.
- **Eşik:** {number(result['thresholds'][method])}. Doğrulama kümesinde seçilmiştir; test verisine göre değiştirilmez.
- **Yakalanan olay:** Enjekte olayın en az bir saatinde {caught_rule} olay yakalanmış sayılır.
- **Yanlış alarm:** Hiçbir enjekte saatle kesişmeyen alarm. Final yöntem (Seq2Seq artık skoru) enjeksiyonsuz testte de 8 alarm verir; bunlar gerçek ölçümdeki olağandışı saatler olabilir. Bu yüzden sayı üst sınır gibi okunmalıdır.
- **Skorlanan saat:** {int(result['valid'].sum())} / {len(result['valid'])}. Doldurulmuş saatler alarm dışı bırakılır.
- **Senaryolar:** Ana senaryoda ani yükselme +3–5 eğitim σ, seviye kayması ×2–2,5, düşüş yalnız 1 kW üstünde başlar. Zor senaryoda ani yükselme ×2–4, seviye kayması ×1,5.
- **Enjeksiyon kapsamı:** Protokoldeki gibi {split_label} kopyasının tamamına yapılır; tarih aralığı yalnız görünümü seçer. Tek koşudur, raporlanan genel başarı Sonuçlar sayfasındadır.{injection_note}
"""
    )


def render(dataset=None):
    """Hane test döneminin anomali sayfasını çizer; canlı analiz yalnız hanede vardır."""
    page_intro("Anomali Tespiti", "Hane test döneminde seçilen yöntemin anomali skoru ve eşiği gösterilir. İsterseniz gerçek "
               "ölçümün bir kopyasına sentetik anomali eklenir (orijinal veri değişmez); kartlar yakalanan olayları ve "
               "yanlış alarmları sayar.")
    dataset = dataset or st.session_state.get("dataset", "household")
    if dataset != "household":
        st.info(
            f"Canlı anomali analizi yalnız {DATASET_LABELS['household']} için hazırlanmıştır; aşağıda hane "
            f"gösterilir. {DATASET_LABELS[dataset]} için kayıtlı test sonuçları Sonuçlar sayfasındadır."
        )

    try:
        config, data_path, models_dir, tables_dir, features = anomaly_demo_paths()
        frame = load_household_data(str(data_path))
        calibration = load_anomaly_calibration(str(tables_dir))
        models = load_f13_models(str(models_dir), features)
    except (FileNotFoundError, OSError, TypeError, ValueError, KeyError) as error:
        st.error(f"Anomali demosu gerekli veri, model veya eşik dosyalarını yükleyemedi: {error}")
        return

    method, anomaly_type, scenario, severity, start_day, end_day, split, days = _controls(frame, config, calibration)
    if method in EXPERIMENTAL_METHODS:
        st.info(
            f"{METHOD_LABELS[method]} final sonuç değildir ve test kümesinde çalıştırılmaz. "
            "Bu yüzden burada doğrulama (val-hold) dönemi gösterilir; açıklama Detaylar'dadır."
        )
    if start_day not in days or end_day not in days or end_day < start_day:
        st.error("Başlangıç ve bitiş, bitiş sonra gelecek şekilde dönem içinde tam 24 saatlik günler olmalı.")
        return

    inject = anomaly_type != NO_INJECTION
    start = pd.Timestamp(start_day)
    end = pd.Timestamp(end_day) + pd.Timedelta(hours=23)
    seed = int(config["protocol"]["scenarios"][scenario]["seeds"][split][0])
    if inject:
        st.warning("Kırmızı bantlar sentetiktir, gerçek ölçüm değildir. Orijinal veri değişmez; enjeksiyon kopya üzerinde yapılır.")

    try:
        result = analyze_test_range(
            frame, config, start, end, inject,
            anomaly_type if inject else ANOMALY_TYPES[0],
            scenario, seed, calibration, checkpoint_signature(models_dir), models, severity, split,
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
        st.error(f"Anomali analizi tamamlanamadı: {error}")
        return

    outcomes, false_alarms = event_outcomes(result, method)
    _metric_cards(result, method, outcomes, false_alarms)
    full_col, extra_col = st.columns(2)
    full_period = full_col.toggle("Tüm dönemi göster", value=outcomes.empty, key="anomaly-full")
    show_extra = extra_col.toggle("Ham skoru göster", value=False, key="anomaly-extra",
                                  help="Ham (yumuşatılmamış) skoru ve doldurulmuş (skor dışı) saatleri de çizer.")
    x_range, focus_note = None, "Tüm dönem gösteriliyor."
    if not full_period and not outcomes.empty:
        selection = st.session_state.get(EVENT_TABLE_KEY)
        rows = list(selection.selection.rows) if selection is not None else []
        position = rows[0] if rows and rows[0] < len(outcomes) else 0
        event = outcomes.iloc[position]
        center = event["Başlangıç"] + (event["Bitiş"] - event["Başlangıç"]) / 2
        half = pd.Timedelta(days=WINDOW_DAYS / 2)
        x_range = (center - half, center + half)
        focus_note = (f"Görünüm: {position + 1}. olayın ({event['Başlangıç']:%d.%m.%Y}) çevresindeki {WINDOW_DAYS} gün. "
                      "Başka bir olaya gitmek için aşağıdaki tablodan bir satır seçin.")
    st.caption(focus_note)
    show_chart(
        _make_figure(result, anomaly_type if inject else None, method, show_extra=show_extra, x_range=x_range),
        alt=f"{METHOD_LABELS[method]} yumuşatılmış skoru, eşik çizgisi, enjekte anomaliler ve alarm verilen saatler",
    )

    if inject:
        st.subheader("Enjekte edilen olaylar")
        if outcomes.empty:
            st.info("Seçilen tarih aralığında enjekte olay yok. Tarih aralığını genişletin.")
        else:
            _event_table(outcomes)
        with st.expander("Tüm alarmlar"):
            _alarm_table(result, method)
    else:
        st.subheader("Alarmlar")
        st.caption("Enjeksiyon kapalı: gerçek anomali etiketi olmadığından alarmlar doğrulanmış anomali sayılmamalıdır.")
        _alarm_table(result, method)

    with st.expander("Detaylar"):
        _details(result, method, inject, scenario, seed, severity)
    with st.expander("Tip bazlı yakalama oranı (5 seed)"):
        _capture_table(tables_dir)
