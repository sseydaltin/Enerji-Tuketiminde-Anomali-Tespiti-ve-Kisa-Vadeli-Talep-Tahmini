"""Model Bilgisi sayfası: kaynak dosyalardan doğrulanmış mimari ve deney protokolü.

İç deney kodları (F13, A3 ...) yalnız bu sayfada, parantez içinde görünür.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.model_info_services import load_model_info
from app.results_services import load_results_bundle
from app.ui import ACTUAL_COLOR, DATASET_LABELS, MODEL_COLOR, SECOND_COLOR, number, show_chart

SPLIT_LABELS = {"train": "Eğitim", "val": "Doğrulama", "test": "Test"}
# Kırmızı anomalilere ayrıldığı için test kümesi turuncudur.
SPLIT_COLORS = {"train": MODEL_COLOR, "val": SECOND_COLOR, "test": "#E0872B"}
LOSS_LABELS = {"huber": "Huber", "l1": "L1 (MAE)"}


@st.cache_data(ttl=600, show_spinner=False)
def _cached_model_info():
    """Kaynak metadata özetini kısa süreli önbellekten döndürür."""
    return load_model_info()


def _seq2seq_figure(lookback, horizon, n_features, hidden, layers):
    """Seq2Seq mimarisini blok diyagramla gösterir.

    Girdi: checkpoint metadata'sından pencere, ufuk, öznitelik, gizli boyut ve katman.
    Çıktı: encoder → gizli durum → decoder (+ takvim) → çıkış akışını çizen figür.
    Neden: Kutulardaki boyutlar elle yazılmaz; ağırlık dosyasındaki değerlerden gelir.
    """
    calendar = n_features - 1
    boxes = {
        "history": (0, 150, 150, 240, f"Geçmiş {lookback} saat", f"[{lookback}, {n_features}]", False),
        "encoder": (170, 320, 150, 240, "Encoder LSTM", f"{layers} katman, {hidden} birim", True),
        "state": (340, 490, 150, 240, "Gizli durum (h, c)", f"[{layers}, {hidden}]", True),
        "decoder": (510, 660, 150, 240, "Decoder LSTM", f"{layers} katman, {hidden} birim", True),
        "calendar": (510, 660, 10, 90, "Gelecek takvim", f"[{horizon}, {calendar}]", False),
        "output": (680, 830, 150, 240, "Doğrusal çıkış", f"Saat başına 1 değer", True),
        "forecast": (850, 1000, 150, 240, f"{horizon} saatlik tahmin", f"[{horizon}]", False),
    }
    figure = go.Figure()
    for x0, x1, y0, y1, title, detail, is_model in boxes.values():
        figure.add_shape(
            type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
            line={"color": MODEL_COLOR if is_model else ACTUAL_COLOR, "width": 1.5},
            fillcolor="#E3EBF7" if is_model else "#F4F6F9",
        )
        figure.add_annotation(
            x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=f"<b>{title}</b><br>{detail}",
            showarrow=False, font={"size": 12},
        )
    for start, end in (
        ("history", "encoder"), ("encoder", "state"), ("state", "decoder"),
        ("decoder", "output"), ("output", "forecast"),
    ):
        y = 195
        figure.add_annotation(
            x=boxes[end][0], y=y, ax=boxes[start][1], ay=y, xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=2, arrowwidth=1.7, arrowcolor=ACTUAL_COLOR,
        )
    figure.add_annotation(
        x=585, y=150, ax=585, ay=90, xref="x", yref="y", axref="x", ayref="y",
        text="", showarrow=True, arrowhead=2, arrowwidth=1.7, arrowcolor=ACTUAL_COLOR,
    )
    figure.update_layout(
        height=270, margin={"l": 5, "r": 5, "t": 5, "b": 5}, hovermode=False,
        xaxis={"visible": False, "range": [-5, 1005], "fixedrange": True},
        yaxis={"visible": False, "range": [0, 250], "fixedrange": True},
    )
    return figure


def _hyperparameter_table(checkpoints):
    """İki tahmin modelinin ağırlık dosyasındaki ayarlarını yan yana tablolaştırır."""
    rows = (
        ("Gizli boyut", "hidden", str),
        ("Katman sayısı", "layers", str),
        ("Dropout", "dropout", lambda value: number(value, 3)),
        ("Öğrenme oranı", "lr", lambda value: number(value, 5)),
        ("Weight decay", "weight_decay", lambda value: f"{value:.2e}".replace(".", ",")),
        ("Batch boyutu", "batch_size", str),
        ("Kayıp fonksiyonu", "loss", lambda value: LOSS_LABELS.get(value, value)),
        ("Girdi penceresi (saat)", "lookback", str),
        ("Tahmin ufku (saat)", "horizon", str),
        ("Azami epoch", "max_epochs", str),
        ("Erken durdurma sabrı (epoch)", "early_stopping_patience", str),
        ("Gradyan kırpma", "grad_clip", lambda value: number(value, 1)),
    )
    table = {"Hiperparametre": [label for label, _, _ in rows]}
    for name, column in (("F10-opt", "Seq2Seq (F10-opt)"), ("F5-opt", "LSTM (F5-opt)")):
        if checkpoints[name]["available"]:
            forecast = checkpoints[name]["forecast"]
            table[column] = [show(forecast[key]) for _, key, show in rows]
    return pd.DataFrame(table)


def _split_timeline(household, eld):
    """Kronolojik bölmeyi iki veri seti için renkli zaman çizelgesi olarak çizer.

    Neden: Kalın çizgi parçaları tarih ekseninde sade bir Gantt görünümü verir.
    Bitiş tarihi gün dahil olduğu için çizimde bir gün sonrasına uzatılır.
    """
    figure = go.Figure()
    shown = set()
    for label, config in (
        (DATASET_LABELS["eld"], eld),
        (DATASET_LABELS["household"], household),
    ):
        for split, name in SPLIT_LABELS.items():
            settings = config["split"][split]
            end = pd.Timestamp(settings["end"]) + pd.Timedelta(days=1)
            figure.add_trace(go.Scatter(
                x=[pd.Timestamp(settings["start"]), end],
                y=[label, label],
                mode="lines",
                name=name,
                legendgroup=split,
                showlegend=split not in shown,
                line={"color": SPLIT_COLORS[split], "width": 30},
                hovertemplate=f"{label}<br>{name}: {settings['start']} – {settings['end']}<extra></extra>",
            ))
            shown.add(split)
    figure.update_xaxes(title_text="Yıl", tickformat="%Y", dtick="M12", showgrid=True, gridcolor="#EEF0F3")
    figure.update_yaxes(showgrid=False, rangemode="normal")
    figure.update_layout(
        height=280, hovermode="closest", margin={"l": 150, "r": 20, "t": 80, "b": 45},
        title="Kronolojik veri bölmesi (eğitim, doğrulama, test)",
    )
    return figure


def _flow_figure():
    """Eğitim, doğrulama ve test akışlarını ayrı, geniş kutularla gösterir."""
    figure = go.Figure()

    lanes = (
        (425, 705, "EĞİTİM: ölçekleme, model eğitimi ve OOF kalibrasyonu", "#f3f7fb"),
        (225, 395, "DOĞRULAMA: ağırlık, model ve anomali ayarı seçimi", "#f6f8fc"),
        (35, 185, "FİNAL TEST: dondurulmuş seçimlerle tek değerlendirme", "#f3f7fb"),
    )
    for bottom, top, label, color in lanes:
        figure.add_shape(
            type="rect", x0=10, x1=990, y0=bottom, y1=top,
            line={"color": "#d7e0ea", "width": 1},
            fillcolor=color, layer="below",
        )
        figure.add_annotation(
            x=28, y=top - 18, text=f"<b>{label}</b>",
            showarrow=False, xanchor="left",
            font={"size": 13, "color": "#34445a"},
        )

    nodes = {
        "hourly": (45, 315, 860, 940, "Saatlik veri", "İşlenmiş seri<br>hedef + takvim özellikleri"),
        "split": (625, 955, 860, 940, "Kronolojik bölme", "Tarih aralıkları config'ten<br>rastgele bölme yok"),
        "train_scaler": (25, 210, 615, 670, "Eğitim verisi", "Saatlik eğitim bölümü"),
        "scaler_fit": (250, 430, 615, 670, "Ölçekleyici fit", "Yalnız eğitim"),
        "scaler_stats": (470, 650, 615, 670, "Ölçek istatistikleri", "Doğrulama/teste fit yok"),
        "train_model": (25, 210, 540, 595, "Eğitim verisi", "Saatlik eğitim bölümü"),
        "train_windows": (250, 430, 540, 595, "Eğitim pencereleri", "Yalnız eğitim aralığı"),
        "fit_weights": (500, 685, 540, 595, "Model ağırlıkları", "Eğitim pencereleriyle<br>gradyan güncellemesi"),
        "candidate": (760, 980, 540, 595, "Aday<br>ağırlık", "Epoch seçimi: doğrulama MAE"),
        "train_folds": (25, 210, 465, 520, "Eğitim verisi", "Yalnız eğitim dönemi"),
        "fold_models": (250, 410, 465, 520, "Zamansal CV", "3 ardışık kat<br>yalnız geçmişle fit"),
        "oof_residual": (450, 620, 465, 520, "OOF artıklar", "Modelin görmediği<br>eğitim saatleri"),
        "oof_std": (660, 825, 465, 520, "A3 std tablosu", "Saat × haftanın günü"),
        "val_inputs": (25, 245, 270, 355, "Doğrulama girdileri", "Doğrulama serisi + eğitim ölçeği,<br>ağırlık ve OOF std"),
        "val_scores": (300, 505, 270, 355, "Doğrulama skorları", "Tahmin MAE; A0/A3/A5<br>val-dev ana senaryo"),
        "val_select": (520, 750, 270, 355, "Doğrulama seçimi", "Ağırlık/epoch: doğrulama MAE<br>eşik + yumuşatma: val-dev"),
        "frozen": (780, 975, 270, 355, "Kararları<br>dondur", "Model, eşik, pencere"),
        "test_inputs": (25, 285, 75, 150, "Test girdileri", "Test serisi + eğitim ölçeği,<br>ağırlık ve OOF std"),
        "test_apply": (350, 650, 75, 150, "Sabitleri uygula", "Doğrulamada seçilen<br>model ve eşikler"),
        "test_report": (700, 975, 75, 150, "Final metrikler", "Sonuçları raporla;<br>seçim/ayar yapma"),
    }

    def arrow(start, end):
        """Kutu kenarları arasında tek yönlü, gerçek süreç bağlantısı çizer."""
        x1, y1 = start
        x2, y2 = end
        figure.add_annotation(
            x=x2, y=y2, ax=x1, ay=y1,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=2, arrowsize=1,
            arrowwidth=1.7, arrowcolor="#536b86",
            standoff=8, startstandoff=8,
        )

    def east(name):
        """Kutunun sağ kenar ortası."""
        x0, x1, y0, y1, *_ = nodes[name]
        return x1, (y0 + y1) / 2

    def west(name):
        """Kutunun sol kenar ortası."""
        x0, x1, y0, y1, *_ = nodes[name]
        return x0, (y0 + y1) / 2

    def north(name):
        """Kutunun üst kenar ortası."""
        x0, x1, y0, y1, *_ = nodes[name]
        return (x0 + x1) / 2, y1

    def south(name):
        """Kutunun alt kenar ortası."""
        x0, x1, y0, y1, *_ = nodes[name]
        return (x0 + x1) / 2, y0

    edges = (
        ("hourly", "split"),
        ("train_scaler", "scaler_fit"),
        ("scaler_fit", "scaler_stats"),
        ("train_model", "train_windows"),
        ("train_windows", "fit_weights"),
        ("fit_weights", "candidate"),
        ("train_folds", "fold_models"),
        ("fold_models", "oof_residual"),
        ("oof_residual", "oof_std"),
        ("val_inputs", "val_scores"),
        ("val_scores", "val_select"),
        ("val_select", "frozen"),
        ("test_inputs", "test_apply"),
        ("test_apply", "test_report"),
    )
    cross_edges = (
        ("candidate", "frozen"),
        ("frozen", "test_apply"),
    )
    for start, end in edges:
        arrow(east(start), west(end))

    arrow(south("candidate"), north("frozen"))
    arrow(south("frozen"), north("test_apply"))

    for x0, x1, y0, y1, title, detail in nodes.values():
        figure.add_shape(
            type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
            line={"color": MODEL_COLOR, "width": 1.5},
            fillcolor="#ffffff",
            layer="above",
        )
        figure.add_annotation(
            x=(x0 + x1) / 2,
            y=(y0 + y1) / 2,
            text=f"<b>{title}</b><br>{detail}",
            showarrow=False,
            align="center",
            font={"size": 12, "color": "#172b4d"},
        )

    figure.update_layout(
        height=920,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis={"visible": False, "range": [0, 1000], "fixedrange": True},
        yaxis={"visible": False, "range": [0, 1000], "fixedrange": True},
        hovermode=False,
        meta={
            "flow_nodes": list(nodes),
            "flow_edges": [
                [start, end] for start, end in (
                    edges + cross_edges
                )
            ],
        },
    )
    return figure


def _model_rows(info):
    """Kaydedilmiş tahmin modeli metadata'sını mimari tablosuna çevirir."""
    rows = []
    for name, label in (("F5-opt", "LSTM (F5-opt)"), ("F10-opt", "Seq2Seq (F10-opt)")):
        model = info["checkpoints"][name]
        if not model["available"]:
            rows.append({"Model": label, "Durum": "Ağırlık dosyası bulunamadı"})
            continue
        forecast = model["forecast"]
        if name == "F5-opt":
            architecture = (
                f"LSTM ({forecast['layers']} katman); son gizli durumdan "
                "24 çıkışlı doğrusal katman"
            )
            input_shape = f"[48 saat, {len(model['features'])} özellik]"
        else:
            architecture = (
                f"Encoder–decoder LSTM ({forecast['layers']} katman); "
                "decoder her hedef saatin takvimini alır"
            )
            input_shape = (
                f"encoder [48, {len(model['features'])}], "
                f"decoder [24, {len(model['features']) - 1}]"
            )
        rows.append({
            "Model": label,
            "Mimari": architecture,
            "Girdi / ufuk": f"{input_shape} → 24 saat",
            "Özellikler": ", ".join(model["features"]),
        })
    return rows


def _split_rows(config):
    """Config'teki kapsayıcı başlangıç ve bitiş tarihlerini tabloya çevirir."""
    return [
        {
            "Küme": SPLIT_LABELS[name],
            "Başlangıç": pd.Timestamp(settings["start"]).strftime("%d.%m.%Y"),
            "Bitiş": pd.Timestamp(settings["end"]).strftime("%d.%m.%Y"),
        }
        for name, settings in config["split"].items()
        if name in SPLIT_LABELS
    ]


def _threshold_table(frame):
    """Doğrulama seçim dosyasındaki yöntem, pencere ve eşikleri biçimlendirir."""
    table = frame.rename(columns={
        "method": "Yöntem (kod)",
        "smooth": "Geçmişe bakan yumuşatma (saat)",
        "threshold": "Doğrulama eşiği",
    })[["Yöntem (kod)", "Geçmişe bakan yumuşatma (saat)", "Doğrulama eşiği"]].copy()
    table["Doğrulama eşiği"] = table["Doğrulama eşiği"].map(number)
    return table


def _models_section(info):
    """Mimari diyagramı, hiperparametre ve model tablolarını gösterir."""
    checkpoints = info["checkpoints"]
    f5 = checkpoints["F5-opt"]
    f10 = checkpoints["F10-opt"]
    st.subheader("Seq2Seq mimarisi (F10-opt)")
    if f10["available"]:
        forecast = f10["forecast"]
        show_chart(
            _seq2seq_figure(
                forecast["lookback"], forecast["horizon"], len(f10["features"]),
                forecast["hidden"], forecast["layers"],
            ),
            alt="Encoder, gizli durum, takvim girdili decoder ve çıkış katmanından oluşan Seq2Seq blok diyagramı",
        )
        st.caption(
            "Encoder geçmişi özetler; decoder bu özetten başlar ve her hedef saatin önceden "
            "bilinen takvimini alır. Önceki tahminini girdi almadığı için hata birikmez."
        )
    else:
        st.info("Seq2Seq ağırlık dosyası bulunamadığı için mimari diyagramı çizilemedi.")

    if f5["available"] and f10["available"]:
        if f5["features"] != f10["features"]:
            st.error("LSTM ve Seq2Seq özellik listeleri uyuşmuyor; Topluluk üyeleri aynı girişi görmüyor.")
        elif f5["forecast"]["lookback"] != f10["forecast"]["lookback"]:
            st.error("LSTM ve Seq2Seq girdi pencereleri uyuşmuyor; Topluluk modeli uyumsuz.")
        else:
            st.info(
                "**Topluluk modeli (F13):** 0,8 × Seq2Seq (F10-opt) + 0,2 × LSTM (F5-opt). Ağırlıklar "
                f"kayıtlı final protokolündendir. Modellerin gerçek girdi penceresi "
                f"{f10['forecast']['lookback']} saat, ufku {f10['forecast']['horizon']} saattir; "
                "config.yaml'daki genel 168 saat varsayılanıyla karıştırılmamalıdır."
            )

    st.subheader("Hiperparametreler")
    st.table(_hyperparameter_table(checkpoints).style.hide(axis="index"))
    st.caption(
        "Değerler ağırlık dosyalarından okunur ve Optuna en iyi deneme CSV'leriyle doğrulanır "
        "(optuna_lstm_best.csv, optuna_seq2seq_best.csv)."
    )
    with st.expander("Mimari ve özellik ayrıntıları"):
        st.dataframe(_model_rows(info), hide_index=True, alt="Tahmin modeli mimari, pencere ve özellik tablosu")
        for name, label in (("F5-opt", "LSTM (F5-opt)"), ("F10-opt", "Seq2Seq (F10-opt)")):
            checkpoint = checkpoints[name]
            if checkpoint["available"]:
                st.write(f"**{label} özellik ortalaması:** {np.array2string(np.asarray(checkpoint['feature_mean']), precision=5)}")
                st.write(f"**{label} özellik ölçeği:** {np.array2string(np.asarray(checkpoint['feature_scale']), precision=5)}")
                st.write(f"**{label} hedef ortalama / ölçek:** {checkpoint['target_mean'][0]:.6g} / {checkpoint['target_scale'][0]:.6g}")

    xgb = info["household"]["forecast"]["xgboost"]
    xgb_lags = [lag for lag in xgb["lags"] if lag <= 336]
    st.subheader("Karşılaştırma modelleri")
    if info["xgb_artifacts"]:
        st.write(f"XGBoost model dosyaları bulundu: {', '.join(info['xgb_artifacts'])}")
    else:
        st.info(
            "Bilgi: XGBoost (F12-L336) ağırlık dosyası yerelde yok. Bu sayfa tahmin üretmez ve "
            "modeli yeniden eğitmez; aşağıdaki değerler config ayarlarıdır."
        )
    st.write(
        f"**XGBoost (F12-L336):** Girdi penceresi 336 saat, hedef gecikmeleri {xgb_lags}, "
        f"ağaç sayısı üst sınırı {xgb['n_estimators']}, öğrenme oranı {number(xgb['learning_rate'], 2)}, "
        f"azami derinlik {xgb['max_depth']}, subsample {number(xgb['subsample'], 1)}, colsample_bytree "
        f"{number(xgb['colsample_bytree'], 1)}, erken durdurma {xgb['early_stopping_rounds']} tur (doğrulama MAE)."
    )
    st.write(
        "**Mevsimsel naif (F2, SN168):** Tahmin ufkuna önceki haftanın aynı saatlerini kopyalar; "
        "eğitilen ağırlığı yoktur."
    )


def _dataset_section(info, dataset):
    """Seçili veri setinin hedefini, bölme tarihlerini ve dosya durumunu gösterir."""
    availability = info["availability"]
    f10 = info["checkpoints"]["F10-opt"]
    st.subheader(f"Veri seti: {DATASET_LABELS[dataset]}")
    if dataset == "household":
        config = info["household"]
        st.write(
            f"Hedef **{config['preprocess']['target']}** (kW); dakikalık hane ölçümleri saatlik "
            "ortalamaya indirgenir. Özellik seti B: hedef + "
            f"{len(f10['features']) - 1 if f10['available'] else 8} takvim özelliği."
        )
        raw, processed = availability["household_raw"], availability["household_processed"]
    else:
        config = info["eld"]
        st.write(
            f"Hedef **{config['preprocess']['target']}** (kW); 160 seçilmiş müşterinin 15 dakikalık "
            "toplam yükü saatlik ortalamaya indirgenir."
        )
        raw, processed = availability["eld_raw"], availability["eld_processed"]
    st.dataframe(_split_rows(config), hide_index=True, alt="Seçili veri setinin kronolojik bölme tarihleri")
    st.caption(
        f"Bu bilgisayarda ham veri: {'var' if raw else 'yok'}; "
        f"işlenmiş saatlik dosya: {'var' if processed else 'yok'}."
    )
    if dataset == "eld":
        for name, label in (("ELD-F5-opt", "LSTM (ELD-F5-opt)"), ("ELD-F10-opt", "Seq2Seq (ELD-F10-opt)")):
            status = "var" if info["checkpoints"][name]["available"] else "yok"
            st.caption(f"{label} ağırlık dosyası: {status}.")


def _anomaly_section(info, dataset):
    """Anomali skorlarının tanımını, eşikleri ve kaynak notunu gösterir."""
    household = info["household"]
    st.subheader("Anomali yöntemleri")
    st.markdown(
        rf"""
**Hareketli z-skoru (A0):** aynı saatlik geçmişe göre sapma.

`A0[t] = |y[t] − mean(y[t−24], …, y[t−{household['anomaly']['zscore_window']}])| / max(std(geçmiş), {household['anomaly']['zscore_min_std']} kW)`.
Önceki yedi günün aynı saati kullanılır; hedef saat referans ortalamasına katılmaz.

**Seq2Seq artık skoru (A3-s2s-norm, final):** 1 saat sonrası tahmin hatası, olağan hata büyüklüğüne bölünür.

`A3[t] = |y[t] − ŷ_F10,h=1[t]| / std_OOF[saat(t), haftanın_günü(t)]`.
Payda, eğitimin son 12 ayındaki üç zamansal CV katında üretilen out-of-fold artıkların standart sapmasıdır.

**Birleşik skor (A5-mean-z):** A0 ile Topluluk artığının ortak ölçekte ortalaması.

`A5[t] = (z_val(A0[t]) + z_val(A3-ens-norm[t])) / 2`.
A3-ens-norm, `0,8 × F10-opt + 0,2 × F5-opt` tahmin artığını kullanır. Referans, enjeksiyonsuz doğrulama skor dağılımıdır.
"""
    )
    st.dataframe(
        _threshold_table(info["thresholds"][dataset]),
        hide_index=True,
        alt=f"{DATASET_LABELS[dataset]} için doğrulama seçim dosyasındaki anomali eşikleri",
    )
    source = (
        "results/tables/anomaly_protocol_selection.csv (val-dev, ana senaryo)"
        if dataset == "household" else "results/eld/tables/anomaly_selection.csv (doğrulama, ana senaryo)"
    )
    st.caption(f"Eşikler doğrulama seçimindendir, testte ayarlanmaz. Kaynak: {source}.")
    with st.expander("Enjeksiyon, metrikler ve kaynak notu"):
        st.write(
            "Ana senaryoda enjeksiyon, orijinal serinin kopyasında ani yükselme (+3–5 eğitim σ), "
            "ani düşüş (~0,05 kW), seviye kayması (×2–2,5) ve 00:00–05:00 gece örüntüsü oluşturur."
        )
        st.markdown(
            "**Anomali metrikleri:** PR-AUC, etiketli saatlerin sürekli skor sıralamasındaki "
            "kesinlik–duyarlılık başarısını özetler. Olay recall, her etiketli olayın en az bir "
            "saatinin yakalanma oranıdır; olay precision, gerçek anomaliyle kesişen alarm "
            "bloklarının tüm alarm bloklarına oranıdır; olay F1 bu ikisinin harmonik ortalamasıdır."
        )
        st.warning(
            "Kaynak notu: docs/final_test_protokolu.md §5'te A0, revizyon öncesi seçim "
            "olarak geçer; §5a revizyonuyla final yöntem A3-s2s-norm olarak değiştirilmiştir "
            "(README ve seçim CSV'siyle uyumlu). Final A5 referansı için "
            "docs/final_test_protokolu.md test komutu ve src/anomaly_protocol.py içindeki "
            "açık reference_split='val_dev' çağrısı, ayrıca results/experiments.csv notu "
            "val-dev kullanımını destekler. src/evaluate_anomaly.py içindeki genel "
            "fit_combination varsayılanı tam doğrulama kümesidir; final protokol bu varsayılanı "
            "açıkça geçersiz kılar. Kaydedilmiş test tablolarında reference_split alanı ve bağımsız "
            "çalıştırma günlüğü bulunmadığından kanıt final kod yolu ve deney notuna dayanır. "
            "Eşikler burada CSV'den aynen okunur ve yeniden seçilmez."
        )


def _experiment_section(info):
    """Test sonrası A6 denemesini kayıtlı tablolardan gösterir (final sonuç değildir)."""
    tables_dir = Path(info["root"]) / "results/tables"
    selection_path = tables_dir / "a6_selection.csv"
    summary_path = tables_dir / "a6_val_summary.csv"
    if not selection_path.is_file() or not summary_path.is_file():
        return
    with st.expander("Test sonrası geliştirme: Geliştirilmiş (A6-dual, A6-dual-floor), final sonuç değildir"):
        selection = pd.read_csv(selection_path)
        demo = selection.loc[selection["demo"], "method"].iloc[0]
        st.write(
            f"Demodaki \"Geliştirilmiş (deneysel)\" seçeneği {demo} varyantıdır. Ayarlar yalnız val-dev'de seçildi, "
            "val-hold'da bir kez ölçüldü; test kümesi kullanılmadı. Ayrıntı: docs/final_sonuclar.md."
        )
        summary = pd.read_csv(summary_path)
        table = pd.DataFrame({
            "Küme": summary["split"],
            "Yöntem (kod)": summary["method"],
            "Olay F1": [f"{number(m)} ± {number(d)}" for m, d in zip(summary["event_f1_mean"], summary["event_f1_std"])],
            "Olay recall": summary["event_recall_mean"].map(number),
            "Olay precision": summary["event_precision_mean"].map(number),
            "Yanlış alarm (kopya başına)": summary["false_alarms_mean"].map(lambda value: number(value, 1)),
        })
        st.table(table.style.hide(axis="index"))


def _protocol_section(info):
    """Deney protokolünü, sınırlılıkları ve belge alıntılarını açılır bölümlerde gösterir."""
    household = info["household"]
    availability = info["availability"]
    with st.expander("Sistem akışı"):
        show_chart(
            _flow_figure(),
            alt="Kronolojik veri akışını, eğitim tabanlı ölçeklemeyi, doğrulama seçimlerini ve test değerlendirmesini gösteren şema",
        )
    with st.expander("Deney protokolü ve metrikler"):
        st.markdown(
            f"""
- Ölçekleyici yalnız eğitim kümesinde fit edilir; doğrulama/test yalnız dönüştürülür.
- Tahmin hiperparametreleri Optuna'da eğitimin son 12 ayındaki 3 zamansal CV katında seçilir; seçilen ayar doğrulama ile izlenir.
- Anomali skoru normalizasyonu eğitim OOF artıklarından; eşik ve yumuşatma yalnız doğrulamadan gelir.
- Test kümesi model/eşik seçimi için kullanılmaz; final sabitlenmiş seçimle raporlanır.
- Final tahmin ağırlıklarında erken durdurma tam doğrulama kümesini kullandığından val-hold bu ağırlıklar için tamamen bağımsız değildir; final test seçimden ayrıdır.
- Tahmin seed'leri config'te `{household['multi_seeds']}`; final test tahminleri 3 seed, anomali test raporları 5 seed ile ortalama ± örnek standart sapması sunar.
- **MAPE (MAPE\\*):** yalnız gerçek hedef `y > {household['evaluation']['mape_min_y']} kW` iken `mean(|(y−ŷ)/y|) × 100`.
- **MAE:** ortalama mutlak hata (kW). **RMSE:** karekök ortalama karesel hata (kW), büyük hatalara daha duyarlı.
- **R²:** `1 − SSE/SST`; hedef varyansına göre açıklanan uyumu ölçer, boyutsuzdur.
- Hane ve Şebeke arasında mutlak MAE kıyaslanmaz; bunun yerine MAPE, R² ve aynı veri setindeki SN168'e göre MAE kazancı kullanılır.
"""
        )
        st.caption(
            "Kaynaklar: docs/final_test_protokolu.md, docs/eld_protokol.md, "
            "src/data/windows.py ve src/utils/metrics.py."
        )
    with st.expander("Teknik sınırlılıklar"):
        eld_state = (
            "Talep Tahmini sayfası Şebeke için kayıtlı ağırlıklarla canlı tahmin üretir (seed 42). Canlı anomali analizi yalnız hane için yazılmıştır; Şebeke anomali sonuçları Sonuçlar sayfasındadır."
            if availability["eld_processed"]
            else "Şebeke işlenmiş verisi bu bilgisayarda yok; Şebeke yalnız kayıtlı sonuçlarla gösterilir."
        )
        st.markdown(
            f"""
- **Şebeke canlı analizi:** {eld_state}
- **XGBoost canlı tahmini:** Eğitilmiş model dosyası yoksa sayfa yalnız kayıtlı deneyin config/protokol bilgisini gösterir.
- **macOS/XGBoost:** XGBoost eğitim testinin macOS'ta segmentation fault oluşturduğu bildirilmiştir. Bu sayfa testi/eğitimi çalıştırmaz; kök neden burada teşhis edilmemiştir.
- **Zaman birimi:** `src/data/preprocess.py` içindeki `DatetimeIndex.asi8` mesafesi `Timedelta.value` ile karşılaştırılıyor. İç gösterim birimleri aynı olmayabileceğinden bu bilinen çözülmemiş uyumsuzluktur. Bu sayfa ön işleme kodunu çağırmaz.
"""
        )
    with st.expander("Protokol ve rapor alıntıları"):
        try:
            bundle = load_results_bundle("household")
        except (FileNotFoundError, OSError, ValueError, KeyError) as error:
            st.error(f"Belge alıntıları yüklenemedi: {error}")
            return
        st.markdown(bundle["final_doc_excerpt"])
        st.markdown(bundle["comparison_doc_excerpt"])
        st.caption("Kaynaklar: docs/final_sonuclar.md, docs/hane_vs_sebeke.md.")


def render(dataset="household"):
    """Kaynakları okur ve eğitim/tahmin başlatmadan model bilgi sayfasını gösterir."""
    st.title("Model Bilgisi")
    try:
        info = _cached_model_info()
    except (FileNotFoundError, ValueError, KeyError) as error:
        st.error(f"Model bilgileri kaynak dosyalardan doğrulanamadı: {error}")
        return
    st.caption("Salt bilgilendirme: model eğitimi, tahmin veya sonuçların yeniden hesaplanması yapılmaz.")

    _models_section(info)
    st.subheader("Veri bölmesi")
    show_chart(
        _split_timeline(info["household"], info["eld"]),
        alt="Hane ve Şebeke için eğitim, doğrulama ve test dönemlerini gösteren zaman çizelgesi",
    )
    st.caption("Bölme tarihlere göre kronolojiktir; rastgele bölme yapılmaz. Veri setlerinin dönemleri farklıdır.")
    _dataset_section(info, dataset)
    _anomaly_section(info, dataset)
    _experiment_section(info)
    _protocol_section(info)
