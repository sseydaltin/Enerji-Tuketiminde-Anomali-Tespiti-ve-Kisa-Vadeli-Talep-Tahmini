"""Sonuçlar sayfası: kayıtlı final test CSV'lerinin salt okunur karşılaştırması."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.results_services import (
    ANOMALY_METHODS,
    FORECAST_MODELS,
    load_results_bundle,
    metric_text,
)
from app.ui import (
    ANOMALY_LABELS,
    DATASET_LABELS,
    FINAL_METHOD,
    FINAL_MODEL,
    FORECAST_COLORS,
    FORECAST_LABELS,
    METHOD_COLORS,
    METHOD_LABELS,
    decimal_comma,
    highlighted_table,
    number,
    show_chart,
)

FORECAST_METRICS = (
    ("MAE", "MAE", "kW"),
    ("RMSE", "RMSE", "kW"),
    ("MAPE (%)", "MAPE*", "%"),
    ("R²", "R2", ""),
)
GAIN_COLUMN = "SN168'e göre MAE kazancı (%)"
LEVEL_LABELS = {
    "a_h1": "1 saat sonrası (kW)",
    "b_24h": "24 saatlik ufuk (kW)",
    "c_daily": "Günlük toplam (kWh)",
}
LEVEL_GROUPS = {
    "F13-ens (final)": "F13-ens",
    "F12-L336 XGBoost": "XGBoost",
    "F2 Seasonal Naive 168": "SN168",
}


def _text(mean, std):
    """Ortalama ± std metnini Türkçe ondalık virgülüyle üretir."""
    return decimal_comma(metric_text(mean, std))


def _forecast_table(frame):
    """Sonuç CSV'sindeki model metriklerini ortalama ± std biçiminde gösterir."""
    rows = []
    for model in FORECAST_MODELS:
        source = frame.loc[frame["model_key"] == model].iloc[0]
        rows.append({
            "Model": FORECAST_LABELS[model],
            "Seed sayısı": int(source["n_seeds"]),
            **{
                label: _text(source[column], source[f"{column}_std"])
                for label, column, _ in FORECAST_METRICS
            },
            GAIN_COLUMN: _text(source["gain_vs_sn168_pct"], source["gain_vs_sn168_pct_std"]),
        })
    return pd.DataFrame(rows)


def _anomaly_table(anomaly):
    """Anomali seed ortalamalarını ve std değerlerini Türkçe sütunlarla tablolaştırır."""
    rows = []
    for _, source in anomaly.iterrows():
        row = {
            "Yöntem": METHOD_LABELS[source["method"]],
            "Seed sayısı": int(source["n_seeds"]),
            "PR-AUC": _text(source["pr_auc_mean"], source["pr_auc_std"]),
            "Olay F1": _text(source["event_f1_mean"], source["event_f1_std"]),
        }
        for kind, label in ANOMALY_LABELS.items():
            row[f"{label} recall"] = _text(
                source[f"event_recall_{kind}_mean"], source[f"event_recall_{kind}_std"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _horizon_figure(horizon, dataset, unit):
    """Kayıtlı ufuk MAE ortalamasını, mevcutsa seed std hata çubuklarıyla çizer."""
    figure = go.Figure()
    for model in FORECAST_MODELS:
        data = horizon.loc[horizon["model_key"] == model].sort_values("horizon")
        if data.empty:
            continue
        line = {"color": FORECAST_COLORS[model], "width": 2.5 if model == FINAL_MODEL else 2}
        if model == "SN168":
            line["dash"] = "dash"
        figure.add_trace(go.Scatter(
            x=data["horizon"],
            y=data["MAE_mean"],
            mode="lines+markers" if model != "SN168" else "lines",
            name=FORECAST_LABELS[model],
            line=line,
            error_y={
                "type": "data",
                "array": data["MAE_std"].fillna(0).to_numpy(),
                "visible": bool(data["n_seeds"].max() > 1),
                "thickness": 1,
                "width": 2,
            },
        ))
    figure.update_xaxes(title_text="Tahmin ufku (saat)", tick0=1, dtick=1, range=[0.5, 24.5])
    figure.update_yaxes(title_text=f"MAE ({unit})", hoverformat=".3f")
    figure.update_layout(title=f"{dataset}, test kümesi: tahmin ufkuna göre MAE", height=430)
    return figure


def _method_bars(anomaly, metrics, title, y_title):
    """Yöntemleri ortak renklerle gruplu çubukta karşılaştırır.

    Girdi: anomali özeti; metrics = (x etiketi, ortalama sütunu, std sütunu) listesi.
    Çıktı: Plotly figürü. Tam değerler üstteki tablodadır; hata çubuklarıyla çakışmasın
    diye çubuk üstüne sayı yazılmaz. Seed std'si 1'i aşabildiği için eksen 1,15'e kadar açıktır.
    """
    figure = go.Figure()
    for method in ANOMALY_METHODS:
        row = anomaly.loc[anomaly["method"] == method].iloc[0]
        figure.add_trace(go.Bar(
            x=[label for label, _, _ in metrics],
            y=[row[mean] for _, mean, _ in metrics],
            name=METHOD_LABELS[method],
            marker_color=METHOD_COLORS[method],
            error_y={
                "type": "data",
                "array": [0 if pd.isna(row[std]) else row[std] for _, _, std in metrics],
                "visible": bool(row["n_seeds"] > 1),
                "thickness": 1,
            },
            hovertemplate="%{y:.3f}",
        ))
    figure.update_yaxes(title_text=y_title, range=[0, 1.15], hoverformat=".3f")
    figure.update_layout(barmode="group", bargap=0.3, bargroupgap=0.05, title=title, height=400)
    return figure


def _cross_dataset_table(cross):
    """Veri setleri arasında ölçekten bağımsız, SN168'e göreli metrikleri gösterir."""
    rows = []
    for dataset in ("household", "eld"):
        for model in FORECAST_MODELS:
            entry = cross.loc[
                (cross["dataset"] == dataset) & (cross["model_key"] == model)
            ].iloc[0]
            rows.append({
                "Veri seti": DATASET_LABELS[dataset],
                "Model": FORECAST_LABELS[model],
                "MAPE (%)": _text(entry["MAPE*"], entry["MAPE*_std"]),
                "R²": _text(entry["R2"], entry["R2_std"]),
                GAIN_COLUMN: _text(entry["gain_vs_sn168_pct"], entry["gain_vs_sn168_pct_std"]),
            })
    return pd.DataFrame(rows)


def _level_table(forecast_path):
    """Kayıtlı seviye tablosundan hatanın kapsama göre nasıl değiştiğini okur.

    Girdi: forecast_test_results.csv yolu; seviye tablosu aynı klasördedir.
    Çıktı: Seviye/Model sütunlu DataFrame veya dosya yoksa None.
    Neden: 1 saat, 24 saat ve günlük toplam farklı soruları yanıtlar; en iyi model
    metriğe göre değişebilir. Değerler CSV'den aynen okunur, yeniden hesaplanmaz.
    """
    path = Path(forecast_path).parent / "forecast_levels_test.csv"
    if not path.is_file():
        return None
    levels = pd.read_csv(path)
    rows = []
    for level, level_label in LEVEL_LABELS.items():
        for group, model in LEVEL_GROUPS.items():
            entry = levels.loc[(levels["level"] == level) & (levels["group"] == group)]
            if entry.empty:
                continue
            entry = entry.iloc[0]
            rows.append({
                "Seviye": level_label,
                "Model": FORECAST_LABELS[model],
                "MAE": _text(entry["MAE_mean"], entry["MAE_std"]),
                "MAPE (%)": _text(entry["MAPE*_mean"], entry["MAPE*_std"]),
                "R²": _text(entry["R2_mean"], entry["R2_std"]),
            })
    return pd.DataFrame(rows)


def _comparison_cards(household, eld):
    """Hane ve Şebeke'yi MAPE, R² ve olay F1 için yan yana kartlarla karşılaştırır.

    Kaynak: tahminde kayıtlı karşılaştırma CSV'sinin final model satırı, anomalide her
    veri setinin kendi final test CSV'sindeki birincil yöntem satırı.
    """
    cross = household["cross_dataset"]

    def forecast_row(dataset):
        """Bir veri setinin final tahmin modeli satırını döndürür."""
        return cross.loc[(cross["dataset"] == dataset) & (cross["model_key"] == FINAL_MODEL)].iloc[0]

    def anomaly_row(bundle):
        """Bir veri setinin birincil anomali yöntemi satırını döndürür."""
        return bundle["anomaly"].loc[bundle["anomaly"]["method"] == FINAL_METHOD].iloc[0]

    cards = (
        ("MAPE", "Tahmin, küçük olan iyi", "%", 2, lambda dataset, _: forecast_row(dataset)["MAPE*"]),
        ("R²", "Tahmin, 1'e yakın olan iyi", "", 3, lambda dataset, _: forecast_row(dataset)["R2"]),
        ("Olay F1", "Anomali, 1'e yakın olan iyi", "", 3, lambda _, bundle: anomaly_row(bundle)["event_f1_mean"]),
    )
    for column, (title, hint, prefix, digits, read_value) in zip(st.columns(3), cards):
        with column.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(hint)
            left, right = st.columns(2)
            left.metric(DATASET_LABELS["household"], f"{prefix}{number(read_value('household', household), digits)}")
            right.metric(DATASET_LABELS["eld"], f"{prefix}{number(read_value('eld', eld), digits)}")


def _forecast_tab(bundle, label):
    """Tahmin sekmesi: seçili veri setinin model tablosu ve iki veri setinin kıyası."""
    st.subheader(f"{label}: model karşılaştırması")
    st.table(highlighted_table(
        _forecast_table(bundle["forecast"]), "Model", FORECAST_LABELS[FINAL_MODEL],
    ))
    st.caption(
        "Değerler seed ortalaması ± standart sapmadır; tek koşulu modellerde std yoktur. "
        "MAPE yalnız gerçek tüketimi 0,1 kW üstündeki saatlerde hesaplanır. Final model vurguludur."
    )
    st.subheader("Hane ve Şebeke: tüm modeller")
    st.table(_cross_dataset_table(bundle["cross_dataset"]).style.hide(axis="index"))


def _anomaly_tab(bundle, label):
    """Anomali sekmesi: yöntem tablosu, PR-AUC/F1 ve türe göre olay recall grafikleri."""
    anomaly = bundle["anomaly"]
    st.table(highlighted_table(_anomaly_table(anomaly), "Yöntem", METHOD_LABELS[FINAL_METHOD]))
    st.caption(
        "Ana senaryoda seed ortalaması ± standart sapma. Eşik ve yumuşatma doğrulama kümesinde "
        "seçilmiştir. Birincil yöntem doğrulama kümesinde belirlenmiştir ve vurguludur; testte "
        "başka bir yöntem daha yüksek çıksa da sonradan değiştirilmez."
    )
    left, right = st.columns(2)
    with left:
        show_chart(
            _method_bars(
                anomaly,
                (("PR-AUC", "pr_auc_mean", "pr_auc_std"), ("Olay F1", "event_f1_mean", "event_f1_std")),
                f"{label}, test kümesi: genel başarı",
                "Skor (0–1)",
            ),
            alt=f"{label} için yöntemlerin PR-AUC ve olay F1 değerleri",
        )
    with right:
        show_chart(
            _method_bars(
                anomaly,
                tuple(
                    (name, f"event_recall_{kind}_mean", f"event_recall_{kind}_std")
                    for kind, name in ANOMALY_LABELS.items()
                ),
                f"{label}, test kümesi: türe göre olay recall",
                "Olay recall (0–1)",
            ),
            alt=f"{label} için yöntemlerin anomali türüne göre olay recall değerleri",
        )


def _error_tab(bundle, label):
    """Hata analizi sekmesi: hatanın ufka ve toplama seviyesine göre değişimi."""
    show_chart(
        _horizon_figure(bundle["horizon"], label, bundle["unit"]),
        alt=f"{label} test kümesinde modellerin tahmin ufkuna göre MAE değerleri",
    )
    levels = _level_table(bundle["sources"]["forecast"])
    if levels is None:
        st.info("Seviye tablosu bulunamadı; yalnız ufuk grafiği gösteriliyor.")
        return
    st.subheader("Toplama seviyesine göre hata")
    st.table(highlighted_table(levels, "Model", FORECAST_LABELS[FINAL_MODEL]))
    st.caption(
        "En iyi model metriğe ve seviyeye göre değişebilir. Her satır kayıtlı "
        "test tablosundan alınmıştır."
    )


def render(dataset="household"):
    """Kayıtlı final test sonuçlarını üstte karşılaştırır, altta sekmelerde ayrıntılandırır."""
    st.title("Sonuçlar")
    try:
        household = load_results_bundle("household")
        eld = load_results_bundle("eld")
    except (FileNotFoundError, OSError, ValueError, KeyError) as error:
        st.error(f"Kayıtlı sonuçlar yüklenemedi: {error}")
        return
    bundle = household if dataset == "household" else eld
    label = DATASET_LABELS[dataset]

    st.subheader("Hane ve Şebeke")
    _comparison_cards(household, eld)
    st.caption(
        "Salt okunur görünüm: tüm sayılar kayıtlı final test dosyalarından gelir. Kartlar "
        "Topluluk modelini ve Seq2Seq artık skorunu gösterir. Mutlak MAE karşılaştırılmaz: hane "
        "yaklaşık 1 kW, şebeke yaklaşık 106 MW ölçeğindedir ve dönemleri farklıdır."
    )

    st.caption(f"Sekmelerde gösterilen veri seti: **{label}** (kenar çubuğundan değiştirilebilir).")
    forecast_tab, anomaly_tab, error_tab = st.tabs(["Tahmin", "Anomali", "Hata analizi"])
    with forecast_tab:
        _forecast_tab(bundle, label)
    with anomaly_tab:
        _anomaly_tab(bundle, label)
    with error_tab:
        _error_tab(bundle, label)
