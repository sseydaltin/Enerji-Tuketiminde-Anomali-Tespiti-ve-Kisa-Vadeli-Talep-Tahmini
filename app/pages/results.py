"""Kayıtlı test sonuçlarını salt okunur karşılaştırma sayfası."""

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

DATASET_LABELS = {"household": "Household (tek hane)", "eld": "ELD (160 müşteri toplamı)"}
FORECAST_METRICS = (
    ("MAE", "MAE", "kW"),
    ("RMSE", "RMSE", "kW"),
    ("MAPE*", "MAPE*", "%"),
    ("R²", "R2", ""),
)


def _forecast_table(frame):
    """Sonuç CSV'sindeki model metriklerini ortalama ± std biçiminde gösterir."""
    rows = []
    for model in FORECAST_MODELS:
        source = frame.loc[frame["model_key"] == model].iloc[0]
        rows.append({
            "Model": source["model_label"],
            "Seed sayısı": int(source["n_seeds"]),
            **{
                label: metric_text(source[column], source[f"{column}_std"])
                for label, column, _ in FORECAST_METRICS
            },
            "SN168'e göre MAE kazancı": metric_text(
                source["gain_vs_sn168_pct"], source["gain_vs_sn168_pct_std"], suffix="%"
            ),
        })
    return pd.DataFrame(rows)


def _horizon_figure(horizon, dataset, unit):
    """Kayıtlı horizon MAE ortalamasını, mevcutsa seed std hata çubuklarıyla çizer."""
    figure = go.Figure()
    colors = {"F13-ens": "#2364aa", "XGBoost": "#d1495b", "SN168": "#2a9d8f"}
    for model in FORECAST_MODELS:
        data = horizon.loc[horizon["model_key"] == model].sort_values("horizon")
        if data.empty:
            continue
        figure.add_trace(go.Scatter(
            x=data["horizon"],
            y=data["MAE_mean"],
            mode="lines+markers",
            name=data["model_label"].iloc[0],
            line={"color": colors[model], "width": 2},
            error_y={
                "type": "data",
                "array": data["MAE_std"].fillna(0).to_numpy(),
                "visible": bool(data["n_seeds"].max() > 1),
                "thickness": 1,
                "width": 2,
            },
            hovertemplate="Ufuk: %{x}. saat<br>MAE: %{y:.3f} " + unit + "<extra>%{fullData.name}</extra>",
        ))
    figure.update_layout(
        title=f"{dataset} test kümesi — tahmin ufkuna göre MAE",
        xaxis_title="Tahmin ufku (saat)",
        yaxis_title=f"MAE ({unit})",
        hovermode="x unified",
        legend_title="Model",
    )
    return figure


def _anomaly_performance_figure(anomaly, dataset):
    """Yöntemlerin PR-AUC ve olay F1 ortalamalarını/std'lerini birlikte gösterir."""
    figure = go.Figure()
    for column, std_column, label, color in (
        ("pr_auc_mean", "pr_auc_std", "PR-AUC", "#2364aa"),
        ("event_f1_mean", "event_f1_std", "Olay F1", "#e07a5f"),
    ):
        figure.add_trace(go.Bar(
            x=anomaly["method_label"],
            y=anomaly[column],
            name=label,
            error_y={
                "type": "data",
                "array": anomaly[std_column].fillna(0).to_numpy(),
                "visible": bool(anomaly["n_seeds"].max() > 1),
            },
            marker_color=color,
            hovertemplate="%{x}<br>" + label + ": %{y:.3f}<extra></extra>",
        ))
    figure.update_layout(
        barmode="group",
        title=f"{dataset} test kümesi — ana senaryo, seed ortalaması ± std",
        xaxis_title="Yöntem",
        yaxis_title="Skor (0–1)",
        yaxis={"range": [0, 1]},
        legend_title="Metrik",
    )
    return figure


def _type_recall_figure(anomaly, dataset):
    """Kayıtlı anomali olaylarının tür bazlı event recall değerlerini çizer."""
    figure = go.Figure()
    anomaly_types = (
        ("spike", "Ani yükselme"),
        ("drop", "Ani düşüş"),
        ("level_shift", "Seviye kayması"),
        ("pattern", "Gece örüntüsü"),
    )
    for kind, label in anomaly_types:
        column = f"event_recall_{kind}_mean"
        std_column = f"event_recall_{kind}_std"
        figure.add_trace(go.Bar(
            x=anomaly["method_label"],
            y=anomaly[column],
            name=label,
            error_y={
                "type": "data",
                "array": anomaly[std_column].fillna(0).to_numpy(),
                "visible": bool(anomaly["n_seeds"].max() > 1),
            },
            hovertemplate="%{x}<br>" + label + " event recall: %{y:.3f}<extra></extra>",
        ))
    figure.update_layout(
        barmode="group",
        title=f"{dataset} test kümesi — anomali türüne göre olay recall",
        xaxis_title="Yöntem",
        yaxis_title="Olay recall (0–1)",
        yaxis={"range": [0, 1]},
        legend_title="Sentetik anomali türü",
    )
    return figure


def _cross_dataset_table(cross):
    """Veri kümeleri arasında ölçekten bağımsız, SN168'e göreli metrikleri gösterir."""
    rows = []
    for dataset in ("household", "eld"):
        for model in FORECAST_MODELS:
            entry = cross.loc[
                (cross["dataset"] == dataset) & (cross["model_key"] == model)
            ].iloc[0]
            rows.append({
                "Veri kümesi": DATASET_LABELS[dataset],
                "Model": entry["model_label"],
                "MAPE* (%)": metric_text(entry["MAPE*"], entry["MAPE*_std"], suffix="%"),
                "R²": metric_text(entry["R2"], entry["R2_std"]),
                "SN168'e göre MAE kazancı": metric_text(
                    entry["gain_vs_sn168_pct"],
                    entry["gain_vs_sn168_pct_std"],
                    suffix="%",
                ),
            })
    return pd.DataFrame(rows)


def render():
    """Kayıtlı Household/ELD final test sonuçlarını model eğitimi olmadan karşılaştırır."""
    st.title("Sonuçlar")
    st.caption(
        "Salt okunur görünüm: tüm metrikler kayıtlı final test CSV'lerinden alınır. "
        "Model, yöntem ve eşik seçimi validation üzerinden yapılmıştır; test sonuçları "
        "seçim veya yeniden ayarlama için kullanılmamıştır."
    )
    dataset = st.selectbox(
        "Tahmin ve anomali sonuçlarını göster",
        options=list(DATASET_LABELS),
        format_func=DATASET_LABELS.get,
        key="results-dataset",
    )

    try:
        bundle = load_results_bundle(dataset)
    except (FileNotFoundError, OSError, ValueError, KeyError) as error:
        st.error(f"Kayıtlı sonuçlar yüklenemedi: {error}")
        return

    dataset_label = DATASET_LABELS[dataset]
    st.subheader(f"{dataset_label} — tahmin")
    st.caption(
        f"Test sonuçları, 24 saatlik tahmin. MAE/RMSE birimi: {bundle['unit']}. "
        "MAPE* hane için yalnız y > 0,1 kW saatlerde hesaplanır; ELD'de bu eşik etkisizdir. "
        "Seed sütunu, kayıtlı koşu sayısını belirtir."
    )
    st.dataframe(
        _forecast_table(bundle["forecast"]),
        width="stretch",
        hide_index=True,
        alt=f"{dataset_label} test kümesi F13 ensemble, XGBoost ve SN168 metrik tablosu",
    )
    st.caption(
        "Tahmin kaynakları: "
        + ", ".join(Path(path).name for path in (
            bundle["sources"]["forecast"], bundle["sources"]["horizon"]
        ))
    )
    st.plotly_chart(
        _horizon_figure(bundle["horizon"], dataset_label, bundle["unit"]),
        width="stretch",
        alt=f"{dataset_label} test kümesinde F13 ensemble, XGBoost ve SN168 MAE değerleri tahmin ufkuna göre",
    )

    st.subheader(f"{dataset_label} — anomali tespiti")
    st.caption(
        "Test / ana senaryo / evf1max: PR-AUC eşikten bağımsız sıralama ölçütüdür; "
        "olay F1 ve tür bazlı event recall kayıtlı validation eşiği ve smoothing "
        "ile değerlendirilmiştir. Değerler mevcut seed'lerin ortalaması ± standart sapmasıdır."
    )
    st.plotly_chart(
        _anomaly_performance_figure(bundle["anomaly"], dataset_label),
        width="stretch",
        alt=f"{dataset_label} test kümesinde A0, A3-s2s-norm ve A5-mean-z PR-AUC ve olay F1 sonuçları",
    )
    st.plotly_chart(
        _type_recall_figure(bundle["anomaly"], dataset_label),
        width="stretch",
        alt=f"{dataset_label} test kümesinde yöntemlerin spike, drop, seviye kayması ve gece örüntüsü olay recall sonuçları",
    )
    st.dataframe(
        bundle["anomaly_display"],
        width="stretch",
        hide_index=True,
        alt=f"{dataset_label} test anomali yöntemleri, seed sayısı, PR-AUC, olay F1 ve tür bazlı event recall tablosu",
    )
    st.caption(
        "Anomali kaynakları: "
        + ", ".join(Path(path).name for path in (
            bundle["sources"]["anomaly"], bundle["sources"]["anomaly_per_seed"]
        ))
    )

    st.subheader("Hane ve şebeke: ölçeği gözeten karşılaştırma")
    st.warning(
        "Household ile ELD arasında mutlak MAE karşılaştırması yapılmamalıdır: "
        "hane yaklaşık 1 kW, ELD 160 müşterinin toplam yüküyle yaklaşık 106 MW ölçeğindedir. "
        "Aşağıda yalnız MAPE*, R² ve her veri kümesindeki kendi SN168 baseline'ına göre "
        "MAE kazancı gösterilir. Kazanç, kayıtlı b_24h karşılaştırma tablosundaki "
        "değerdir; seed std'si final forecast test koşularından gelir."
    )
    st.dataframe(
        _cross_dataset_table(bundle["cross_dataset"]),
        width="stretch",
        hide_index=True,
        alt="Household ve ELD için MAPE, R² ve veri kümesine özgü SN168'e göre MAE kazancı",
    )
    st.caption(
        "Karşılaştırma kaynağı: "
        + Path(bundle["sources"]["cross_dataset"]).name
        + ". Household ve ELD ölçümlerinin dönemleri/ölçekleri farklıdır."
    )
    with st.expander("Protokol ve rapor kaynakları"):
        st.markdown(bundle["final_doc_excerpt"])
        st.markdown(bundle["comparison_doc_excerpt"])
        st.caption(
            "Yorum kaynakları: "
            + ", ".join(
                Path(path).name
                for path in bundle["documentation_sources"].values()
            )
            + ". Sayısal sonuçların kaynağı, yukarıda belirtilen final CSV dosyalarıdır."
        )
