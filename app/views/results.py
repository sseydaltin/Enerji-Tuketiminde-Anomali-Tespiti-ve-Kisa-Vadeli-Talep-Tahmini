"""Sonuçlar sayfası: kayıtlı final test CSV'lerinin salt okunur karşılaştırması."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.pipelines.lead_models import LITERATURE

from app.lead_services import FINAL_LEAD_MODEL, LEAD_LABELS, val_test_summary
from app.results_services import (
    ANOMALY_METHODS,
    FORECAST_MODELS,
    load_results_bundle,
    metric_text,
)
from app.ui import (
    ACTUAL_COLOR,
    MODEL_COLOR,
    SECOND_COLOR,
    THIRD_COLOR,
    compact_table,
    kpi_cards,
    dataset_strip,
    page_intro,
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
    """Üç KPI kartı: hane MAE + SN168'e göre iyileşme, şebeke MAPE + R², iki veri setinin olay F1'i.

    Kaynak: hane için final test tahmin satırı (bundle["forecast"]), şebeke için kayıtlı karşılaştırma CSV'sinin
    final satırı (cross_dataset), anomalide her veri setinin final test CSV'sindeki birincil yöntem satırı.
    """
    cross = household["cross_dataset"]
    eld_row = cross.loc[(cross["dataset"] == "eld") & (cross["model_key"] == FINAL_MODEL)].iloc[0]
    hane = household["forecast"].set_index("model_key").loc[FINAL_MODEL]

    def f1(bundle):
        return bundle["anomaly"].loc[bundle["anomaly"]["method"] == FINAL_METHOD].iloc[0]["event_f1_mean"]
    kpi_cards([
        ("Hane tahmini: MAE", f"{number(hane['MAE'], 3)} kW",
         f"Haftalık naif yönteme (SN168) göre %{number(hane['gain_vs_sn168_pct'], 1)} daha iyi."),
        ("Şebeke tahmini: MAPE", f"%{number(eld_row['MAPE*'], 2)}", f"R² {number(eld_row['R2'], 3)} (1'e yakın olan iyi)."),
        ("Anomali: olay F1", f"{number(f1(household), 3)} · {number(f1(eld), 3)}", "Hane · şebeke; 1'e yakın olan iyi."),
    ])
    st.caption("Neden farklı metrik: hane tüketimi gece sıfıra yakın olduğundan yüzde hata (MAPE) küçük kW hatalarını "
               "çok büyütür ve yanıltır; hane için MAE ve SN168'e göre iyileşme, şebeke yükü hep yüksek olduğu için "
               "MAPE ve R² verilir.")


FORECAST_TABLE_COLUMNS = {
    "Seed sayısı": ("Seed", "Kaç eğitim seed'inin ortalaması (tek seed'li modellerde std yok)."),
    "SN168'e göre MAE kazancı (%)": ("Kazanç (%)", "Haftalık naif yönteme (SN168) göre MAE kazancı (%)."),
    "MAPE (%)": ("MAPE (%)", "Yalnız gerçek tüketimi 0,1 kW üstündeki saatlerde."),
}
ANOMALY_TABLE_COLUMNS = {"Seed sayısı": ("Seed", "Kaç enjeksiyon seed'inin ortalaması.")}
SHORT_TYPES = {"spike": "Yükselme", "drop": "Düşüş", "level_shift": "Seviye", "pattern": "Gece"}
ANOMALY_TABLE_COLUMNS.update({f"{ANOMALY_LABELS[kind]} recall": (short, f"{ANOMALY_LABELS[kind]}: olay recall "
                                                                        "(yakalanan olay oranı).")
                              for kind, short in SHORT_TYPES.items()})


def _forecast_tab(bundle, label):
    """Tahmin sekmesi: seçili veri setinin model tablosu ve iki veri setinin kıyası."""
    st.subheader(f"{label}: model karşılaştırması")
    table = _forecast_table(bundle["forecast"])
    final = lambda frame: highlighted_table(frame, "Model", FORECAST_LABELS[FINAL_MODEL])   # noqa: E731
    # En çok 5 sütun: dar ekranda yatay kaydırma olmasın; kalan sütunlar açılır bölümde.
    compact_table(table[["Model", "MAE", "R²", "SN168'e göre MAE kazancı (%)"]], FORECAST_TABLE_COLUMNS, styler=final)
    with st.expander("RMSE, MAPE ve seed sayısı"):
        compact_table(table[["Model", "Seed sayısı", "RMSE", "MAPE (%)"]], FORECAST_TABLE_COLUMNS, styler=final)
    st.caption(
        "Değerler seed ortalaması ± standart sapmadır; tek koşulu modellerde std yoktur. "
        "MAPE yalnız gerçek tüketimi 0,1 kW üstündeki saatlerde hesaplanır. Final model vurguludur."
    )
    with st.expander("Hane ve Şebeke: tüm modeller (uzun tablo)"):
        compact_table(_cross_dataset_table(bundle["cross_dataset"]), FORECAST_TABLE_COLUMNS)


def _anomaly_tab(bundle, label):
    """Anomali sekmesi: yöntem tablosu, PR-AUC/F1 ve türe göre olay recall grafikleri."""
    anomaly = bundle["anomaly"]
    table = _anomaly_table(anomaly)
    final = lambda frame: highlighted_table(frame, "Yöntem", METHOD_LABELS[FINAL_METHOD])   # noqa: E731
    recalls = [f"{ANOMALY_LABELS[kind]} recall" for kind in SHORT_TYPES]
    compact_table(table[["Yöntem", "Seed sayısı", "PR-AUC", "Olay F1"]], ANOMALY_TABLE_COLUMNS, styler=final)
    with st.expander("Türe göre olay recall"):
        compact_table(table[["Yöntem", *recalls]], ANOMALY_TABLE_COLUMNS, styler=final)
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
    with st.expander("Toplama seviyesine göre hata (1 saat, 24 saat, günlük toplam)"):
        compact_table(levels, FORECAST_TABLE_COLUMNS,
                      styler=lambda frame: highlighted_table(frame, "Model", FORECAST_LABELS[FINAL_MODEL]))
        st.caption(
            "En iyi model metriğe ve seviyeye göre değişebilir. Her satır kayıtlı "
            "test tablosundan alınmıştır."
        )


LEAD_SLICES = {"other": "Davranışsal", "gap_edge": "Boşluk kenarı", "frozen": "Donmuş okuma", "genel": "Genel"}
LEAD_TABLE_COLUMNS = {
    "Other PR-AUC val": ("Davr. val", "Davranışsal (other) dilimi PR-AUC, val (birincil metrik)."),
    "Other PR-AUC test": ("Davr. test", "Davranışsal (other) dilimi PR-AUC, test (birincil metrik)."),
    "Genel PR-AUC val": ("Genel val", "Bütün anomali saatleri, PR-AUC, val."),
    "Genel PR-AUC test": ("Genel test", "Bütün anomali saatleri, PR-AUC, test."),
    "Genel ROC-AUC test": ("ROC test", "Genel ROC-AUC, test (literatür kıyası için)."),
}


def _lead_rows(summary):
    """val→test tablosu: model başına other/genel PR-AUC (val, test) ve genel ROC-AUC (test); 3 seed ort. ± std."""
    def cell(key, part, dilim, metric):
        row = summary[(summary["model"] == key[0]) & (summary["set"] == key[1]) & (summary["kume"] == part)
                      & (summary["dilim"] == dilim)]
        return _text(row[metric].iloc[0], row[f"{metric}_std"].iloc[0]) if len(row) else "—"
    rows = []
    for key, label in LEAD_LABELS.items():
        rows.append({"Model": label,
                     "Other PR-AUC val": cell(key, "val", "other", "pr_auc"),
                     "Other PR-AUC test": cell(key, "test", "other", "pr_auc"),
                     "Genel PR-AUC val": cell(key, "val", "genel", "pr_auc"),
                     "Genel PR-AUC test": cell(key, "test", "genel", "pr_auc"),
                     "Genel ROC-AUC test": cell(key, "test", "genel", "roc_auc")})
    return pd.DataFrame(rows)


def _lead_slice_figure(summary):
    """Test: alt tür (dilim) bazında PR-AUC — final, en iyi genelleşen karşılaştırma ve etiketsiz model (lejant 4)."""
    shown = {("L1", "behavior"): MODEL_COLOR, ("L1-FE", "full"): SECOND_COLOR, ("L4", "-"): THIRD_COLOR}
    test = summary[summary["kume"] == "test"]
    figure = go.Figure()
    for key, color in shown.items():
        part = test[(test["model"] == key[0]) & (test["set"] == key[1])].set_index("dilim")
        figure.add_trace(go.Bar(x=list(LEAD_SLICES.values()), y=[part.loc[d, "pr_auc"] for d in LEAD_SLICES],
                                name=LEAD_LABELS[key], marker_color=color,
                                error_y={"type": "data", "array": [part.loc[d, "pr_auc_std"] for d in LEAD_SLICES],
                                         "thickness": 1}))
    base = test[(test["model"] == "L1") & (test["set"] == "behavior")].set_index("dilim")
    figure.add_trace(go.Scatter(x=list(LEAD_SLICES.values()), y=[base.loc[d, "taban"] for d in LEAD_SLICES],
                                mode="markers", name="Taban (rastgele skor)",
                                marker={"symbol": "line-ew-open", "size": 28, "color": ACTUAL_COLOR, "line": {"width": 2}}))
    figure.update_yaxes(title_text="PR-AUC (test)", range=[0, 1.05], hoverformat=".3f")
    figure.update_layout(barmode="group", bargap=0.3, height=400, hovermode="closest",
                         title="LEAD, test kümesi: alt tür bazında PR-AUC (40 test binası × 3 seed)")
    return figure


def _lead_tab():
    """LEAD sekmesi: val→test tablosu, alt tür PR-AUC grafiği, literatür kıyası ve test sonrası gözlem."""
    try:
        summary = val_test_summary()
    except (FileNotFoundError, OSError) as error:
        st.error(f"Kayıtlı LEAD özeti okunamadı: {error}")
        return
    st.subheader("Val → test (bina bölmesi, 3 seed)")
    rows = _lead_rows(summary)
    final = lambda frame: highlighted_table(frame, "Model", LEAD_LABELS[FINAL_LEAD_MODEL])   # noqa: E731
    compact_table(rows[["Model", "Other PR-AUC val", "Other PR-AUC test"]], LEAD_TABLE_COLUMNS, styler=final)
    with st.expander("Genel PR-AUC (val, test) ve genel ROC-AUC (test)"):
        compact_table(rows[["Model", "Genel PR-AUC val", "Genel PR-AUC test", "Genel ROC-AUC test"]], LEAD_TABLE_COLUMNS,
                      styler=final)
    st.caption("Final model val'de önceden yazılmış kuralla seçildi ve vurguludur. Birincil metrik davranışsal (other) "
               "PR-AUC'dir; kaynak results/lead/tables/test_summary_val_vs_test.csv.")
    show_chart(_lead_slice_figure(summary), alt="LEAD test kümesinde üç modelin alt tür bazında PR-AUC değerleri")
    test = summary[summary["kume"] == "test"]

    def roc(key):
        row = test[(test["model"] == key[0]) & (test["set"] == key[1]) & (test["dilim"] == "genel")].iloc[0]
        return f"{number(row['roc_auc'], 3)} ± {number(row['roc_auc_std'], 3)}"
    st.subheader("Literatür kıyası (koşullar farklı)")
    lines = ["| Kaynak | Değer | Koşul farkı |", "|---|---|---|"]
    lines += [f"| {name} | {value} | {cond} |" for name, value, cond in LITERATURE]
    lines += [f"| **Bu proje, final** ({LEAD_LABELS[FINAL_LEAD_MODEL]}) | ROC-AUC {roc(FINAL_LEAD_MODEL)} | "
              "nedensel (yalnız geçmiş); 200 etiketli bina 120/40/40; test |",
              f"| Bu proje, offline satır | ROC-AUC {roc(('L1-OFF', 'full'))} | iki yönlü öznitelik; yalnız kıyas için |"]
    st.markdown("\n".join(lines))
    st.caption("Kıyas yalnız bu koşul farklarıyla okunur; \"daha iyi / daha kötü\" doğrudan söylenmez.")

    def other(key, part):
        row = summary[(summary["model"] == key[0]) & (summary["set"] == key[1]) & (summary["kume"] == part)
                      & (summary["dilim"] == "other")].iloc[0]
        return number(row["pr_auc"], 3)
    with st.container(border=True):
        st.markdown("**Test sonrası gözlem** (final modeli değiştirmez)")
        st.markdown(f"Val'de ayırt edilemeyen iki model testte ayrıştı: davranışsal PR-AUC final "
                    f"{LEAD_LABELS[FINAL_LEAD_MODEL]} için {other(FINAL_LEAD_MODEL, 'test')}, "
                    f"{LEAD_LABELS[('L1-FE', 'full')]} için {other(('L1-FE', 'full'), 'test')}. Seçim val'de yapıldığı "
                    "için final model değişmez. Ders: tek bir val kümesi (seed başına 40 bina) küçük farkları ayırt "
                    "edemedi; davranışsal skorlar parçanın bina bileşimine çok duyarlı. Ayrıntı: docs/lead_sonuclar.md §7.")


def render(dataset=None):
    """Kayıtlı final test sonuçlarını üstte karşılaştırır, altta sekmelerde ayrıntılandırır."""
    page_intro("Sonuçlar", "Hane ve şebeke için kayıtlı final test sonuçları: üstte iki veri setinin ana sayıları, altta "
               "seçili veri setinde model karşılaştırması, anomali yöntemleri ve hata analizi. Burada hiçbir şey yeniden "
               "hesaplanmaz.")
    dataset = dataset or dataset_strip("results", "Sekmelerdeki veri seti")
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

    st.caption(f"Sekmelerde gösterilen veri seti: **{label}** (üstteki şeritten değiştirilebilir).")
    forecast_tab, anomaly_tab, error_tab, lead_tab = st.tabs(["Tahmin", "Anomali", "Hata analizi", "LEAD (gerçek etiket)"])
    with forecast_tab:
        _forecast_tab(bundle, label)
    with anomaly_tab:
        _anomaly_tab(bundle, label)
    with error_tab:
        _error_tab(bundle, label)
    with lead_tab:
        _lead_tab()
