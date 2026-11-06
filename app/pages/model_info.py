"""Projenin kaynak kodlarla doğrulanmış mimari ve deney protokolü."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.model_info_services import load_model_info


@st.cache_data(ttl=600, show_spinner=False)
def _cached_model_info():
    """Kaynak metadata özetini kısa süreli önbellekten döndürür."""
    return load_model_info()


def _flow_figure():
    """Train, validation ve test akışlarını ayrı, geniş kutularla gösterir."""
    figure = go.Figure()

    lanes = (
        (425, 705, "TRAIN — ölçekleme, model eğitimi ve OOF kalibrasyonu", "#f3f7fb"),
        (225, 395, "VALIDATION — checkpoint, model ve anomali ayarı", "#f6f8fc"),
        (35, 185, "FINAL TEST — dondurulmuş seçimlerle tek değerlendirme", "#f3f7fb"),
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
        "train_scaler": (25, 210, 615, 670, "Train verisi", "Saatlik train bölümü"),
        "scaler_fit": (250, 430, 615, 670, "Scaler fit", "Yalnız train"),
        "scaler_stats": (470, 650, 615, 670, "Scaler istatistikleri", "Val/test'e fit yok"),
        "train_model": (25, 210, 540, 595, "Train verisi", "Saatlik train bölümü"),
        "train_windows": (250, 430, 540, 595, "Train pencereleri", "Yalnız train aralığı"),
        "fit_weights": (500, 685, 540, 595, "Model ağırlıkları", "Train pencereleriyle<br>gradient update"),
        "candidate": (760, 980, 540, 595, "Aday<br>checkpoint", "Epoch seçimi val MAE"),
        "train_folds": (25, 210, 465, 520, "Train verisi", "Yalnız train dönemi"),
        "fold_models": (250, 410, 465, 520, "Zamansal CV", "3 ardışık fold<br>yalnız geçmişle fit"),
        "oof_residual": (450, 620, 465, 520, "OOF artıklar", "Modelin görmediği<br>train saatleri"),
        "oof_std": (660, 825, 465, 520, "A3 std tablosu", "Saat × haftanın günü"),
        "val_inputs": (25, 245, 270, 355, "Validation girdileri", "Val serisi + train scaler,<br>checkpoint ve OOF std"),
        "val_scores": (300, 505, 270, 355, "Validation skorları", "Tahmin MAE; A0/A3/A5<br>val-dev ana senaryo"),
        "val_select": (520, 750, 270, 355, "Validation seçimi", "Checkpoint/epoch: val MAE<br>eşik + smoothing: val-dev"),
        "frozen": (780, 975, 270, 355, "Kararları<br>dondur", "Model, eşik, pencere"),
        "test_inputs": (25, 285, 75, 150, "Test girdileri", "Test serisi + train scaler,<br>checkpoint ve OOF std"),
        "test_apply": (350, 650, 75, 150, "Sabitleri uygula", "Validation'da seçilen<br>model ve eşikler"),
        "test_report": (700, 975, 75, 150, "Final metrikler", "Sonuçları raporla;<br>seçim/ayar yapma"),
    }

    def arrow(start, end):
        """Kutu merkezleri arasında tek yönlü, gerçek süreç bağlantısı çizer."""
        x1, y1 = start
        x2, y2 = end
        figure.add_annotation(
            x=x2, y=y2, ax=x1, ay=y1,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=2, arrowsize=1,
            arrowwidth=1.7, arrowcolor="#536b86",
            standoff=8, startstandoff=8,
        )

    def center(name):
        x0, x1, y0, y1, *_ = nodes[name]
        return (x0 + x1) / 2, (y0 + y1) / 2

    def east(name):
        x0, x1, y0, y1, *_ = nodes[name]
        return x1, (y0 + y1) / 2

    def west(name):
        x0, x1, y0, y1, *_ = nodes[name]
        return x0, (y0 + y1) / 2

    def north(name):
        x0, x1, y0, y1, *_ = nodes[name]
        return (x0 + x1) / 2, y1

    def south(name):
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
            line={"color": "#2364aa", "width": 1.5},
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
        plot_bgcolor="white",
        paper_bgcolor="white",
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
    """Kaydedilmiş tahmin modeli metadata'sını jüriye yönelik tablolaştırır."""
    rows = []
    for name, label in (("F5-opt", "F5-opt LSTM"), ("F10-opt", "F10-opt Seq2Seq")):
        model = info["checkpoints"][name]
        if not model["available"]:
            rows.append({"Model": label, "Durum": "Checkpoint bulunamadı"})
            continue
        forecast = model["forecast"]
        optuna = model["optuna"]
        if name == "F5-opt":
            architecture = (
                f"LSTM ({forecast['layers']} katman); son gizil durumdan "
                "24 çıkışlı doğrusal başlık"
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
            "Optuna best": (
                f"hidden={optuna['hidden']}, layers={optuna['layers']}, "
                f"dropout={float(optuna['dropout']):.4f}, "
                f"lr={float(optuna['lr']):.6g}, batch={optuna['batch_size']}, "
                f"weight_decay={float(optuna['weight_decay']):.3g}, "
                f"loss={optuna['loss']}"
            ),
            "Kaynak": f"{model['path']} + Optuna best CSV",
        })
    return rows


def _split_rows(config):
    """Config içindeki kapsayıcı başlangıç ve bitiş tarihlerini tabloya çevirir."""
    return [
        {
            "Küme": name,
            "Başlangıç": settings["start"],
            "Bitiş": settings["end"],
        }
        for name, settings in config["split"].items()
        if name in {"train", "val", "test"}
    ]


def _threshold_table(frame):
    """Validation seçim dosyasındaki yöntem, pencere ve eşikleri biçimlendirir."""
    return frame.rename(columns={
        "method": "Yöntem",
        "smooth": "Geçmişe bakan yumuşatma (saat)",
        "threshold": "Validation eşiği",
    })[["Yöntem", "Geçmişe bakan yumuşatma (saat)", "Validation eşiği"]]


def render():
    """Kaynakları okur ve eğitim/inference başlatmadan jüri bilgi sayfasını gösterir."""
    st.title("Model Info")
    try:
        info = _cached_model_info()
    except (FileNotFoundError, ValueError, KeyError) as error:
        st.error(f"Model bilgileri kaynak dosyalardan doğrulanamadı: {error}")
        return

    household = info["household"]
    eld = info["eld"]
    st.caption(
        "Salt bilgilendirme sayfası: model eğitimi, tahmin inference'ı veya "
        "sonuçların yeniden hesaplanması yapılmaz."
    )

    st.header("Sistem mimarisi")
    st.plotly_chart(
        _flow_figure(),
        alt="Kronolojik veri akışını, train tabanlı ölçeklemeyi, validation seçimlerini ve test değerlendirmesini gösteren şema",
        config={"displayModeBar": False},
    )
    st.markdown(
        "Saatlik veri ön işlenir; train/validation/test zaman sırasıyla ayrılır. "
        "Tahmin modelleri train ile eğitilir; A3 normalizasyonu train OOF artıklarından "
        "kurulur. Model, eşik ve yumuşatma seçimi validation'da yapılır. Test yalnız "
        "sabitlenen seçimlerle final ölçüm içindir."
    )

    st.header("Tahmin modelleri")
    st.dataframe(_model_rows(info), hide_index=True, alt="Tahmin modeli mimari, pencere, özellik ve Optuna ayarları")
    st.caption(
        "Mimari kaynakları: src/models/lstm.py, src/models/seq2seq.py; seçilmiş "
        "parametreler: results/tables/optuna_lstm_best.csv ve optuna_seq2seq_best.csv."
    )
    f5 = info["checkpoints"]["F5-opt"]
    f10 = info["checkpoints"]["F10-opt"]
    if f5["available"] and f10["available"]:
        if f5["features"] != f10["features"]:
            st.error("F5 ve F10 checkpoint özellik listeleri uyuşmuyor; F13 üyeleri aynı girişi görmüyor.")
        elif f5["forecast"]["lookback"] != f10["forecast"]["lookback"]:
            st.error("F5 ve F10 checkpoint lookback değerleri uyuşmuyor; F13 ensemble uyumsuz.")
        else:
            st.info(
                "F13 ensemble: 0,8 × F10-opt + 0,2 × F5-opt. Ağırlıklar kayıtlı "
                "final protokolündendir. Bu checkpoint'lerin gerçek giriş penceresi "
                f"{f10['forecast']['lookback']} saat ve ufku {f10['forecast']['horizon']} saattir; "
                "config.yaml'daki genel 168 saat varsayılanıyla karıştırılmamalıdır."
            )
        with st.expander("Checkpoint özellikleri ve train ölçek istatistikleri"):
            for name in ("F5-opt", "F10-opt"):
                checkpoint = info["checkpoints"][name]
                if checkpoint["available"]:
                    st.write(f"**{name} özellik sırası:** {', '.join(checkpoint['features'])}")
                    st.write(
                        f"**Özellik mean:** {np.array2string(np.asarray(checkpoint['feature_mean']), precision=5)}"
                    )
                    st.write(
                        f"**Özellik scale:** {np.array2string(np.asarray(checkpoint['feature_scale']), precision=5)}"
                    )
                    st.write(
                        f"**Hedef mean / scale:** {checkpoint['target_mean'][0]:.6g} / "
                        f"{checkpoint['target_scale'][0]:.6g}"
                    )
                    st.caption(f"Kaynak: {checkpoint['path']} (weights_only metadata)")

    xgb = household["forecast"]["xgboost"]
    xgb_lags = [lag for lag in xgb["lags"] if lag <= 336]
    st.subheader("F12-L336 XGBoost")
    if info["xgb_artifacts"]:
        st.write(f"Model artifact'ları bulundu: {', '.join(info['xgb_artifacts'])}")
    else:
        st.warning(
            "Yerel eğitilmiş XGBoost model checkpoint'i bulunamadı. Bu sayfa tahmin "
            "üretmez ve modeli yeniden eğitmez. Aşağıdaki değerler yalnız config "
            "ayarlarıdır; kaydedilmiş ağırlık metadata'sı değildir."
        )
    st.write(
        f"Config'teki XGBRegressor ayarları: lookback 336 saat (F12-L336 deney "
        "kimliğinden), hedef gecikmeleri "
        f"{xgb_lags}, hedef saatin son gözlenen takvim özellikleri; "
        f"n_estimators üst sınırı {xgb['n_estimators']}, learning_rate "
        f"{xgb['learning_rate']}, max_depth {xgb['max_depth']}, "
        f"subsample {xgb['subsample']}, colsample_bytree {xgb['colsample_bytree']}, "
        f"early stopping {xgb['early_stopping_rounds']} tur (validation MAE)."
    )
    st.caption(
        "Config'inde XGBoost 336 saatlik deneyin lag listesi 1–24, 48, 168 ve "
        "336 saat. Bu deney lookback=336 override eder; config.yaml genel varsayılanı "
        "168 saattir. Konfigürasyondaki boosting sayısı early stopping ile sınırlandırılır."
    )
    st.write(
        "SN168, tahmin ufkuna önceki haftanın aynı saatlerini kopyalar. Girdi olarak "
        "168 saatlik hedef geçmişi kullanır; ayrıca eğitilen ağırlığı yoktur."
    )

    st.header("Veri kümeleri")
    dataset_columns = st.columns(2)
    household_split = _split_rows(household)
    eld_split = _split_rows(eld)
    with dataset_columns[0]:
        st.subheader("Household")
        st.write(
            f"Hedef **{household['preprocess']['target']}** (kW); dakikalık hane "
            "ölçümleri saatlik ortalamaya indirgenir."
        )
        st.dataframe(household_split, hide_index=True, alt="Household kronolojik bölme tarihleri")
        st.write(
            "Checkpoint özellik seti B: hedef + "
            f"{len(f10['features']) - 1 if f10['available'] else 8} takvim özellikleri. "
            f"Ham veri: {'mevcut' if info['availability']['household_raw'] else 'eksik'}; "
            f"hourly.parquet: {'mevcut' if info['availability']['household_processed'] else 'eksik'}."
        )
    with dataset_columns[1]:
        st.subheader("ELD")
        st.write(
            f"Hedef **{eld['preprocess']['target']}** (kW); 160 seçilmiş müşterinin "
            "15 dakikalık toplam yükü saatlik ortalamaya indirgenir."
        )
        st.dataframe(eld_split, hide_index=True, alt="ELD kronolojik bölme tarihleri")
        st.write(
            f"ELD raw veri: {'mevcut' if info['availability']['eld_raw'] else 'yerelde yok'}; "
            f"işlenmiş saatlik dosya: {'mevcut' if info['availability']['eld_processed'] else 'yerelde yok'}."
        )
        st.write(
            "ELD checkpoint'leri ve kayıtlı deney sonuçları bulunsa da ham/işlenmiş veri "
            "olmadığından bu çalışma alanında canlı ELD veri akışı ve tahmini yapılamaz."
        )
        for name in ("ELD-F5-opt", "ELD-F10-opt"):
            checkpoint = info["checkpoints"][name]
            status = "mevcut" if checkpoint["available"] else "eksik"
            st.caption(f"{name} checkpoint: {status}.")

    st.header("Anomali tespiti")
    st.markdown(
        rf"""
**A0 — aynı saatlik geçmişe göre hareketli z-skoru**

`A0[t] = |y[t] − mean(y[t−24], …, y[t−{household['anomaly']['zscore_window']}])| / max(std(past), {household['anomaly']['zscore_min_std']} kW)`.
Önceki yedi günün aynı saati kullanılır; hedef saat referans ortalamasına katılmaz.

**A3-s2s-norm — F10-opt h=1 tahmin artığı**

`A3[t] = |y[t] − ŷ_F10,h=1[t]| / std_OOF[hour(t), weekday(t)]`.
Payda, train'in son 12 ayındaki üç zamansal CV katında üretilen out-of-fold
artıkların saat × haftanın günü hücresindeki standart sapmasıdır.

**A5-mean-z — A0 ve ensemble artığının ortak ölçekte ortalaması**

`A5[t] = (z_val(A0[t]) + z_val(A3-ens-norm[t])) / 2`.
A3-ens-norm, `0,8 × F10-opt + 0,2 × F5-opt` h=1 tahmin artığını kullanır.
İki z-skorunun referansı enjeksiyonsuz validation skor dağılımıdır.
"""
    )
    st.write(
        "Ana senaryoda enjeksiyon, orijinal serinin kopyasında spike (+3–5 train σ), "
        "drop (~0,05 kW), seviye kayması (×2–2,5) ve 00:00–05:00 gece örüntüsü "
        "oluşturur. Eşik ve yalnız geçmişe bakan smoothing penceresi validation "
        "seçiminden gelir; testte ayarlanmaz."
    )
    st.markdown(
        "**Anomali metrikleri:** PR-AUC, etiketli saatlerin sürekli skor sıralamasındaki "
        "precision–recall başarısını özetler. Olay recall, her etiketli olayın en az bir "
        "saatinin yakalanma oranıdır; olay precision, gerçek anomaliyle kesişen alarm "
        "bloklarının tüm alarm bloklarına oranıdır; olay F1 bu ikisinin harmonik "
        "ortalamasıdır. Tür bazlı recall aynı olay hesabını spike/drop/seviye kayması/"
        "örüntü türlerinin her biri için ayrı yapar. Olay sınırları birleştirilmiş "
        "değerlendirme zaman eksenindeki ardışık bloklardan çıkarılır."
    )
    threshold_columns = st.columns(2)
    with threshold_columns[0]:
        st.subheader("Household validation seçimi")
        st.dataframe(
            _threshold_table(info["thresholds"]["household"]),
            hide_index=True,
            alt="Household için validation seçim CSV'sindeki anomali eşikleri",
        )
        st.caption("Kaynak: results/tables/anomaly_protocol_selection.csv (val-dev, ana senaryo).")
    with threshold_columns[1]:
        st.subheader("ELD validation seçimi")
        st.dataframe(
            _threshold_table(info["thresholds"]["eld"]),
            hide_index=True,
            alt="ELD için validation seçim CSV'sindeki anomali eşikleri",
        )
        st.caption("Kaynak: results/eld/tables/anomaly_selection.csv (validation, ana senaryo).")
    st.warning(
        "Kaynak notu: docs/final_test_protokolu.md §5'te A0, revizyon öncesi seçim "
        "olarak geçer; §5a revizyonuyla final yöntem A3-s2s-norm olarak değiştirilmiştir "
        "(README ve seçim CSV'siyle uyumlu). Final A5 referansı için "
        "docs/final_test_protokolu.md test komutu ve src/anomaly_protocol.py içindeki "
        "açık reference_split='val_dev' çağrısı, ayrıca results/experiments.csv notu "
        "val-dev kullanımını destekler. src/evaluate_anomaly.py içindeki genel "
        "fit_combination varsayılanı tam val'dir; final protokol bu varsayılanı açıkça "
        "geçersiz kılar. Kaydedilmiş test tablolarında reference_split alanı ve bağımsız "
        "çalıştırma günlüğü bulunmadığından kanıt final kod yolu ve deney notuna dayanır. "
        "Eşikler burada CSV'den aynen okunur ve yeniden seçilmez."
    )

    st.header("Deney protokolü ve metrikler")
    st.markdown(
        f"""
- Train-only scaler: veri özellikleri eğitim kümesinde fit edilir; validation/test yalnız transform edilir.
- Tahmin hiperparametreleri Optuna'da train'in son 12 ayındaki 3 zamansal CV katında seçilir; seçilen konfigürasyon validation ile izlenir.
- Anomali skoru normalizasyonu train OOF artıklarından; eşik ve smoothing yalnız validation'dan gelir.
- Test kümesi model/eşik seçimi için kullanılmaz; final sabitlenmiş seçimle raporlanır.
- Final tahmin checkpoint'lerinde early stopping tam validation'ı kullandığından val-hold bu checkpoint'ler için tamamen bağımsız değildir; final test seçimden ayrıdır.
- Tahmin seed'leri config'te `{household['multi_seeds']}`; final test tahminleri 3 seed, anomali test raporları 5 seed ile ortalama ± örnek standart sapması sunar.
- MAPE*: yalnız gerçek hedef `y > {household['evaluation']['mape_min_y']} kW` iken `mean(|(y−ŷ)/y|) × 100`.
- **MAE**: ortalama mutlak hata (kW); **RMSE**: karekök ortalama karesel hata (kW), büyük hatalara daha duyarlı.
- **R²**: `1 − SSE/SST`; hedef varyansına göre açıklanan uyumu ölçer, boyutsuzdur.
- Household ve ELD arasında mutlak MAE kıyaslanmaz: tek hane ile 160 müşteri toplamının ölçekleri farklıdır.
  Bunun yerine MAPE*, R² ve aynı veri kümesindeki SN168'e göre MAE kazancı kullanılır.
"""
    )
    st.caption(
        "Protokol kaynakları: docs/final_test_protokolu.md, docs/eld_protokol.md, "
        "src/data/windows.py ve src/utils/metrics.py."
    )

    st.header("Teknik sınırlılıklar")
    st.markdown(
        """
- **ELD canlı veri:** `LD2011_2014.txt` ve `eld_hourly.parquet` yerelde yok; ELD test sonuçları ve bazı model checkpoint'leri kayıtlı.
- **XGBoost canlı inference:** eğitilmiş model dosyası yerelde yok; sayfa yalnız kayıtlı deneyin config/protokol bilgisini gösterir.
- **macOS/XGBoost:** XGBoost eğitim testi macOS'ta segmentation fault oluşturduğu bildirilmiştir. Bu sayfa testi/eğitimi çalıştırmaz;
  native OpenMP/runtime kök nedeni burada teşhis edilmemiştir.
- **Zaman birimi:** `src/data/preprocess.py` içindeki `DatetimeIndex.asi8` mesafesi `Timedelta.value` ile karşılaştırılıyor.
  Pandas datetime/timedelta iç gösterim birimleri aynı olmayabileceğinden bu bilinen çözülmemiş uyumsuzluktur.
  Bu sayfa ön işleme kodunu çağırmaz veya değiştirmez.
- **Canlı demo kapsamı:** Household Forecast, checkpoint'lerin gerçek feature/ölçek metadata'sını kullanır; kayıtlı sonuç tabloları
  ayrı test raporlarıdır. ELD sonuç tabloları canlı ELD tahmini bulunduğu anlamına gelmez.
"""
    )
