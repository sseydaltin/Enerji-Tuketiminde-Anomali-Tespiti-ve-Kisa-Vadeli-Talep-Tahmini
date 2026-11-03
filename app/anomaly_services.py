"""Household anomali demosunda deney protokolüyle uyumlu skor servisleri."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from app.build_calibration import REFERENCE_FILE, STD_FILE
from app.services import demo_paths
from app.ui import ANOMALY_LABELS
from src.anomaly_protocol import scenario_settings
from src.data.inject_anomalies import ANOMALY_TYPES, inject_anomalies, sigma_for, train_sigma_table
from src.data.split import chronological_split
from src.models.anomaly_baselines import rolling_zscore
from src.models.residual_detector import (
    combine_predictions,
    normalized_scores,
    one_step_predictions,
)
from src.utils.anomaly_metrics import runs
from src.utils.results_io import resolve_path
from src.evaluate_anomaly import add_combinations, smooth_scores

METHODS = ("A0", "A3-s2s-norm", "A5-mean-z")
SCENARIO_SEEDS = {"main": "main", "hard": "hard"}
HISTORY_HOURS = 192
A5_COMPONENTS = ("A0", "A3-ens-norm")
SEVERITY_TYPES = ("spike", "level_shift")   # büyüklüğü ayarlanabilen tipler
CAUGHT = "Yakalandı"
MISSED = "Kaçırıldı"


@st.cache_data(show_spinner=False)
def load_anomaly_calibration(tables_dir):
    """Kayıtlı doğrulama eşiklerini ve protokol kalibrasyonunu yükler.

    Girdi: results/tables klasörünün mutlak yolu.
    Çıktı: sabit eşikler, yumuşatma pencereleri, eğitim OOF std tabloları ve A5'in
    enjeksiyonsuz val-dev referans skorları.
    Neden: Eşikler bu ölçeklerle seçildi. A3-*_train_residual_std.csv dosyaları eski
    aşamanın in-sample tablolarıdır; onlarla skor olduğundan küçük çıkar ve eşik
    anlamını yitirir. Test verisiyle eşik veya normalizasyon seçilmez.
    """
    directory = Path(tables_dir)
    selection_path = directory / "anomaly_protocol_selection.csv"
    reference_path = directory / REFERENCE_FILE
    std_paths = {name: directory / STD_FILE.format(name=name) for name in ("A3-s2s", "A3-ens")}
    for path in (selection_path, reference_path, *std_paths.values()):
        if not path.is_file():
            raise FileNotFoundError(
                f"Anomali kalibrasyon dosyası bulunamadı: {path}. Üretmek için: python -m app.build_calibration"
            )

    selection = pd.read_csv(selection_path)
    if "final" not in selection or "method" not in selection:
        raise ValueError(f"{selection_path.name} gerekli final/method sütunlarını içermiyor.")
    if selection["final"].dtype != bool:
        selection["final"] = selection["final"].astype(str).str.lower().map(
            {"true": True, "false": False}
        )
    expected_final = selection.loc[selection["final"] == True, "method"].tolist()
    if expected_final != ["A3-s2s-norm"]:
        raise ValueError(
            "Kayıtlı validation seçimi beklenen final yöntem A3-s2s-norm ile uyuşmuyor."
        )
    selected = selection.set_index("method")
    missing_methods = sorted(set(METHODS) - set(selected.index))
    if missing_methods:
        raise ValueError(f"Validation seçiminde yöntem eksik: {missing_methods}")

    thresholds = selected.loc[list(METHODS), "threshold"].astype(float).to_dict()
    windows = selected.loc[list(METHODS), "smooth"].astype(int).to_dict()
    if not np.isfinite(list(thresholds.values())).all() or min(windows.values()) < 1:
        raise ValueError("Validation eşikleri veya smoothing pencereleri geçersiz.")

    std_tables = {}
    expected_cells = {(hour, dow) for hour in range(24) for dow in range(7)}
    for name, path in std_paths.items():
        table = pd.read_csv(path)
        if set(("hour", "dow", "std")) - set(table.columns):
            raise ValueError(f"{path.name} hour/dow/std sütunlarını içermiyor.")
        actual_cells = set(map(tuple, table[["hour", "dow"]].to_numpy()))
        if len(table) != 168 or actual_cells != expected_cells:
            raise ValueError(f"{path.name} tam 168 benzersiz saat × gün hücresi içermeli.")
        if not np.isfinite(table["std"].to_numpy(dtype=float)).all() or (table["std"] <= 0).any():
            raise ValueError(f"{path.name} geçersiz residual std değerleri içeriyor.")
        std_tables[name] = table

    reference = pd.read_csv(reference_path)
    references = {
        name: np.sort(reference.loc[reference["component"] == name, "score"].to_numpy(dtype=np.float64))
        for name in A5_COMPONENTS
    }
    if min(len(values) for values in references.values()) == 0:
        raise ValueError(f"{reference_path.name} A5 bileşenlerinin referansını içermiyor.")

    return {
        "thresholds": thresholds,
        "windows": windows,
        "std_tables": std_tables,
        "references": references,
    }


@st.cache_data(show_spinner=False)
def complete_test_days(frame, config):
    """Test kümesinde tam 24 satırlı günleri döndürür; imputasyon dışlama sonradır."""
    test = chronological_split(frame, config)["test"]
    days = []
    for day in pd.date_range(test.index.min().normalize(), test.index.max().normalize(), freq="D"):
        hours = pd.date_range(day, periods=24, freq="h")
        if hours.isin(test.index).all():
            days.append(day.date())
    if not days:
        raise ValueError("Test döneminde tam saatlik gün bulunamadı.")
    return days


def _analysis_history(frame, start):
    """Seçili test saatlerinden önceki 192 gerçek gözlemi sızıntısız bağlam olarak alır."""
    history = frame.loc[frame.index < start].tail(HISTORY_HOURS)
    if len(history) != HISTORY_HOURS:
        raise ValueError("Skor bağlamı için seçili aralıktan önce 192 saat veri gerekli.")
    expected = pd.date_range(history.index[0], periods=HISTORY_HOURS, freq="h")
    expected = expected.as_unit(history.index.unit)
    if not history.index.equals(expected):
        raise ValueError("Skor bağlamı saatlik ve kesintisiz değil.")
    return history


def _event_table(scores, thresholds, valid, labels):
    """Eşik üstü ardışık alarm bloklarını başlangıç/bitiş ve skorla özetler."""
    rows = []
    for method in METHODS:
        detected = (scores[method] >= thresholds[method]).fillna(False).to_numpy() & valid
        values = scores[method].to_numpy(dtype=np.float64)
        for start, end in runs(detected):
            event_scores = values[start:end]
            rows.append({
                "Başlangıç": scores.index[start],
                "Bitiş": scores.index[end - 1],
                "Yöntem": method,
                "En yüksek skor": float(np.nanmax(event_scores)),
                "Eşik": float(thresholds[method]),
                "Süre (saat)": end - start,
                "Sentetik olayla örtüşüyor": bool(labels["is_anomaly"].iloc[start:end].any()),
            })
    return pd.DataFrame(rows, columns=[
        "Başlangıç", "Bitiş", "Yöntem", "En yüksek skor", "Eşik",
        "Süre (saat)", "Sentetik olayla örtüşüyor",
    ])


def scale_injection(settings, severity):
    """Ani yükselme ve seviye kaymasının büyüklüğünü demo kaydırıcısıyla ölçekler.

    Girdi: senaryo ayarları, şiddet çarpanı (1,0 = protokolün kendisi).
    Çıktı: ayarların kopyası. σ katsayıları çarpılır; "×f" biçimindeki katsayılarda
    f'in 1'den farkı çarpılır, çünkü ×1 hiç değişiklik demektir. Düşüş ve gece
    örüntüsünün ayarlanacak büyüklüğü yoktur; o tipler değişmez.
    """
    scaled = deepcopy(settings)
    for kind in SEVERITY_TYPES:
        spec = scaled[kind]
        for key in ("sigma_low", "sigma_high"):
            if key in spec:
                spec[key] = spec[key] * severity
        for key in ("factor", "factor_low", "factor_high"):
            if key in spec:
                spec[key] = 1.0 + (spec[key] - 1.0) * severity
    return scaled


@st.cache_data(show_spinner="Anomali skorları hesaplanıyor...")
def _score_whole_test(
    frame, config, inject, anomaly_type, scenario, seed, severity, calibration, model_signature, _models,
):
    """Tüm test kümesini (isteğe bağlı enjeksiyonlu kopyasıyla) bir kez puanlar.

    Çıktı: enjekte kopya, etiketler, ham ve yumuşatılmış skorlar (tüm test saatleri).
    Neden: Protokolde enjeksiyon kümenin tamamına yapılır. Kısa aralığa yapılırsa
    oran sınırı (max_ratio) 12–48 saatlik seviye kaymasına yer bırakmaz ve olaylar
    tarih değiştikçe yer değiştirir. Tarih aralığı burada yalnız görünümü seçer.
    """
    splits = chronological_split(frame, config)
    test = splits["test"]
    target = config["preprocess"]["target"]
    history = _analysis_history(frame, test.index[0])
    if inject:
        settings = scale_injection(scenario_settings(config, SCENARIO_SEEDS[scenario]), severity)
        sigma = sigma_for(test, train_sigma_table(splits["train"], target))
        if not np.isfinite(sigma).all():
            raise ValueError("Train σ tablosunda test saatleri için değer eksik.")
        injected, labels = inject_anomalies(
            test, target, settings, seed=seed, sigma=sigma, anomaly_types=(anomaly_type,),
        )
    else:
        injected = test.copy()
        labels = pd.DataFrame(
            {"is_anomaly": np.zeros(len(test), dtype="int8"),
             "anomaly_type": np.full(len(test), "none", dtype=object)},
            index=test.index,
        )

    full = pd.concat((history, injected))
    if not full.index.is_monotonic_increasing or full.index.has_duplicates:
        raise ValueError("Skor girdisi saat sırasını veya benzersizliği sağlamıyor.")
    predictions = {
        name: one_step_predictions(model, checkpoint, full, "cpu")
        for name, (model, checkpoint) in _models.items()
    }
    error_seq2seq = (full[target] - predictions["F10-opt"]).abs()
    error_ensemble = (
        full[target]
        - combine_predictions(predictions, config["anomaly"]["residual"]["ensemble"])
    ).abs()
    raw_scores = pd.DataFrame(index=full.index)
    raw_scores["A0"] = rolling_zscore(
        full[target], config["anomaly"]["zscore_window"], config["anomaly"]["zscore_min_std"],
    )
    raw_scores["A3-s2s-norm"] = normalized_scores(error_seq2seq, calibration["std_tables"]["A3-s2s"])
    raw_scores["A3-ens-norm"] = normalized_scores(error_ensemble, calibration["std_tables"]["A3-ens"])
    add_combinations(raw_scores, {"components": list(A5_COMPONENTS), "references": calibration["references"]})

    smoothed = pd.DataFrame(index=test.index)
    for method in METHODS:
        smoothed[method] = smooth_scores(
            raw_scores[[method]], calibration["windows"][method]
        )[method].loc[test.index]
    return {
        "injected": injected,
        "labels": labels,
        "raw": raw_scores.loc[test.index, list(METHODS)],
        "smoothed": smoothed,
    }


def analyze_test_range(
    frame, config, start, end, inject, anomaly_type, scenario, seed, calibration,
    model_signature, _models, severity=1.0,
):
    """Tüm testin skorlarından seçili aralığı dilimler; olay tablosunu kurar.

    Girdi: tam saatlik seri, config, görünüm başlangıç/bitişi, enjeksiyon seçimi,
    senaryo, seed, kayıtlı doğrulama kalibrasyonu, ağırlıklar, şiddet çarpanı.
    Çıktı: aralıkla hizalı gerçek/enjekte seri, ham ve yumuşatılmış skorlar (eşik
    yumuşatılmış skora uygulanır), sabit eşikler, geçerli saat maskesi, etiketler, alarmlar.
    Neden: Enjeksiyon yalnız kopyaya yapılır; eşik ve artık ölçekleri değişmez.
    """
    test = chronological_split(frame, config)["test"]
    start_time = pd.Timestamp(start)
    end_time = pd.Timestamp(end)
    if start_time not in test.index or end_time not in test.index or end_time < start_time:
        raise ValueError("Seçilen aralık kronolojik test saatleri içinde olmalı.")
    requested = pd.date_range(start_time, end_time, freq="h").as_unit(test.index.unit)
    if not requested.isin(test.index).all():
        raise ValueError("Seçilen test aralığında eksik saat var; eksiksiz aralık seçin.")
    target = config["preprocess"]["target"]
    selected = test.loc[requested]
    if not np.isfinite(selected[target].to_numpy(dtype=np.float64)).all():
        raise ValueError("Seçilen gerçek tüketim aralığında eksik değer var.")

    scored = _score_whole_test(
        frame, config, inject, anomaly_type, scenario, seed, severity, calibration,
        model_signature, _models,
    )
    selected_scores = scored["smoothed"].loc[requested]
    selected_injected = scored["injected"].loc[requested]
    labels = scored["labels"].loc[requested]
    valid = (
        selected_scores.notna().all(axis=1).to_numpy()
        & (selected_injected["is_imputed"].to_numpy() == 0)
        & np.isfinite(selected_injected[target].to_numpy(dtype=np.float64))
    )
    if not valid.any():
        raise ValueError("Seçili aralıkta skorlanabilir imputasyonsuz test saati yok.")
    return {
        "real_kw": selected[target].astype(float),
        "imputed": selected["is_imputed"].astype(bool),
        "injected_kw": selected_injected[target].astype(float),
        "scores": selected_scores,
        "raw_scores": scored["raw"].loc[requested],
        "thresholds": calibration["thresholds"],
        "windows": calibration["windows"],
        "valid": pd.Series(valid, index=requested),
        "labels": labels,
        "events": _event_table(selected_scores, calibration["thresholds"], valid, labels),
        "seed": seed if inject else None,
        "scenario": scenario if inject else None,
    }


def flagged_hours(result, method):
    """Eşiği aşan geçerli saatleri [N] bool dizisi verir (kural: yumuşatılmış skor >= eşik)."""
    scores = result["scores"][method]
    return (result["valid"] & scores.ge(result["thresholds"][method])).to_numpy()


def event_outcomes(result, method):
    """Enjekte olayları yakalandı/kaçırıldı diye sınıflar, yanlış alarmları sayar.

    Girdi: analyze_test_range çıktısı ve yöntem kodu.
    Çıktı: (olay tablosu, yanlış alarm blok sayısı).
    Neden: Tanımlar src/utils/anomaly_metrics.event_metrics ile aynıdır. Olayın en az
    bir saati işaretlenince yakalanmış sayılır; hiçbir etiketli saatle kesişmeyen alarm
    bloğu yanlış alarmdır. Ham skor yalnız bilgi içindir; eşik yumuşatılmış skora uygulanır.
    """
    types = result["labels"]["anomaly_type"].to_numpy()
    flagged = flagged_hours(result, method)
    smoothed = result["scores"][method]
    raw = result["raw_scores"][method]
    index = smoothed.index
    lag = result["windows"][method] - 1              # yumuşatmanın olaydan sonra taşıdığı saat
    rows = []
    for kind in ANOMALY_TYPES:
        for start, end in runs(types == kind):
            rows.append({
                "Başlangıç": index[start],
                "Bitiş": index[end - 1],
                "Tür": ANOMALY_LABELS[kind],
                "Süre (saat)": end - start,
                "Ham skor (en yüksek)": float(raw.iloc[start:end].max()),
                "Yumuşatılmış skor (en yüksek)": float(smoothed.iloc[start:end].max()),
                "Durum": CAUGHT if flagged[start:end].any() else MISSED,
                "Olay bitince alarm": "Evet" if flagged[end:end + lag].any() else "Hayır",
            })
    columns = ["Başlangıç", "Bitiş", "Tür", "Süre (saat)", "Ham skor (en yüksek)",
               "Yumuşatılmış skor (en yüksek)", "Durum", "Olay bitince alarm"]
    table = pd.DataFrame(rows, columns=columns).sort_values("Başlangıç").reset_index(drop=True)
    anomalous = types != "none"
    false_alarms = sum(not anomalous[start:end].any() for start, end in runs(flagged))
    return table, false_alarms


def checkpoint_signature(models_dir):
    """F10/F5 checkpoint dosyalarının cache anahtarı için boyut ve değişim zamanını verir."""
    signatures = []
    for name in ("F10-opt_best.pt", "F5-opt_best.pt"):
        path = Path(models_dir) / name
        if not path.is_file():
            raise FileNotFoundError(f"Gerekli model checkpoint'i bulunamadı: {path}")
        stat = path.stat()
        signatures.append((name, stat.st_size, stat.st_mtime_ns))
    return tuple(signatures)


def anomaly_demo_paths():
    """Household tablo, checkpoint, sonuç tablo yollarını config'ten çözer."""
    config, data_path, models_dir, expected_features = demo_paths()
    project_root = Path(__file__).resolve().parents[1]
    return (
        config,
        data_path,
        models_dir,
        resolve_path(project_root, config["paths"]["tables_dir"]),
        expected_features,
    )
