"""Household anomali demosunda deney protokolüyle uyumlu skor servisleri."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from app.services import demo_paths
from src.anomaly_protocol import scenario_settings
from src.data.inject_anomalies import sigma_for, train_sigma_table, inject_anomalies
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


@st.cache_data(show_spinner=False)
def load_anomaly_calibration(tables_dir):
    """Yalnız kayıtlı val-dev eşiklerini ve OOF/val kalibrasyon tablolarını yükler.

    Girdi: results/tables klasörünün mutlak yolu.
    Çıktı: sabit eşikler, smoothing pencereleri, train-OOF std'leri ve temiz
    val skor referans özeti.
    Neden: Test verisiyle eşik veya normalizasyon seçilmesini önler.
    """
    directory = Path(tables_dir)
    selection_path = directory / "anomaly_protocol_selection.csv"
    combination_path = directory / "anomaly_val_combination.csv"
    residual_paths = {
        "A3-s2s": directory / "A3-s2s_train_residual_std.csv",
        "A3-ens": directory / "A3-ens_train_residual_std.csv",
    }
    for path in (selection_path, combination_path, *residual_paths.values()):
        if not path.is_file():
            raise FileNotFoundError(f"Anomali protokolü kalibrasyon dosyası bulunamadı: {path}")

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
    for name, path in residual_paths.items():
        table = pd.read_csv(path)
        if set(("hour", "dow", "std")) - set(table.columns):
            raise ValueError(f"{path.name} hour/dow/std sütunlarını içermiyor.")
        expected_cells = {(hour, dow) for hour in range(24) for dow in range(7)}
        actual_cells = set(map(tuple, table[["hour", "dow"]].to_numpy()))
        if len(table) != 168 or actual_cells != expected_cells:
            raise ValueError(f"{path.name} tam 168 benzersiz saat × gün hücresi içermeli.")
        if not np.isfinite(table["std"].to_numpy(dtype=float)).all() or (table["std"] <= 0).any():
            raise ValueError(f"{path.name} geçersiz residual std değerleri içeriyor.")
        std_tables[name] = table

    combination = pd.read_csv(combination_path).set_index("component")
    required_references = ("A0", "A3-ens-norm")
    if not set(required_references).issubset(combination.index):
        raise ValueError("Val birleşim tablosunda A0 veya A3-ens-norm referansı eksik.")
    if not combination.loc[list(required_references), "reference"].astype(str).str.contains(
        "val enjeksiyonsuz", case=False
    ).all():
        raise ValueError("A5 referansları beklenen enjeksiyonsuz validation skorlarından değil.")
    for column in ("n_hours", "mean", "std"):
        if column not in combination:
            raise ValueError(f"A5 validation referans tablosunda {column} sütunu eksik.")
    if (combination.loc[list(required_references), "std"] <= 0).any():
        raise ValueError("A5 validation referans std değeri pozitif olmalı.")

    return {
        "thresholds": thresholds,
        "windows": windows,
        "std_tables": std_tables,
        "combination_summary": combination,
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


@st.cache_data(show_spinner=False)
def _validation_reference(
    frame,
    splits,
    config,
    std_tables,
    expected_summary,
    model_signature,
    _models,
):
    """A5 için temiz validation referans skor dizilerini deney koduyla aynı biçimde üretir."""
    train = splits["train"]
    val = splits["val"]
    context = pd.concat((train.tail(HISTORY_HOURS), val))
    target = config["preprocess"]["target"]
    predictions = {
        name: one_step_predictions(model, checkpoint, context, "cpu")
        for name, (model, checkpoint) in _models.items()
    }
    ensemble_prediction = combine_predictions(
        predictions, config["anomaly"]["residual"]["ensemble"]
    )
    error = {
        "A3-s2s": (context[target] - predictions["F10-opt"]).abs(),
        "A3-ens": (context[target] - ensemble_prediction).abs(),
    }
    a0 = rolling_zscore(
        context[target],
        config["anomaly"]["zscore_window"],
        config["anomaly"]["zscore_min_std"],
    ).loc[val.index]
    a3_ensemble = normalized_scores(error["A3-ens"], std_tables["A3-ens"]).loc[val.index]
    clean = val["is_imputed"].to_numpy() == 0
    references = {
        "A0": np.sort(a0.iloc[np.flatnonzero(clean)].dropna().to_numpy(dtype=np.float64)),
        "A3-ens-norm": np.sort(
            a3_ensemble.iloc[np.flatnonzero(clean)].dropna().to_numpy(dtype=np.float64)
        ),
    }
    for name, values in references.items():
        summary = expected_summary.loc[name]
        if len(values) != int(summary["n_hours"]):
            raise ValueError(
                f"{name} validation referansı kayıtlı özetle aynı sayıda saat içermiyor."
            )
        mean = values.mean()
        std = values.std(ddof=1)
        if not np.isclose(mean, summary["mean"], rtol=1e-4, atol=1e-7):
            raise ValueError(f"{name} validation referans ortalaması kayıtlı özetle uyuşmuyor.")
        if not np.isclose(std, summary["std"], rtol=1e-4, atol=1e-7):
            raise ValueError(f"{name} validation referans std'si kayıtlı özetle uyuşmuyor.")
        # Kayıtlı validation momentleri, farklı CPU çıkarımlarındaki küçük yuvarlama
        # farklarını kaldırır; sıralama/ampirik referans dağılımı aynen korunur.
        references[name] = np.sort(
            (values - mean) * (summary["std"] / std) + summary["mean"]
        )
    return references


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


@st.cache_data(show_spinner="Anomali skorları hesaplanıyor...")
def analyze_test_range(
    frame,
    config,
    start,
    end,
    inject,
    anomaly_type,
    scenario,
    seed,
    calibration,
    model_signature,
    _models,
):
    """Seçili test aralığında A0/A3/A5 skorlarını ve opsiyonel sentetik etiketleri üretir.

    Girdi: tam saatlik seri, config, test başlangıç/bitiş zamanları, enjeksiyon
    seçimi, severity senaryosu, test enjeksiyon seed'i, kayıtlı val kalibrasyonu
    ve cache'li train checkpoint'leri.
    Çıktı: aralıkla hizalı gerçek/enjekte seri, saat başı skorlar, sabit eşikler,
    geçerli saat maskesi, etiket tablosu ve alarm olayları.
    Neden: Enjeksiyon yalnız kopya test penceresinde, skorlar seçilen kopyada;
    validation eşikleri ve OOF residual ölçekleri değiştirilmeden kullanılır.
    """
    test = chronological_split(frame, config)["test"]
    start_time = pd.Timestamp(start)
    end_time = pd.Timestamp(end)
    if start_time not in test.index or end_time not in test.index or end_time < start_time:
        raise ValueError("Seçilen aralık kronolojik test saatleri içinde olmalı.")
    requested = pd.date_range(start_time, end_time, freq="h").as_unit(test.index.unit)
    if not requested.isin(test.index).all():
        raise ValueError("Seçilen test aralığında eksik saat var; eksiksiz aralık seçin.")
    selected = test.loc[requested].copy()
    target = config["preprocess"]["target"]
    if not np.isfinite(selected[target].to_numpy(dtype=np.float64)).all():
        raise ValueError("Seçilen gerçek tüketim aralığında eksik değer var.")
    history = _analysis_history(frame, start_time)

    if inject:
        settings = scenario_settings(config, SCENARIO_SEEDS[scenario])
        sigma_table = train_sigma_table(
            chronological_split(frame, config)["train"], target
        )
        sigma = sigma_for(selected, sigma_table)
        if not np.isfinite(sigma).all():
            raise ValueError("Train σ tablosunda seçilen aralık için değer eksik.")
        injected, labels = inject_anomalies(
            selected,
            target,
            settings,
            seed=seed,
            sigma=sigma,
            anomaly_types=(anomaly_type,),
        )
        if not labels["is_anomaly"].any():
            raise RuntimeError("Seçilen aralıkta enjeksiyon üretilemedi.")
    else:
        injected = selected.copy()
        labels = pd.DataFrame(
            {"is_anomaly": np.zeros(len(selected), dtype="int8"),
             "anomaly_type": np.full(len(selected), "none", dtype=object)},
            index=selected.index,
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
        full[target],
        config["anomaly"]["zscore_window"],
        config["anomaly"]["zscore_min_std"],
    )
    raw_scores["A3-s2s-norm"] = normalized_scores(
        error_seq2seq, calibration["std_tables"]["A3-s2s"]
    )
    raw_scores["A3-ens-norm"] = normalized_scores(
        error_ensemble, calibration["std_tables"]["A3-ens"]
    )

    splits = chronological_split(frame, config)
    references = _validation_reference(
        frame,
        splits,
        config,
        calibration["std_tables"],
        calibration["combination_summary"],
        model_signature,
        _models,
    )
    combination = {
        "components": ["A0", "A3-ens-norm"],
        "references": references,
    }
    add_combinations(raw_scores, combination)

    selected_scores = pd.DataFrame(index=requested)
    for method in METHODS:
        selected_scores[method] = smooth_scores(
            raw_scores[[method]], calibration["windows"][method]
        )[method].loc[requested]

    selected_injected = injected.loc[requested]
    valid = (
        selected_scores.notna().all(axis=1).to_numpy()
        & (selected_injected["is_imputed"].to_numpy() == 0)
        & np.isfinite(selected_injected[target].to_numpy(dtype=np.float64))
    )
    if not valid.any():
        raise ValueError("Seçili aralıkta skorlanabilir imputasyonsuz test saati yok.")
    selected_scores = selected_scores.loc[requested]
    events = _event_table(selected_scores, calibration["thresholds"], valid, labels)
    return {
        "real_kw": selected[target].astype(float),
        "imputed": selected["is_imputed"].astype(bool),
        "injected_kw": selected_injected[target].astype(float),
        "scores": selected_scores,
        "thresholds": calibration["thresholds"],
        "windows": calibration["windows"],
        "valid": pd.Series(valid, index=requested),
        "labels": labels.loc[requested],
        "events": events,
        "seed": seed if inject else None,
        "scenario": scenario if inject else None,
    }


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


def validate_injection_range(start, end, anomaly_type, scenario_settings_value):
    """Protokol hedef/max oranlarıyla seçili tip için yeterli saat var mı denetler."""
    start_time = pd.Timestamp(start)
    end_time = pd.Timestamp(end)
    hours = int((end_time - start_time) / pd.Timedelta(hours=1)) + 1
    minimum_segment = (
        len(scenario_settings_value["pattern"]["night_hours"])
        if anomaly_type == "pattern"
        else scenario_settings_value[anomaly_type]["min_len"]
    )
    minimum_hours = int(np.ceil(minimum_segment / scenario_settings_value["max_ratio"]))
    if hours < minimum_hours:
        minimum_days = int(np.ceil(minimum_hours / 24))
        raise ValueError(
            f"{anomaly_type} enjeksiyonu için mevcut max_ratio={scenario_settings_value['max_ratio']:.3f} "
            f"kuralıyla en az {minimum_days} tam gün seçilmeli."
        )
