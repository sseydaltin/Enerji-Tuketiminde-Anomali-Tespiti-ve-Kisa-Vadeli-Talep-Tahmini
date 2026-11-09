"""Household anomali demosunda deney protokolüyle uyumlu skor servisleri."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from app.build_calibration import REFERENCE_FILE, STD_FILE
from app.services import demo_paths
from app.ui import ANOMALY_LABELS
from src.anomaly_dual import BASE_METHOD, DUAL, DUAL_FLOOR, SELECTION_FILE, dual_alarm, floor_alarm
from src.anomaly_protocol import protocol_splits, scenario_settings
from src.anomaly_v2 import V2_METHOD, train_floor, v2_score
from src.data.inject_anomalies import ANOMALY_TYPES, inject_anomalies, sigma_for, train_sigma_table
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
EXPERIMENTAL_METHODS = (DUAL, DUAL_FLOOR)   # "Geliştirilmiş" seçenekleri; final sonuç değildir
EXPERIMENT_SPLIT = "val_hold"       # deneysel yöntem test kümesinde çalıştırılmaz
V2_METHODS = (V2_METHOD,)           # A3 + taban (v2): test sonrası, final değil; testte A3 ile yan yana
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
        "experimental": _load_experimental(directory, thresholds),
    }


def _load_experimental(directory, thresholds):
    """İki A6 varyantının kayıtlı ayarlarını okur; deney çalıştırılmadıysa None döndürür.

    Çıktı: {yöntem: {t1, t2, k, floor_kw}}. A6-dual taban kuralı kullanmadığı için
    k ve floor_kw onda None olur. Burada hiçbir ayar seçilmez; değerler val-dev'de seçilmişti.
    """
    path = directory / SELECTION_FILE
    if not path.is_file():
        return None
    selection = pd.read_csv(path).set_index("method")
    settings = {}
    for method in EXPERIMENTAL_METHODS:
        row = selection.loc[method]
        uses_floor = method == DUAL_FLOOR
        settings[method] = {
            "t1": float(thresholds[BASE_METHOD]),
            "t2": float(row["t2"]),
            "k": int(row["k"]) if uses_floor else None,
            "floor_kw": float(row["floor_kw"]) if uses_floor else None,
        }
    return settings


@st.cache_data(show_spinner=False)
def complete_test_days(frame, config, split="test"):
    """Seçili kümede tam 24 satırlı günleri döndürür; imputasyon dışlama sonradır."""
    test = protocol_splits(frame, config)[split]
    days = []
    for day in pd.date_range(test.index.min().normalize(), test.index.max().normalize(), freq="D"):
        hours = pd.date_range(day, periods=24, freq="h")
        if hours.isin(test.index).all():
            days.append(day.date())
    if not days:
        raise ValueError("Seçili dönemde tam saatlik gün bulunamadı.")
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


def _event_table(scores, thresholds, valid, labels, flags=None):
    """Eşik üstü ardışık alarm bloklarını başlangıç/bitiş ve skorla özetler.

    flags: {yöntem: [N] bool}; kuralı eşikten ibaret olmayan yöntemin alarmları.
    """
    flags = flags or {}
    rows = []
    for method in scores.columns:
        if method in flags:
            detected = flags[method] & valid
        else:
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
    split="test",
):
    """Tüm kümeyi (isteğe bağlı enjeksiyonlu kopyasıyla) bir kez puanlar.

    Çıktı: enjekte kopya, etiketler, ham ve yumuşatılmış skorlar (tüm test saatleri).
    Neden: Protokolde enjeksiyon kümenin tamamına yapılır. Kısa aralığa yapılırsa
    oran sınırı (max_ratio) 12–48 saatlik seviye kaymasına yer bırakmaz ve olaylar
    tarih değiştikçe yer değiştirir. Tarih aralığı burada yalnız görünümü seçer.
    """
    splits = protocol_splits(frame, config)
    test = splits[split]
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


def experimental_flags(scored, settings, target):
    """Bir A6 varyantının alarm saatlerini kümenin tamamında hesaplar.

    Girdi: _score_whole_test çıktısı, varyant ayarları, hedef sütun adı.
    Çıktı: küme saatleriyle hizalı [N] bool seri. Doldurulmuş saat maskesi burada uygulanmaz.
    Neden: Taban sayacı görünüm aralığının başında sıfırlanmasın diye tüm kümede hesaplanır.
    """
    rule = dual_alarm(scored["smoothed"][BASE_METHOD], scored["raw"][BASE_METHOD], settings["t1"], settings["t2"])
    if settings["k"] is not None:
        rule = rule | floor_alarm(scored["injected"][target].to_numpy(), settings["floor_kw"], settings["k"])
    return pd.Series(rule, index=scored["injected"].index)


def analyze_test_range(
    frame, config, start, end, inject, anomaly_type, scenario, seed, calibration,
    model_signature, _models, severity=1.0, split="test", v2=False,
):
    """Kümenin skorlarından seçili aralığı dilimler; olay tablosunu kurar.

    Girdi: tam saatlik seri, config, görünüm başlangıç/bitişi, enjeksiyon seçimi,
    senaryo, seed, kayıtlı doğrulama kalibrasyonu, ağırlıklar, şiddet çarpanı, küme.
    Çıktı: aralıkla hizalı gerçek/enjekte seri, ham ve yumuşatılmış skorlar (eşik
    yumuşatılmış skora uygulanır), sabit eşikler, geçerli saat maskesi, etiketler, alarmlar.
    Neden: Enjeksiyon yalnız kopyaya yapılır; eşik ve artık ölçekleri değişmez.
    v2=True (yalnız test): "A3 + taban (v2)" alarmları src.anomaly_v2.v2_score ile, deney betiğiyle
    aynı kuralla eklenir; taban yalnız train'den hesaplanır, hiçbir ayar seçilmez.
    """
    splits = protocol_splits(frame, config)
    test = splits[split]
    start_time = pd.Timestamp(start)
    end_time = pd.Timestamp(end)
    if start_time not in test.index or end_time not in test.index or end_time < start_time:
        raise ValueError("Seçilen aralık kronolojik küme saatleri içinde olmalı.")
    requested = pd.date_range(start_time, end_time, freq="h").as_unit(test.index.unit)
    if not requested.isin(test.index).all():
        raise ValueError("Seçilen aralıkta eksik saat var; eksiksiz aralık seçin.")
    target = config["preprocess"]["target"]
    selected = test.loc[requested]
    if not np.isfinite(selected[target].to_numpy(dtype=np.float64)).all():
        raise ValueError("Seçilen gerçek tüketim aralığında eksik değer var.")

    scored = _score_whole_test(
        frame, config, inject, anomaly_type, scenario, seed, severity, calibration,
        model_signature, _models, split,
    )
    selected_scores = scored["smoothed"].loc[requested]
    selected_raw = scored["raw"].loc[requested]
    selected_injected = scored["injected"].loc[requested]
    labels = scored["labels"].loc[requested]
    valid = (
        selected_scores.notna().all(axis=1).to_numpy()
        & (selected_injected["is_imputed"].to_numpy() == 0)
        & np.isfinite(selected_injected[target].to_numpy(dtype=np.float64))
    )
    if not valid.any():
        raise ValueError("Seçili aralıkta skorlanabilir imputasyonsuz saat yok.")

    thresholds = dict(calibration["thresholds"])
    windows = dict(calibration["windows"])
    flags = {}
    experimental = None
    if split == EXPERIMENT_SPLIT and calibration.get("experimental"):
        experimental = calibration["experimental"]
        for method, settings in experimental.items():
            # Deneysel kurallar final yöntemin skorları üzerine kuruludur; skor sütunları onundur.
            selected_scores = selected_scores.assign(**{method: selected_scores[BASE_METHOD]})
            selected_raw = selected_raw.assign(**{method: selected_raw[BASE_METHOD]})
            thresholds[method] = settings["t1"]
            windows[method] = windows[BASE_METHOD]
            flags[method] = experimental_flags(scored, settings, target).loc[requested].to_numpy()
    floor_kw = None
    if v2 and split == "test":
        floor_kw = train_floor(splits["train"], target)
        injected = scored["injected"]
        # Bütün kümede hesaplanır: aralık başındaki saat de bir önceki saate bakabilsin.
        v2_values, _ = v2_score(scored["smoothed"][BASE_METHOD].to_numpy(dtype=float), thresholds[BASE_METHOD],
                                injected[target].to_numpy(), floor_kw, injected.index,
                                injected["is_imputed"].to_numpy() == 1)
        selected_scores = selected_scores.assign(**{V2_METHOD: selected_scores[BASE_METHOD]})
        selected_raw = selected_raw.assign(**{V2_METHOD: selected_raw[BASE_METHOD]})
        thresholds[V2_METHOD] = thresholds[BASE_METHOD]
        windows[V2_METHOD] = windows[BASE_METHOD]
        flags[V2_METHOD] = pd.Series(v2_values >= 1.0, index=injected.index).loc[requested].to_numpy()

    return {
        "real_kw": selected[target].astype(float),
        "imputed": selected["is_imputed"].astype(bool),
        "injected_kw": selected_injected[target].astype(float),
        "scores": selected_scores,
        "raw_scores": selected_raw,
        "thresholds": thresholds,
        "windows": windows,
        "valid": pd.Series(valid, index=requested),
        "labels": labels,
        "events": _event_table(selected_scores, thresholds, valid, labels, flags),
        "seed": seed if inject else None,
        "scenario": scenario if inject else None,
        "split": split,
        "experimental": experimental,
        "flags": flags,
        "floor_kw": floor_kw,
    }


def flagged_hours(result, method):
    """Eşiği aşan geçerli saatleri [N] bool dizisi verir (kural: yumuşatılmış skor >= eşik)."""
    if method in result.get("flags", {}):
        return result["flags"][method] & result["valid"].to_numpy()
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
