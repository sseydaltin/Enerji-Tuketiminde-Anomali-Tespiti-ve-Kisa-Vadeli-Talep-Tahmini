"""Kayıtlı final tahmin ve anomali sonuçlarını salt okunur hazırlayan servisler."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.data.eld import eld_config
from src.data.load import PROJECT_ROOT, load_config

FORECAST_MODELS = ("F13-ens", "XGBoost", "SN168")
ANOMALY_METHODS = ("A0", "A3-s2s-norm", "A5-mean-z")
MODEL_LABELS = {
    "F13-ens": "F13 ensemble (0,8 F10-opt + 0,2 F5-opt)",
    "XGBoost": "F12-L336 XGBoost",
    "SN168": "Seasonal Naive 168",
}
SOURCE_MODEL_LABELS = {
    "F13-ens": "F13-ens (final)",
    "XGBoost": "F12-L336 XGBoost",
    "SN168": "F2 Seasonal Naive 168",
}
FORECAST_IDS = {
    "household": {
        "F13-ens": ("F13-ens", "F13-ens-s7", "F13-ens-s2026"),
        "XGBoost": ("F12-L336",),
        "SN168": ("F2",),
    },
    "eld": {
        "F13-ens": ("ELD-F13-ens", "ELD-F13-ens-s7", "ELD-F13-ens-s2026"),
        "XGBoost": ("ELD-F12-L336",),
        "SN168": ("ELD-F2",),
    },
}
ANOMALY_FILES = {
    "household": ("anomaly_protocol_test.csv", "anomaly_protocol_test_per_seed.csv"),
    "eld": ("anomaly_test.csv", "anomaly_test_per_seed.csv"),
}
FORECAST_COLUMNS = ("exp_id", "split", "model", "MAE", "RMSE", "MAPE*", "R2")
HORIZON_COLUMNS = ("exp_id", "model", "horizon", "MAE")
ANOMALY_COLUMNS = (
    "split", "scenario", "method", "rule", "n_seeds",
    "pr_auc_mean", "pr_auc_std", "event_f1_mean", "event_f1_std",
)
ANOMALY_TYPE_COLUMNS = tuple(
    f"event_recall_{kind}_{stat}"
    for kind in ("spike", "drop", "level_shift", "pattern")
    for stat in ("mean", "std")
)
ANOMALY_SEED_COLUMNS = (
    "split", "scenario", "seed", "method", "rule",
    "pr_auc", "event_f1",
    "event_recall_spike", "event_recall_drop",
    "event_recall_level_shift", "event_recall_pattern",
)
DISPLAY_ANOMALY_COLUMNS = {
    "pr_auc_mean": "PR-AUC",
    "event_f1_mean": "Olay F1",
    "event_recall_spike_mean": "Spike olay recall",
    "event_recall_drop_mean": "Drop olay recall",
    "event_recall_level_shift_mean": "Seviye kayması olay recall",
    "event_recall_pattern_mean": "Gece örüntüsü olay recall",
}


def _require_columns(frame, required, path):
    """Gerekli kayıtlı sonuç sütunlarının bulunduğunu açık hata ile doğrular."""
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} gerekli sütunları içermiyor: {missing}")


def _read_csv(path, required):
    """CSV dosyasını okur, gerekli sütunları doğrular ve bozuk dosyayı açıkça bildirir."""
    if not path.is_file():
        raise FileNotFoundError(f"Kayıtlı sonuç dosyası bulunamadı: {path}")
    frame = pd.read_csv(path)
    _require_columns(frame, required, path)
    if frame.empty:
        raise ValueError(f"Kayıtlı sonuç dosyası boş: {path}")
    return frame


def _numeric_or_missing(values, column, path):
    """Sütunu sayıya çevirir; metin biçimindeki hatalı değerleri sessizce kabul etmez."""
    try:
        return pd.to_numeric(values, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} içindeki {column} sütunu sayısal değil.") from error


def _mean_and_std(values):
    """Kayıtlı koşuların ortalamasını ve varsa örnek standart sapmasını döndürür."""
    numbers = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    if len(numbers) == 0 or not np.isfinite(numbers).all():
        return np.nan, np.nan
    mean = float(numbers.mean())
    std = float(numbers.std(ddof=1)) if len(numbers) > 1 else np.nan
    return mean, std


def _forecast_summary(dataset, forecast, path):
    """Doğru veri kümesinin test koşularından modelleri ve seed özetini hazırlar."""
    test = forecast.loc[forecast["split"] == "test"].copy()
    if test.empty:
        raise ValueError(f"{path} içinde split=test sonucu bulunamadı.")
    test["exp_id"] = test["exp_id"].astype(str)
    test["MAE"] = _numeric_or_missing(test["MAE"], "MAE", path)
    rows = []
    selected_ids = set()
    for model_key in FORECAST_MODELS:
        exp_ids = FORECAST_IDS[dataset][model_key]
        selected_ids.update(exp_ids)
        runs = test.loc[test["exp_id"].isin(exp_ids)]
        if set(runs["exp_id"]) != set(exp_ids):
            missing = sorted(set(exp_ids) - set(runs["exp_id"]))
            raise ValueError(f"{path} içinde {dataset}/{model_key} için koşu eksik: {missing}")
        if runs["exp_id"].duplicated().any():
            raise ValueError(f"{path} içinde {dataset}/{model_key} deney kimliği yineleniyor.")
        expected_label = SOURCE_MODEL_LABELS[model_key]
        if set(runs["model"].astype(str)) != {expected_label}:
            raise ValueError(
                f"{path} {dataset}/{model_key} model etiketi beklenen {expected_label!r} değil."
            )

        row = {
            "dataset": dataset,
            "model_key": model_key,
            "model_label": MODEL_LABELS[model_key],
            "n_seeds": len(runs),
            "source_exp_ids": tuple(runs["exp_id"]),
        }
        for column in ("MAE", "RMSE", "MAPE*", "R2"):
            runs[column] = _numeric_or_missing(runs[column], column, path)
            row[column], row[f"{column}_std"] = _mean_and_std(runs[column])
        rows.append(row)

    summary = pd.DataFrame(rows)
    baseline_mae = summary.loc[summary["model_key"] == "SN168", "MAE"].iloc[0]
    if not np.isfinite(baseline_mae) or baseline_mae <= 0:
        raise ValueError(f"{path} içinde SN168 MAE baseline'ı eksik veya geçersiz.")
    summary["gain_vs_sn168_pct"] = (
        100.0 * (baseline_mae - summary["MAE"]) / baseline_mae
    )
    sn_runs = test.loc[test["exp_id"] == FORECAST_IDS[dataset]["SN168"][0]]
    for index, row in summary.iterrows():
        runs = test.loc[test["exp_id"].isin(row["source_exp_ids"])]
        if len(runs) <= 1:
            summary.loc[index, "gain_vs_sn168_pct_std"] = np.nan
            continue
        gains = 100.0 * (sn_runs["MAE"].iloc[0] - runs["MAE"]) / sn_runs["MAE"].iloc[0]
        summary.loc[index, "gain_vs_sn168_pct_std"] = gains.std(ddof=1)
    return summary, selected_ids


def _horizon_summary(dataset, horizon, expected_ids, path):
    """Horizon MAE'yi kayıtlı 24 adım üzerinden, seed varsa ort ± std özetler."""
    selected = horizon.loc[horizon["exp_id"].astype(str).isin(expected_ids)].copy()
    if selected.empty:
        raise ValueError(f"{path} içinde {dataset} horizon sonuçları bulunamadı.")
    selected["exp_id"] = selected["exp_id"].astype(str)
    selected["horizon"] = _numeric_or_missing(selected["horizon"], "horizon", path).astype(int)
    selected["MAE"] = _numeric_or_missing(selected["MAE"], "MAE", path)
    rows = []
    for model_key, exp_ids in FORECAST_IDS[dataset].items():
        runs = selected.loc[selected["exp_id"].isin(exp_ids)]
        if set(runs["exp_id"]) != set(exp_ids):
            raise ValueError(
                f"{path} içinde {dataset}/{model_key} horizon koşusu eksik: "
                f"{sorted(set(exp_ids) - set(runs['exp_id']))}"
            )
        for exp_id, experiment in runs.groupby("exp_id"):
            horizons = sorted(experiment["horizon"].tolist())
            if horizons != list(range(1, 25)):
                raise ValueError(
                    f"{path} içindeki {exp_id} horizon değerleri 1–24 saat değil."
                )
            if not np.isfinite(experiment["MAE"].to_numpy(dtype=float)).all():
                raise ValueError(f"{path} içindeki {exp_id} için eksik/geçersiz MAE değeri var.")
        for step, group in runs.groupby("horizon", sort=True):
            mean, std = _mean_and_std(group["MAE"])
            rows.append({
                "dataset": dataset,
                "model_key": model_key,
                "model_label": MODEL_LABELS[model_key],
                "horizon": int(step),
                "MAE_mean": mean,
                "MAE_std": std,
                "n_seeds": int(group["exp_id"].nunique()),
            })
    result = pd.DataFrame(rows)
    for model_key in FORECAST_MODELS:
        model_rows = result.loc[result["model_key"] == model_key]
        if len(model_rows) != 24:
            raise ValueError(f"{path} içinde {dataset}/{model_key} için 24 ufuk eksik.")
    return result


def _anomaly_summary(dataset, summary, per_seed, summary_path, seed_path, expected_seeds):
    """Ana senaryonun test metriklerini seed CSV'siyle çapraz doğrular."""
    _require_columns(summary, (*ANOMALY_COLUMNS, *ANOMALY_TYPE_COLUMNS), summary_path)
    _require_columns(per_seed, ANOMALY_SEED_COLUMNS, seed_path)
    selected = summary.loc[
        (summary["split"] == "test")
        & (summary["scenario"] == "main")
        & (summary["rule"] == "evf1max")
        & summary["method"].isin(ANOMALY_METHODS)
    ].copy()
    if selected["method"].duplicated().any() or set(selected["method"]) != set(ANOMALY_METHODS):
        raise ValueError(
            f"{summary_path} test/main/evf1max için her hedef anomali yönteminden "
            "tam bir satır içermeli."
        )

    per_seed = per_seed.loc[
        (per_seed["split"] == "test")
        & (per_seed["scenario"] == "main")
        & (per_seed["rule"] == "evf1max")
        & per_seed["method"].isin(ANOMALY_METHODS)
    ].copy()
    if per_seed.duplicated(["method", "seed"]).any():
        raise ValueError(f"{seed_path} içinde method/seed test koşusu yineleniyor.")

    for _, row in selected.iterrows():
        runs = per_seed.loc[per_seed["method"] == row["method"]]
        run_seeds = set(pd.to_numeric(runs["seed"], errors="raise").astype(int))
        if run_seeds != set(expected_seeds):
            raise ValueError(
                f"{dataset}/{row['method']} seed değerleri final test protokolüyle uyuşmuyor: "
                f"beklenen {sorted(expected_seeds)}, bulunan {sorted(run_seeds)}."
            )
        if len(run_seeds) != int(row["n_seeds"]):
            raise ValueError(
                f"{dataset}/{row['method']} seed sayısı özet CSV ve per-seed CSV'de uyuşmuyor."
            )
        for metric, mean_col, std_col in (
            ("pr_auc", "pr_auc_mean", "pr_auc_std"),
            ("event_f1", "event_f1_mean", "event_f1_std"),
            *(
                (f"event_recall_{kind}", f"event_recall_{kind}_mean", f"event_recall_{kind}_std")
                for kind in ("spike", "drop", "level_shift", "pattern")
            ),
        ):
            mean, std = _mean_and_std(runs[metric])
            if not np.isclose(mean, row[mean_col], rtol=1e-4, atol=1e-7):
                raise ValueError(
                    f"{summary_path} {row['method']} {mean_col}, seed CSV ile uyuşmuyor."
                )
            if np.isfinite(std) and not np.isclose(std, row[std_col], rtol=1e-4, atol=1e-7):
                raise ValueError(
                    f"{summary_path} {row['method']} {std_col}, seed CSV ile uyuşmuyor."
                )

    selected["model_key"] = selected["method"]
    selected["method_label"] = selected["method"].map({
        "A0": "A0 — hareketli z-skoru",
        "A3-s2s-norm": "A3-s2s-norm — F10 artık / OOF std",
        "A5-mean-z": "A5-mean-z — A0 + F13 artık",
    })
    return selected.sort_values("method", key=lambda values: values.map(
        {method: position for position, method in enumerate(ANOMALY_METHODS)}
    )).reset_index(drop=True)


def _markdown_excerpt(path, start_marker, end_marker=None):
    """Bir README/rapor bölümünü sunumda kaynak notu olarak okur."""
    if not path.is_file():
        raise FileNotFoundError(f"Sonuçların yorum belgesi bulunamadı: {path}")
    text = path.read_text(encoding="utf-8")
    start = text.find(start_marker)
    if start < 0:
        raise ValueError(f"{path} içinde beklenen bölüm bulunamadı: {start_marker}")
    start += len(start_marker)
    end = text.find(end_marker, start) if end_marker else -1
    excerpt = text[start:end if end >= 0 else None].strip()
    if not excerpt:
        raise ValueError(f"{path} içindeki kaynak açıklaması boş: {start_marker}")
    return excerpt


def _cross_dataset_summary(path, household_forecast, eld_forecast):
    """b_24h karşılaştırma CSV'sini doğrular ve MAE dışı metrikleri hazırlar."""
    required = (
        "level", "group",
        "hane_MAPE", "hane_R2", "hane_vs_SN168_pct", "hane_n_seeds",
        "sebeke_MAPE", "sebeke_R2", "sebeke_vs_SN168_pct", "sebeke_n_seeds",
    )
    saved = _read_csv(path, required)
    saved = saved.loc[saved["level"] == "b_24h"].copy()
    group_names = {
        "F13-ens": "F13-ens (final)",
        "XGBoost": "F12-L336 XGBoost",
        "SN168": "F2 Seasonal Naive 168",
    }
    selected = saved.loc[saved["group"].isin(group_names.values())]
    if selected["group"].duplicated().any() or set(selected["group"]) != set(group_names.values()):
        raise ValueError(f"{path} b_24h satırları gerekli modellerle eşleşmiyor.")
    rows = []
    dataset_forecasts = {
        "household": household_forecast.set_index("model_key"),
        "eld": eld_forecast.set_index("model_key"),
    }
    column_prefix = {"household": "hane", "eld": "sebeke"}
    for dataset, forecast in dataset_forecasts.items():
        prefix = column_prefix[dataset]
        for model_key, group_name in group_names.items():
            saved_row = selected.loc[selected["group"] == group_name].iloc[0]
            forecast_row = forecast.loc[model_key]
            seed_count = int(saved_row[f"{prefix}_n_seeds"])
            if seed_count != int(forecast_row["n_seeds"]):
                raise ValueError(
                    f"{path} {dataset}/{model_key} seed sayısı tahmin test CSV'siyle uyuşmuyor."
                )
            mape = float(saved_row[f"{prefix}_MAPE"])
            r2 = float(saved_row[f"{prefix}_R2"])
            gain = float(saved_row[f"{prefix}_vs_SN168_pct"])
            for value, reference, metric in (
                (mape, forecast_row["MAPE*"], "MAPE*"),
                (r2, forecast_row["R2"], "R²"),
                (gain, forecast_row["gain_vs_sn168_pct"], "SN168 kazancı"),
            ):
                if not np.isclose(value, reference, rtol=1e-4, atol=1e-6):
                    raise ValueError(
                        f"{path} {dataset}/{model_key} {metric} değeri final test "
                        "tahmin CSV'siyle uyuşmuyor."
                    )
            rows.append({
                "dataset": dataset,
                "model_key": model_key,
                "model_label": MODEL_LABELS[model_key],
                "n_seeds": seed_count,
                "MAPE*": mape,
                "MAPE*_std": forecast_row["MAPE*_std"],
                "R2": r2,
                "R2_std": forecast_row["R2_std"],
                "gain_vs_sn168_pct": gain,
                "gain_vs_sn168_pct_std": forecast_row["gain_vs_sn168_pct_std"],
            })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_results_bundle(dataset, project_root=None):
    """Bir veri kümesinin final CSV'lerini okur ve seed/dataset eşleşmelerini doğrular.

    Girdi: ``household`` veya ``eld`` ve isteğe bağlı depo kök dizini.
    Çıktı: tahmin, horizon ve anomali sonuçları; cross-dataset ölçekten bağımsız
    özet ve dokümantasyon bağlamı. Dosyalar salt okunur açılır.
    Neden: Demo yeniden eğitim veya inference yapmadan yalnız kaydedilmiş final
    test ölçümlerini gösterir; veri kümesi eşleşmesi dizin ve exp_id ile korunur.
    """
    if dataset not in FORECAST_IDS:
        raise ValueError(f"Bilinmeyen sonuç veri kümesi: {dataset}")
    root = Path(project_root) if project_root else PROJECT_ROOT
    bundle = _load_one_dataset(dataset, root)
    other_dataset = "eld" if dataset == "household" else "household"
    other_bundle = _load_one_dataset(other_dataset, root)
    comparison_path = root / "results/eld/tables/hane_vs_sebeke_forecast.csv"
    cross_summary = _cross_dataset_summary(
        comparison_path,
        bundle["forecast"] if dataset == "household" else other_bundle["forecast"],
        bundle["forecast"] if dataset == "eld" else other_bundle["forecast"],
    )
    bundle["cross_dataset"] = cross_summary
    bundle["sources"]["cross_dataset"] = str(comparison_path)
    bundle["dataset"] = dataset
    return bundle


def _load_one_dataset(dataset, root):
    """Tek veri kümesinin dört final CSV'sini doğrular, özetler ve kaynak bilgisi ekler."""
    result_dir = root / ("results/tables" if dataset == "household" else "results/eld/tables")
    forecast_path = result_dir / "forecast_test_results.csv"
    horizon_path = result_dir / "forecast_test_horizon.csv"
    anomaly_summary_name, anomaly_seed_name = ANOMALY_FILES[dataset]
    anomaly_path = result_dir / anomaly_summary_name
    anomaly_seed_path = result_dir / anomaly_seed_name
    forecast = _read_csv(forecast_path, FORECAST_COLUMNS)
    horizon = _read_csv(horizon_path, HORIZON_COLUMNS)
    anomaly = _read_csv(
        anomaly_path,
        (*ANOMALY_COLUMNS, *ANOMALY_TYPE_COLUMNS),
    )
    per_seed = _read_csv(anomaly_seed_path, ANOMALY_SEED_COLUMNS)
    forecast_summary, selected_ids = _forecast_summary(dataset, forecast, forecast_path)
    horizon_summary = _horizon_summary(dataset, horizon, selected_ids, horizon_path)
    anomaly_summary = _anomaly_summary(
        dataset,
        anomaly,
        per_seed,
        anomaly_path,
        anomaly_seed_path,
        (
            load_config()["protocol"]["scenarios"]["main"]["seeds"]["test"]
            if dataset == "household"
            else eld_config()["protocol"]["scenarios"]["main"]["seeds"]["test"]
        ),
    )
    final_doc = root / "docs/final_sonuclar.md"
    comparison_doc = root / "docs/hane_vs_sebeke.md"
    return {
        "unit": "kW",
        "forecast": forecast_summary,
        "horizon": horizon_summary,
        "anomaly": anomaly_summary,
        "anomaly_display": _format_anomaly_table(anomaly_summary),
        "sources": {
            "forecast": str(forecast_path),
            "horizon": str(horizon_path),
            "anomaly": str(anomaly_path),
            "anomaly_per_seed": str(anomaly_seed_path),
            "final_document": str(final_doc),
            "comparison_document": str(comparison_doc),
        },
        "final_doc_excerpt": _markdown_excerpt(
            final_doc,
            "# Final sonuçlar: val (dev, hold) ve test yan yana",
            "## 1. Tahmin (24 saat ufuk)",
        ),
        "comparison_doc_excerpt": _markdown_excerpt(
            comparison_doc,
            "**Okuma notları:**",
            "## 1. Tahmin, 24 saat ufuk",
        ),
        "documentation_sources": {
            "final_results": str(final_doc),
            "cross_dataset": str(comparison_doc),
        },
    }


def _format_anomaly_table(summary):
    """Anomali seed ortalamalarını ve std değerlerini jüriye okunur biçimde biçimler."""
    display = pd.DataFrame({
        "Yöntem": summary["method_label"],
        "Seed sayısı": summary["n_seeds"].astype(int),
    })
    metrics = (
        ("PR-AUC", "pr_auc_mean", "pr_auc_std"),
        ("Olay F1", "event_f1_mean", "event_f1_std"),
        ("Spike olay recall", "event_recall_spike_mean", "event_recall_spike_std"),
        ("Drop olay recall", "event_recall_drop_mean", "event_recall_drop_std"),
        ("Seviye kayması olay recall", "event_recall_level_shift_mean", "event_recall_level_shift_std"),
        ("Gece örüntüsü olay recall", "event_recall_pattern_mean", "event_recall_pattern_std"),
    )
    for label, mean_col, std_col in metrics:
        display[label] = [
            metric_text(mean, std)
            for mean, std in zip(summary[mean_col], summary[std_col])
        ]
    return display


def metric_text(mean, std, suffix=""):
    """Eksik sonucu açıkça 'kayıt yok' olarak, seed std'sini varsa ± ile biçimler."""
    if pd.isna(mean):
        return "Kayıt yok"
    value = f"{mean:.3f}"
    if pd.notna(std):
        value += f" ± {std:.3f}"
    return value + suffix
