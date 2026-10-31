"""Final tahmin değerlendirmesi: kayıtlı (yalnız train ile eğitilmiş) modellerle tek seferlik test.

Hiçbir model test için yeniden seçilmez veya ayarlanmaz. Derin modeller kayıtlı
checkpoint'lerle çalışır; Ridge (kapalı çözüm) ve XGBoost (val ile early
stopping) kayıtlı olmadığından val'deki koşuyla birebir aynı şekilde yeniden
fit edilir. `--split val` aynı kodla val tahminlerini yeniden üretip kayıtlı
val tahminleriyle karşılaştırır (test öncesi doğrulama).

Kadro (her aile için test ÖNCESİ val'de seçilmiş temsilci, docs/tahmin_deneyleri_ozet.md §2):
F0–F2 naif, F3-cal Ridge, F4-torch-128x64-fut MLP, F5-opt LSTM (3 seed), F6-C GRU,
F10-opt seq2seq (3 seed), F11-L48 CNN-LSTM, F12-L336 XGBoost, F13-ens (3 seed).
F7–F9 model değil, deney adımıdır (öznitelik ablasyonu, ızgara, seed kararlılığı).
"""

import argparse
from copy import deepcopy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.load import PROJECT_ROOT, load_config
from src.data.split import chronological_split
from src.forecast_levels import level_metrics
from src.models.baselines import NAIVE_EXPERIMENT_IDS, naive_forecasts
from src.models.forecast_baselines import build_train_val_windows, fit_ridge, ridge_design_matrix
from src.models.xgb_forecast import make_xgboost, xgboost_lags
from src.train_forecast import _predict_in_batches, build_model, prepare_forecast_data
from src.utils.console import setup_console
from src.utils.metrics import calculate_metrics
from src.utils.results_io import record_forecast_results, resolve_path

# exp_id → (aile, model etiketi, öznitelik seti, lookback). Derin modellerde ayarlar checkpoint'ten.
ROSTER = {
    "F3-cal": ("ridge", "Ridge", "B", 168),
    "F4-torch-128x64-fut": ("torch", "MLP", "B", None),
    "F5-opt": ("torch", "LSTM", "B", None),
    "F5-opt-s7": ("torch", "LSTM", "B", None),
    "F5-opt-s2026": ("torch", "LSTM", "B", None),
    "F6-C": ("torch", "GRU", "C", None),
    "F10-opt": ("torch", "Seq2Seq-LSTM", "B", None),
    "F10-opt-s7": ("torch", "Seq2Seq-LSTM", "B", None),
    "F10-opt-s2026": ("torch", "Seq2Seq-LSTM", "B", None),
    "F11-L48": ("torch", "CNN-LSTM", "B", None),
    "F12-L336": ("xgboost", "XGBoost", "B", 336),
}
# F13-ens: seed başına 0,8 · F10-opt + 0,2 · F5-opt (Aşama 5 ağırlıkları).
ENSEMBLE = {"F13-ens": ("F10-opt", "F5-opt"), "F13-ens-s7": ("F10-opt-s7", "F5-opt-s7"),
            "F13-ens-s2026": ("F10-opt-s2026", "F5-opt-s2026")}
ENSEMBLE_WEIGHTS = (0.8, 0.2)
# Raporlama grupları (3 seed'li olanlar ort ± std).
GROUPS = {
    "F0 Persistence": ["F0"], "F1 Seasonal Naive 24": ["F1"], "F2 Seasonal Naive 168": ["F2"],
    "F3-cal Ridge": ["F3-cal"], "F4 MLP (fut)": ["F4-torch-128x64-fut"],
    "F5-opt LSTM": ["F5-opt", "F5-opt-s7", "F5-opt-s2026"], "F6-C GRU": ["F6-C"],
    "F10-opt Seq2Seq": ["F10-opt", "F10-opt-s7", "F10-opt-s2026"],
    "F11-L48 CNN-LSTM": ["F11-L48"], "F12-L336 XGBoost": ["F12-L336"],
    "F13-ens (final)": ["F13-ens", "F13-ens-s7", "F13-ens-s2026"],
}


def torch_predictions(df, config, exp_id, feature_set, split, device):
    """Kayıtlı checkpoint ile bir kümenin tahminleri (kW).

    Girdi: tablo, ana config, exp_id, öznitelik seti, küme ("val" | "test"), cihaz.
    Çıktı: (y_true [N, 24], y_pred [N, 24], ilk hedef zamanları [N]).
    Neden: Pencereler eğitimdekiyle aynı fonksiyonla (prepare_forecast_data)
    kurulur; ölçekleyiciler train'den yeniden fit edilir ve checkpoint'teki
    istatistiklerle aynı olduğu kontrol edilir (yanlış ölçek → hata).
    """
    checkpoint = torch.load(resolve_path(PROJECT_ROOT, config["paths"]["models_dir"])
                            / f"{exp_id}_best.pt", map_location="cpu")
    run_config = deepcopy(config)
    run_config["forecast"] = checkpoint["forecast"]
    run_config["features"]["default_set"] = feature_set
    prepared = prepare_forecast_data(df, run_config, eval_split=split)
    if prepared["features"] != checkpoint["features"]:
        raise RuntimeError(f"{exp_id}: öznitelik listesi checkpoint ile uyuşmuyor.")
    np.testing.assert_allclose(prepared["feature_scaler"].mean_, checkpoint["feature_mean"], rtol=1e-9)
    np.testing.assert_allclose(prepared["target_scaler"].mean_, checkpoint["target_mean"], rtol=1e-9)

    future = prepared["future_val"]
    future_size = 0 if future is None else future.shape[2]
    model = build_model(checkpoint["model_name"], prepared["X_val"].shape[2],
                        checkpoint["forecast"], future_size)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    X = torch.from_numpy(prepared["X_val"].astype(np.float32))
    future = None if future is None else torch.from_numpy(future)
    with torch.no_grad():
        scaled = _predict_in_batches(model, X, future, checkpoint["forecast"]["batch_size"], device)
    y_pred = prepared["target_scaler"].inverse_transform(
        scaled.cpu().numpy().reshape(-1, 1)
    ).reshape(prepared["y_val"].shape) + prepared["base_val"]
    return prepared["y_val"], y_pred, prepared["val_times"]


def ridge_predictions(df, config, lookback, split):
    """F3-cal: Ridge train'de yeniden fit (kapalı çözüm, deterministik), kümede tahmin."""
    run_config = deepcopy(config)
    forecast = run_config["forecast"]
    target = run_config["preprocess"]["target"]
    calendar = run_config["features"]["calendar"]
    splits = chronological_split(df, run_config)
    (X_train, y_train, _), (X_eval, y_eval, times) = build_train_val_windows(
        splits, target, lookback, forecast["horizon"],
        run_config["evaluation"]["exclude_imputed_targets"],
        features=[target] + calendar, eval_split=split,
    )
    lags = forecast["ridge"]["lags"]
    predictions, _, _, _ = fit_ridge(
        ridge_design_matrix(X_train, lags, True), y_train,
        ridge_design_matrix(X_eval, lags, True),
        splits["train"][target].to_numpy(dtype=np.float64), forecast["ridge"]["alpha"],
    )
    return y_eval, predictions, times


def xgboost_predictions(df, config, lookback, split):
    """F12-L336: XGBoost val'deki koşuyla aynı (train'de fit, val MAE ile early stopping), kümede tahmin."""
    run_config = deepcopy(config)
    run_config["forecast"]["lookback"] = lookback
    forecast = run_config["forecast"]
    target = run_config["preprocess"]["target"]
    calendar = run_config["features"]["calendar"]
    lags = xgboost_lags(run_config)
    splits = chronological_split(df, run_config)
    exclude = run_config["evaluation"]["exclude_imputed_targets"]
    (X_train, y_train, _), (X_val, y_val, val_times) = build_train_val_windows(
        splits, target, lookback, forecast["horizon"], exclude, features=[target] + calendar,
    )
    model = make_xgboost(forecast["xgboost"], run_config["seed"])
    model.fit(ridge_design_matrix(X_train, lags, True), y_train,
              eval_set=[(ridge_design_matrix(X_val, lags, True), y_val)], verbose=False)
    if split == "val":
        return y_val, model.predict(ridge_design_matrix(X_val, lags, True)), val_times
    _, (X_eval, y_eval, times) = build_train_val_windows(
        splits, target, lookback, forecast["horizon"], exclude,
        features=[target] + calendar, eval_split=split,
    )
    return y_eval, model.predict(ridge_design_matrix(X_eval, lags, True)), times


def all_predictions(df, config, split, device, roster=None, ensemble=None, prefix=""):
    """Kadrodaki her exp_id için (y_true, y_pred, zamanlar).

    Girdi: roster/ensemble verilmezse household kadrosu; prefix naif modellerin
    exp_id önekidir (ELD: "ELD-"). Kadro test öncesi sabittir.
    """
    roster = ROSTER if roster is None else roster
    ensemble = ENSEMBLE if ensemble is None else ensemble
    results = {}
    part = chronological_split(df, config)[split]
    y_true, naive, times = naive_forecasts(
        df, part, config["preprocess"]["target"], 168, config["forecast"]["horizon"],
        config["evaluation"]["exclude_imputed_targets"],
    )
    for name, y_pred in naive.items():
        results[prefix + NAIVE_EXPERIMENT_IDS[name]] = (y_true, y_pred, times)
    for exp_id, (family, _, feature_set, lookback) in roster.items():
        if family == "torch":
            results[exp_id] = torch_predictions(df, config, exp_id, feature_set, split, device)
        elif family == "ridge":
            results[exp_id] = ridge_predictions(df, config, lookback, split)
        else:
            results[exp_id] = xgboost_predictions(df, config, lookback, split)
        print(f"{exp_id}: {len(results[exp_id][0])} pencere")
    for exp_id, (seq2seq_id, lstm_id) in ensemble.items():
        y_true, seq2seq_pred, times = results[seq2seq_id]
        lstm_true, lstm_pred, lstm_times = results[lstm_id]
        if not (pd.DatetimeIndex(times).equals(pd.DatetimeIndex(lstm_times))
                and np.array_equal(y_true, lstm_true)):
            raise RuntimeError("Topluluk üyelerinin pencereleri farklı.")
        results[exp_id] = (y_true, ENSEMBLE_WEIGHTS[0] * seq2seq_pred
                           + ENSEMBLE_WEIGHTS[1] * lstm_pred, times)
    return results


def verify_against_saved_val(results, config):
    """--split val: yeniden üretilen val tahminleri kayıtlı val tahminleriyle aynı mı?"""
    predictions_dir = resolve_path(PROJECT_ROOT, config["paths"]["predictions_dir"])
    rows = []
    for exp_id, (y_true, y_pred, times) in results.items():
        saved = np.load(predictions_dir / f"{exp_id}_val.npz")
        rows.append({
            "exp_id": exp_id,
            "same_windows": bool(np.array_equal(pd.DatetimeIndex(times).asi8, saved["target_times"])),
            "max_abs_diff_true": float(np.abs(y_true - saved["y_true"]).max()),
            "max_abs_diff_pred_kw": float(np.abs(y_pred - saved["y_pred"]).max()),
        })
    return pd.DataFrame(rows)


def summarize_levels(results, mape_min_y, groups=None):
    """Üç seviye (h=1, 24 s, günlük) metrikleri: seed başına ve grup ort ± std."""
    rows = []
    for group, exp_ids in (GROUPS if groups is None else groups).items():
        for exp_id in exp_ids:
            y_true, y_pred, times = results[exp_id]
            for level, metrics in level_metrics(y_true, y_pred, pd.DatetimeIndex(times).asi8,
                                                mape_min_y).items():
                rows.append({"group": group, "exp_id": exp_id, "level": level, **metrics})
    per_run = pd.DataFrame(rows)
    metrics = ["MAE", "RMSE", "MAPE*", "sMAPE", "R2"]
    summary = per_run.groupby(["group", "level"], sort=False).agg(
        n_seeds=("exp_id", "count"), n=("n", "first"),
        **{f"{m}_mean": (m, "mean") for m in metrics},
        **{f"{m}_std": (m, "std") for m in metrics},
    ).reset_index()
    return per_run, summary


def save_error_figures(results, figures_dir, split, prefix="", dataset="household"):
    """Ufka göre MAE ve günün saatine göre MAE (seed'li modellerde ilk seed) — figürler."""
    shown = {"F2 SN168": "F2", "F3-cal Ridge": "F3-cal", "F5-opt LSTM": "F5-opt",
             "F10-opt Seq2Seq": "F10-opt", "F12-L336 XGBoost": "F12-L336", "F13-ens": "F13-ens"}
    shown = {label: prefix + exp_id for label, exp_id in shown.items()}
    tag = f"{split.upper()} ({dataset})"
    figure, axis = plt.subplots(figsize=(9, 5))
    for label, exp_id in shown.items():
        y_true, y_pred, _ = results[exp_id]
        axis.plot(range(1, 25), np.abs(y_true - y_pred).mean(axis=0), marker="o", markersize=3,
                  label=label, linewidth=2.5 if exp_id.endswith("F13-ens") else 1.2)
    axis.set(xlabel="Ufuk h (saat)", ylabel="MAE (kW)",
             title=f"Ufka göre MAE – {tag} (seed 42)")
    axis.set_xticks(range(1, 25, 2))
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figures_dir / f"{prefix}F_final_{split}_mae_by_horizon.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    for label, exp_id in shown.items():
        y_true, y_pred, times = results[exp_id]
        # Her hedef saatin günün saati: ilk hedef saati + ufuk.
        hours = (pd.DatetimeIndex(times).hour.to_numpy()[:, None] + np.arange(24)) % 24
        errors = pd.Series(np.abs(y_true - y_pred).ravel()).groupby(hours.ravel()).mean()
        axis.plot(errors.index, errors.to_numpy(), marker="o", markersize=3, label=label,
                  linewidth=2.5 if exp_id.endswith("F13-ens") else 1.2)
    axis.set(xlabel="Hedef saatin günün saati", ylabel="MAE (kW, tüm ufuklar)",
             title=f"Günün saatine göre MAE – {tag} (seed 42)")
    axis.set_xticks(range(0, 24, 2))
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figures_dir / f"{prefix}F_final_{split}_mae_by_hour.png", dpi=200)
    plt.close(figure)


def fill_test_columns(experiments_file, results, mape_min_y):
    """experiments.csv'de mevcut satırların yalnız test_mae / test_rmse sütunlarını doldurur."""
    table = pd.read_csv(experiments_file)
    for exp_id, (y_true, y_pred, _) in results.items():
        scores = calculate_metrics(y_true, y_pred, mape_min_y)
        mask = table["exp_id"] == exp_id
        table.loc[mask, "test_mae"] = scores["MAE"]
        table.loc[mask, "test_rmse"] = scores["RMSE"]
    table.to_csv(experiments_file, index=False)


def main():
    """--split val: doğrulama (kayıtlı val ile karşılaştırma). --split test: final değerlendirme."""
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_config()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    df = pd.read_parquet(resolve_path(PROJECT_ROOT, config["paths"]["processed_file"]))
    tables_dir = resolve_path(PROJECT_ROOT, config["paths"]["tables_dir"])
    figures_dir = resolve_path(PROJECT_ROOT, config["paths"]["figures_dir"])
    mape_min_y = config["evaluation"]["mape_min_y"]

    results = all_predictions(df, config, args.split, device)
    if args.split == "val":
        check = verify_against_saved_val(results, config)
        check.to_csv(tables_dir / "forecast_final_val_reproduction.csv", index=False)
        print(check.to_string(index=False))
        return

    for exp_id, (y_true, y_pred, times) in results.items():
        label = next(group for group, ids in GROUPS.items() if exp_id in ids)
        record_forecast_results(PROJECT_ROOT, config, "test", exp_id, label, y_true, y_pred, times,
                                train_time_s=0.0, device=str(device), feature_set="",
                                lookback="", note="final test; yalnız train ağırlıkları")
    fill_test_columns(resolve_path(PROJECT_ROOT, config["paths"]["experiments_file"]),
                      results, mape_min_y)
    per_run, summary = summarize_levels(results, mape_min_y)
    per_run.to_csv(tables_dir / "forecast_levels_test_per_seed.csv", index=False)
    summary.to_csv(tables_dir / "forecast_levels_test.csv", index=False)
    save_error_figures(results, figures_dir, "test")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
