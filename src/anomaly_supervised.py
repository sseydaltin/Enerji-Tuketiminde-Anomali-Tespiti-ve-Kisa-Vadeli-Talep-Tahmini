"""Sentetik anomalilerle eğitilen gözetimli saatlik dedektör (docs/anomaly_supervised.md; geliştirme).

Yöntem değişikliği: tek bir skoru eşiklemek yerine, yalnız geçmişten hesaplanan özelliklerle
(A3 artıkları, sağlam profil sapması, A0 z, değişim, taban, takvim) bir sınıflandırıcı, değerlendirme
dönemlerinden ÖNCEKİ OOF dönemine enjekte edilen çeşitlendirilmiş sentetik anomalilerle eğitilir.
Eşik, eğitim döneminin iç doğrulama kopyalarında seçilir ve dev değerlendirmesinden önce dondurulur.
val_hold ve test KULLANILMAZ. Tahmin modeli eğitilmez (kayıtlı OOF ve ana ağırlıklar).
Komut: python -m src.anomaly_supervised
"""

import argparse
import time
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.anomaly_candidates import S0_SCALED_THRESHOLD, decide, evaluation_arrays, fit_floor, prepare, robust_profile, day_type, summarize
from src.anomaly_dev import SET_KEYS, dev_scenario_settings, prepare_set
from src.anomaly_e1_sustained import MANIFEST_ITEMS, Log, copy_metrics, paired
from src.anomaly_eval_audit import sha256
from src.anomaly_floor_channel import combined, floor_flags
from src.anomaly_short_event import change_table, signal_frame
from src.data.inject_anomalies import ANOMALY_TYPES, inject_anomalies, sigma_for
from src.data.load import PROJECT_ROOT, load_config
from src.models.residual_detector import (
    multi_step_predictions,
    normalized_scores,
    out_of_fold_predictions,
)
from src.utils.anomaly_metrics import event_f1_max_threshold
from src.utils.console import setup_console
from src.utils.results_io import resolve_path, upsert_csv
from src.utils.seed import set_seed

FEATURES = ["a3", "zbar", "zbar_m3", "zbar_m6", "z1", "change", "c1z", "c1z_m6", "ratio", "low2",
            "a0z", "hour_sin", "hour_cos", "daytype"]
# Eğitim dönemleri (protokol §1): fit, 7 gün boşluk, iç doğrulama (eşik).
PERIODS = {
    "winter_dev": {"fit": ("2008-10-01", "2009-07-24 23:00"), "inner": ("2009-08-01", "2009-09-30 23:00")},
    "summer_dev": {"fit": ("2008-10-01", "2009-03-24 23:00"), "inner": ("2009-04-01", "2009-05-31 23:00")},
}
FIT_SEEDS = range(7001, 7009)        # 8 fit kopyası
INNER_SEEDS = range(7101, 7105)      # 4 iç doğrulama kopyası (ilk 4 fit kopyasıyla aynı OOF koşusunda)
DROP_VALUES = (0.05, 0.10, 0.20, 0.30)
PATTERN_ALPHAS = (0.5, 0.75, 1.0)
HISTORY_HOURS = 24 * 16             # C1-L2 (14 gün) + A0 (7 gün) için yeterli geçmiş
FLOOR_QUANTILE = 0.005
MODELS_DIR = Path("models/anomaly_supervised")   # git dışı (.gitignore)


def make_models(seed):
    """Üç aile, sabit küçük ayarlar (ızgara yok)."""
    return {
        "LR": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
        "RF": make_pipeline(SimpleImputer(strategy="median"),
                            RandomForestClassifier(n_estimators=300, min_samples_leaf=20, class_weight="balanced",
                                                   n_jobs=-1, random_state=seed)),
        "HGB": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                              class_weight="balanced", random_state=seed),
    }


def diverse_settings(base, copy_index):
    """Fit kopyası için çeşitlendirilmiş enjeksiyon ayarları (değerlendirme senaryosu değişmez)."""
    settings = deepcopy(base)
    settings["spike"] = {"min_len": 1, "max_len": 2, "sigma_low": 2.0, "sigma_high": 6.0}
    settings["drop"] = {"min_len": 1, "max_len": 6, "value_kw": DROP_VALUES[copy_index % len(DROP_VALUES)],
                        "min_start_kw": 0.6}
    settings["level_shift"] = {"min_len": 12, "max_len": 48, "factor_low": 1.3, "factor_high": 2.5}
    settings["pattern"] = {"night_hours": [0, 1, 2, 3, 4, 5],
                           "alpha": PATTERN_ALPHAS[copy_index % len(PATTERN_ALPHAS)]}
    return settings


def signed_a0(values, days=7, min_std=0.05):
    """İşaretli A0: (y − ort) / std, son 7 günün aynı saati (yalnız geçmiş)."""
    past = pd.concat([values.shift(24 * k) for k in range(1, days + 1)], axis=1)
    std = past.std(axis=1).clip(lower=min_std)
    return ((values - past.mean(axis=1)) / std).where(past.notna().all(axis=1))


def build_features(frame, predictions, multi_std, change_tab, c1_floor, floor, target):
    """[T, F] özellik tablosu; her sütun yalnız t ve öncesine bakar. frame geçmiş bağlam içerir."""
    signals = signal_frame(frame, predictions, multi_std, change_tab, target)
    values = frame[target]
    zbar = signals["spike"] - signals["drop"]
    median, scale, count = robust_profile(frame, target, 2)
    keys = pd.MultiIndex.from_arrays([frame.index.hour, day_type(frame)])
    scale = pd.Series(np.maximum(scale.to_numpy(), c1_floor.reindex(keys).to_numpy()), index=frame.index)
    c1z = ((values - median) / scale).where(count >= 4)
    features = pd.DataFrame({
        "a3": signals["A3"],
        "zbar": zbar,
        "zbar_m3": zbar.rolling(3, min_periods=3).mean(),
        "zbar_m6": zbar.rolling(6, min_periods=6).mean(),
        "z1": normalized_scores(values - predictions[1], multi_std[1]),
        "change": signals["change"],
        "c1z": c1z,
        "c1z_m6": c1z.rolling(6, min_periods=6).mean(),
        "ratio": np.log((values + 0.05) / (median + 0.05)).where(count >= 4),
        "low2": np.log((np.maximum(values, values.shift(1)) + 0.05) / (floor + 0.05)),
        "a0z": signed_a0(values),
        "hour_sin": frame["hour_sin"],
        "hour_cos": frame["hour_cos"],
        "daytype": day_type(frame),
    }, index=frame.index)
    return features[FEATURES]


def season_context(prepared):
    """Mevsime ait train-only istatistikler: değişim tablosu, C1 tabanı, fiziksel taban, OOF yolları."""
    cfg = prepared["config"]
    target = cfg["preprocess"]["target"]
    train = prepared["splits"]["train"]
    residual = cfg["anomaly"]["residual"]
    models_dir = resolve_path(PROJECT_ROOT, cfg["paths"]["models_dir"])
    return {
        "target": target,
        "change_tab": change_table(train, target),
        "c1_floor": fit_floor(train, target, 2),
        "floor": float(train.loc[train["is_imputed"] == 0, target].quantile(FLOOR_QUANTILE)),
        "oof_paths": [models_dir / f"{residual['seq2seq']}-oof{k}_best.pt"
                      for k in range(1, residual["oof_folds"] + 1)],
    }


def training_copies(prepared, context, set_name, log):
    """Eğitim döneminde enjekte kopyalar → (fit X, y), iç doğrulama kopyaları [(types, X, valid)]."""
    cfg, train = prepared["config"], prepared["splits"]["train"]
    target = context["target"]
    sigma = sigma_for(train, prepared["sigma_table"])
    medium = dev_scenario_settings(cfg, "medium", train)
    fit_slice = slice(*PERIODS[set_name]["fit"])
    inner_slice = slice(*PERIODS[set_name]["inner"])
    fit_X, fit_y, inner = [], [], []
    for i, seed in enumerate(FIT_SEEDS):
        injected = train.copy()
        labels = pd.DataFrame({"is_anomaly": 0, "anomaly_type": "none"}, index=train.index)
        parts = [(fit_slice, diverse_settings(cfg["injection"], i), seed)]
        if i < len(INNER_SEEDS):
            parts.append((inner_slice, medium, INNER_SEEDS[i]))
        for period, settings, part_seed in parts:
            sub = train.loc[period]
            sub_injected, sub_labels = inject_anomalies(sub, target, settings, part_seed,
                                                        sigma[train.index.get_indexer(sub.index)])
            injected.loc[sub.index, target] = sub_injected[target]
            labels.loc[sub.index, ["is_anomaly", "anomaly_type"]] = sub_labels[["is_anomaly", "anomaly_type"]].to_numpy()
        predictions = out_of_fold_predictions(injected, context["oof_paths"], "cpu", multi=True)
        features = build_features(injected, predictions, prepared["fitted"]["multi_std"], context["change_tab"],
                                  context["c1_floor"], context["floor"], target)
        clean = injected["is_imputed"] == 0
        fit_index = train.loc[fit_slice].index
        keep = features.loc[fit_index, "a3"].notna() & clean.loc[fit_index]
        fit_X.append(features.loc[fit_index][keep])
        fit_y.append(labels.loc[fit_index, "is_anomaly"][keep].astype(int))
        if i < len(INNER_SEEDS):
            inner_index = train.loc[inner_slice].index
            valid = (features.loc[inner_index, "a3"].notna() & clean.loc[inner_index]).to_numpy()
            inner.append((labels.loc[inner_index, "anomaly_type"].to_numpy(), features.loc[inner_index], valid))
        log(f"   {set_name}: eğitim kopyası {i + 1}/{len(FIT_SEEDS)} (seed {seed})")
    return pd.concat(fit_X), pd.concat(fit_y), inner


def probabilities(model, features, valid):
    """[N] anomali olasılığı; A3 artığı olmayan / doldurulmuş saatler NaN."""
    probability = np.full(len(features), np.nan)
    if valid.any():
        probability[valid] = model.predict_proba(features[valid])[:, 1]
    return probability


def dev_features(prepared, context, frame):
    """Dev kopyası (veya temiz küme) özellikleri: train'in son 16 günü (enjeksiyonsuz) bağlam olarak."""
    train = prepared["splits"]["train"]
    full = pd.concat((train.tail(HISTORY_HOURS), frame))
    model, checkpoint = prepared["fitted"]["forecasters"][prepared["config"]["anomaly"]["residual"]["seq2seq"]]
    predictions = multi_step_predictions(model, checkpoint, full, "cpu")
    features = build_features(full, predictions, prepared["fitted"]["multi_std"], context["change_tab"],
                              context["c1_floor"], context["floor"], context["target"]).loc[frame.index]
    valid = (features["a3"].notna() & (frame["is_imputed"] == 0)).to_numpy()
    return features, valid


def point_metrics(labels, flagged, valid):
    """Nokta (saat) precision/recall/F1, geçerli saatlerde."""
    y, p = labels[valid].astype(bool), flagged[valid]
    tp = int((y & p).sum())
    precision = tp / p.sum() if p.sum() else 0.0
    recall = tp / y.sum() if y.sum() else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"point_precision": precision, "point_recall": recall, "point_f1": f1}


def main():
    setup_console()
    argparse.ArgumentParser(description=__doc__).parse_args()
    config = load_config()
    set_seed(config["seed"])
    tables_dir = resolve_path(PROJECT_ROOT, config["paths"]["tables_dir"])
    log = Log(resolve_path(PROJECT_ROOT, "results/logs/anomaly_supervised.log"))
    manifest = pd.read_csv(tables_dir / "repro_manifest.csv").set_index("item")
    for item in MANIFEST_ITEMS:
        if sha256(PROJECT_ROOT / manifest.loc[item, "path"]) != manifest.loc[item, "sha256"]:
            raise RuntimeError(f"{item} hash'i manifestoyla uyuşmuyor.")
    log("[1/6] manifest denetimi tamam")

    df = pd.read_parquet(resolve_path(PROJECT_ROOT, config["paths"]["processed_file"]))
    settings = config["protocol"]["threshold_quantiles"]
    quantiles = np.linspace(settings["low"], settings["high"], settings["n"])
    models_dir = resolve_path(PROJECT_ROOT, str(MODELS_DIR))
    models_dir.mkdir(parents=True, exist_ok=True)
    q_a3 = pd.read_csv(tables_dir / "E1_sustained_scales.csv").set_index("set")["A3"]

    trained, inner_rows, timing, contexts, prepared_sets = {}, [], [], {}, {}
    for set_name in SET_KEYS:
        prepared = prepare_set(df, config, set_name, "cpu")
        context = season_context(prepared)
        prepared_sets[set_name], contexts[set_name] = prepared, context
        start = time.perf_counter()
        fit_X, fit_y, inner = training_copies(prepared, context, set_name, log)
        build_s = time.perf_counter() - start
        log(f"[2/6] {set_name}: fit satırı {len(fit_X)}, anomali oranı {fit_y.mean():.3f}; veri üretimi {build_s:.0f} s")
        for family, model in make_models(config["seed"]).items():
            start = time.perf_counter()
            model.fit(fit_X, fit_y)
            fit_s = time.perf_counter() - start
            copies = [(types, probabilities(model, X, valid), valid) for types, X, valid in inner]
            threshold, pooled = event_f1_max_threshold(copies, quantiles)
            trained[(set_name, family)] = (model, threshold)
            joblib.dump({"model": model, "threshold": threshold, "features": FEATURES, "set": set_name},
                        models_dir / f"{family}_{set_name}.joblib")
            inner_rows.append({"set": set_name, "family": family, "threshold": threshold,
                               "inner_pooled_block_f1": pooled})
            timing.append({"set": set_name, "family": family, "data_build_s": build_s, "fit_s": fit_s})
            log(f"[3/6] {set_name} {family}: fit {fit_s:.1f} s, iç doğrulama birleşik block_f1 {pooled:.3f}, "
                f"eşik {threshold:.4f}")

    inner_table = pd.DataFrame(inner_rows)
    chosen = inner_table.groupby("family")["inner_pooled_block_f1"].mean().idxmax()
    inner_table["chosen"] = inner_table["family"] == chosen
    inner_table.to_csv(tables_dir / "anomaly_supervised_inner_selection.csv", index=False)
    log(f"[4/6] DONDURULDU: seçilen aile {chosen} (iç doğrulama ortalaması); dev değerlendirmesi başlıyor")

    rows = []
    infer_s = {f: 0.0 for f in ("LR", "RF", "HGB")}
    for set_name in SET_KEYS:
        data = prepare(set_name, df, config)          # aynı dev kopyaları + referans skorları
        prepared, context = prepared_sets[set_name], contexts[set_name]
        clean_features, clean_valid = dev_features(prepared, context, data["background"]["frame"])
        backgrounds = {}
        for family in ("LR", "RF", "HGB"):
            model, threshold = trained[(set_name, family)]
            backgrounds[family] = (probabilities(model, clean_features, clean_valid) >= threshold) & clean_valid
        ref_background = {}
        for method, (window, threshold) in {"S0-yerel": (1, S0_SCALED_THRESHOLD), "A0": (6, 3.9893783062624975),
                                            "A3-s2s-norm": (1, 4.240649915881419)}.items():
            score, valid = evaluation_arrays(data["background"], data["scales"], method, window)
            ref_background[method] = (score >= threshold) & valid
        frame = data["background"]["frame"]
        a_clean, _ = evaluation_arrays(data["background"], data["scales"], "S0-yerel", 1)
        score, valid = combined(a_clean, floor_flags(frame[context["target"]], context["floor"]),
                                frame["is_imputed"].to_numpy() == 1)
        ref_background["S0+taban"] = (score >= 1.0) & valid
        for copy in data["copies"]:
            labels = copy["labels"]
            types, is_anomaly = labels["anomaly_type"].to_numpy(), labels["is_anomaly"].to_numpy()
            features, valid = dev_features(prepared, context, copy["frame"])
            for family in ("LR", "RF", "HGB"):
                model, threshold = trained[(set_name, family)]
                start = time.perf_counter()
                probability = probabilities(model, features, valid)
                infer_s[family] += time.perf_counter() - start
                flagged = (np.nan_to_num(probability, nan=-1) >= threshold) & valid
                rows.append({"set": set_name, "method": f"SUP-{family}", "seed": copy["seed"],
                             **copy_metrics(types, is_anomaly, probability, valid, threshold, backgrounds[family]),
                             **point_metrics(is_anomaly, flagged, valid)})
            for method, (window, threshold) in {"S0-yerel": (1, S0_SCALED_THRESHOLD), "A0": (6, 3.9893783062624975),
                                                "A3-s2s-norm": (1, 4.240649915881419)}.items():
                score, valid_ref = evaluation_arrays(copy, data["scales"], method, window)
                flagged = (np.nan_to_num(score, nan=-np.inf) >= threshold) & valid_ref
                rows.append({"set": set_name, "method": method, "seed": copy["seed"],
                             **copy_metrics(types, is_anomaly, score, valid_ref, threshold, ref_background[method]),
                             **point_metrics(is_anomaly, flagged, valid_ref)})
            a, _ = evaluation_arrays(copy, data["scales"], "S0-yerel", 1)
            score, valid_ref = combined(a, floor_flags(copy["frame"][context["target"]], context["floor"]),
                                        copy["frame"]["is_imputed"].to_numpy() == 1)
            rows.append({"set": set_name, "method": "S0+taban", "seed": copy["seed"],
                         **copy_metrics(types, is_anomaly, score, valid_ref, 1.0, ref_background["S0+taban"]),
                         **point_metrics(is_anomaly, (score >= 1.0) & valid_ref, valid_ref)})
        log(f"[5/6] dev değerlendirildi: {set_name}")

    per_seed = pd.DataFrame(rows)
    summary = summarize(per_seed)
    n_copies = per_seed.groupby("method").size()
    timing = pd.DataFrame(timing)
    timing["infer_s_per_copy"] = timing["family"].map({f: s / n_copies[f"SUP-{f}"] for f, s in infer_s.items()})
    decisions = []
    for method in [f"SUP-{f}" for f in ("LR", "RF", "HGB")] + ["S0+taban"]:
        decision = decide(per_seed, summary, method)
        s = summary.set_index(["method", "set"])
        decision["c7_spike_loss_le_0.05"] = all(
            s.loc[("S0-yerel", k), "event_recall_spike"] - s.loc[(method, k), "event_recall_spike"] <= 0.05
            for k in SET_KEYS)
        decision["accepted"] = bool(decision["accepted"] and decision["c7_spike_loss_le_0.05"])
        decision["target_0.70_both_seasons"] = bool(all(s.loc[(method, k), "block_f1"] >= 0.70 for k in SET_KEYS))
        decision["chosen_family"] = method == f"SUP-{chosen}"
        decisions.append(decision)
    paired_rows = [{"method": m, **r} for m in per_seed["method"].unique() if m != "S0-yerel"
                   for metric in ("block_f1", "count_f1", "point_f1", "block_precision", "event_recall",
                                  "false_alarms_per_day", "detection_delay_mean",
                                  *[f"event_recall_{k}" for k in ANOMALY_TYPES])
                   for r in paired(per_seed, m, metric)]
    prefix = tables_dir / "anomaly_supervised"
    per_seed.to_csv(f"{prefix}_per_seed.csv", index=False)
    summary.to_csv(f"{prefix}_summary.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(f"{prefix}_paired.csv", index=False)
    pd.DataFrame(decisions).to_csv(f"{prefix}_decision.csv", index=False)
    timing.to_csv(f"{prefix}_timing.csv", index=False)
    upsert_csv(resolve_path(PROJECT_ROOT, config["paths"]["experiments_file"]), [
        {"exp_id": f"SUP-{f}", "date": pd.Timestamp.today().date().isoformat(),
         "model": f"anomali gözetimli {f} (sentetik eğitim; docs/anomaly_supervised.md)", "seed": config["seed"],
         "note": "dev orta senaryo block_f1 kış {:.3f} / yaz {:.3f}; eşik iç doğrulamada; val_hold/test yok".format(
             *summary.set_index(["method", "set"]).loc[[(f"SUP-{f}", "winter_dev"), (f"SUP-{f}", "summer_dev")],
                                                       "block_f1"])}
        for f in ("LR", "RF", "HGB")], ["exp_id"])
    log(f"[6/6] bitti; seçilen {chosen}; kararlar {[(d['method'], d['accepted'], d['target_0.70_both_seasons']) for d in decisions]}")

    columns = ["method", "set", "block_f1", "count_f1", "point_f1", "block_precision", "event_recall",
               "event_recall_spike", "event_recall_drop", "event_recall_level_shift", "event_recall_pattern",
               "false_alarms_per_day", "detection_delay_mean", "false_blocks_in_background"]
    print(inner_table.round(4).to_string(index=False))
    print(summary[columns].round(3).to_string(index=False))
    print(pd.DataFrame(decisions).round(4).to_string(index=False))
    print(timing.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
