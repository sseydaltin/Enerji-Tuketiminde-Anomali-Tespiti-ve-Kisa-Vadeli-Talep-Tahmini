"""Donmuş sonuçların yeniden yapılandırmadan etkilenmediğini doğrular.

Kullanım (proje kökünden):
    python scripts/verify_frozen_results.py snapshot --out <repo DIŞI klasör>
    python scripts/verify_frozen_results.py compare <referans klasör> <yeni klasör>

Ne yapar:
  1. Dosya özetleri (SHA-256): final tablolar, ELD tabloları, experiments.csv,
     donmuş belgeler, config'ler, ağırlıklar, işlenmiş veri.
  2. Kayıtlı ağırlıklarla yeniden üretilen diziler (CPU, sabit iş parçacığı):
     hane tahminleri (val, test), hane anomali skorları ve seed başına metrikler
     (val_dev, val_hold, test; ana + zor), A6 (val_dev, val_hold), ELD tahmin ve
     anomali, demo servislerinin bir test günü ve bir enjeksiyon senaryosu çıktısı.
  3. compare: özetler birebir aynı mı, diziler atol = 1e-6 içinde mi?

Neden: Dosyalar taşınınca import yolları değişir; hesap değişmemelidir. Kod
yolu farkı varsa (yanlış config, yanlış ağırlık, yanlış kök klasör) sayılar
değişir ve burada yakalanır.

Kurallar (CLAUDE.md, deney günlüğü 2026-11-25):
  - CLI main() fonksiyonları ÇAĞRILMAZ; onlar results/ ve experiments.csv'ye yazar.
    Yalnız yan etkisiz kütüphane fonksiyonları çağrılır.
  - Çıktı yalnız proje DIŞINDAKİ bir klasöre yazılır.
  - Test kümesi yalnız regresyon kontrolü için kullanılır: seçim, eşik, ayar yoktur;
    kayıtlı seçim tabloları aynen okunur.
"""

import argparse
import hashlib
import importlib
import json
import platform
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))   # betik doğrudan çalıştırıldığında `src` bulunabilsin

ATOL = 1e-6
# İş parçacığı sayıları sabittir; iki koşu (taşımadan önce/sonra) aynı sayıyla çalışır.
# torch: 4. XGBoost: 12 = bu makinedeki çekirdek sayısı ve final koşudaki varsayılan.
# Ölçüldü (2026-11-25): XGBoost çıktısı iş parçacığı sayısına bağlı (4 → kayıtlı final
# tahminden 0,65 kW'a kadar fark), 12 ile kayıtlı tahminler birebir çıkıyor. Her ayar
# kendi içinde deterministik. OMP_NUM_THREADS bu yüzden ayarlanmaz (XGBoost'u sınırlar).
TORCH_THREADS = 4
XGB_THREADS = 12

# Özeti alınacak dosyalar (proje köküne göre glob).
HASHED = [
    "results/experiments.csv",
    "results/tables/final_*.csv",
    "results/tables/anomaly_protocol_*.csv",
    "results/tables/anomaly_protocol_*.md",
    "results/tables/forecast_levels_*.csv",
    "results/tables/forecast_test_*.csv",
    "results/tables/a6_*.csv",
    "results/tables/hane_vs_sebeke_*.csv",
    "results/tables/demo_*.csv",
    "results/eld/experiments.csv",
    "results/eld/tables/*",
    "docs/final_sonuclar.md",
    "docs/hane_vs_sebeke.md",
    "docs/final_test_protokolu.md",
    "docs/eld_protokol.md",
    "configs/*.yaml",
    "models/*.pt",
    "models/eld/*.pt",
    "data/processed/*.parquet",
]


def first_module(*names):
    """Verilen modül adlarından ilk import edilebileni döndürür.

    Neden: Aynı betik taşımadan önce (eski yol) ve sonra (yeni yol) çalışır.
    Yeni yol önce denenir; eski yerdeki ince CLI yönlendirmesi kütüphane
    fonksiyonlarını taşımadığı için gereken ad bulunamazsa sıradakine geçilir.
    """
    errors = []
    for name in names:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as error:
            errors.append(str(error))
    raise ModuleNotFoundError("; ".join(errors))


def modules():
    """Kullanılan modülleri yeni yol → eski yol sırasıyla yükler."""
    return {
        "config": first_module("src.core.config", "src.data.load"),
        "inject": first_module("src.core.inject_anomalies", "src.data.inject_anomalies"),
        "split": first_module("src.core.split", "src.data.split"),
        "results_io": first_module("src.core.results_io", "src.utils.results_io"),
        "forecast": first_module("src.pipelines.evaluate_forecast", "src.evaluate_forecast"),
        "anomaly": first_module("src.pipelines.evaluate_anomaly", "src.evaluate_anomaly"),
        "protocol": first_module("src.pipelines.anomaly_protocol", "src.anomaly_protocol"),
        "dual": first_module("src.pipelines.anomaly_dual", "src.anomaly_dual"),
        "eld_run": first_module("src.pipelines.eld_run", "src.eld_run"),
        "eld_data": first_module("src.datasets.eld.eld", "src.data.eld"),
        "services": importlib.import_module("app.services"),
        "anomaly_services": importlib.import_module("app.anomaly_services"),
        "eld_services": importlib.import_module("app.eld_services"),
    }


def sha256(path):
    """Bir dosyanın SHA-256 özeti (1 MB'lık parçalarla okunur)."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_hashes():
    """HASHED kalıplarına uyan her dosya için {göreli yol: özet}."""
    hashes = {}
    for pattern in HASHED:
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file():
                hashes[path.relative_to(ROOT).as_posix()] = sha256(path)
    return hashes


def times_to_int(times):
    """Zaman dizisini karşılaştırılabilir int64 (ns) dizisine çevirir."""
    return pd.DatetimeIndex(times).as_unit("ns").asi8


def save_forecasts(results, folder, tag):
    """{exp_id: (y_true [N,24], y_pred [N,24], zamanlar [N])} → exp_id başına bir .npz."""
    for exp_id, (y_true, y_pred, times) in results.items():
        np.savez(folder / f"{tag}__{exp_id}.npz", y_true=np.asarray(y_true, dtype=np.float64),
                 y_pred=np.asarray(y_pred, dtype=np.float64), times=times_to_int(times))


def save_scores(outputs, folder, tag):
    """score_copies çıktısı → seed başına skor matrisi [T, yöntem], etiket sütunları [T], zamanlar [T].

    Etiket tablosunda sayısal (is_anomaly) ve metin (anomaly_type) sütunlar vardır;
    her sütun kendi türüyle ayrı saklanır (metin birebir, sayı atol ile karşılaştırılır).
    """
    for seed, scores, labels, _ in outputs:
        label_frame = pd.DataFrame(labels)
        label_arrays = {}
        for column in label_frame.columns:
            values = label_frame[column]
            label_arrays[f"label_{column}"] = (values.to_numpy(dtype=np.float64)
                                               if pd.api.types.is_numeric_dtype(values)
                                               else values.astype(str).to_numpy(dtype=str))
        np.savez(folder / f"{tag}__seed{seed}.npz",
                 scores=scores.to_numpy(dtype=np.float64), columns=np.array(scores.columns, dtype=str),
                 times=times_to_int(scores.index), **label_arrays)


def household_forecasts(m, config, df, folder):
    """Final tahmin kadrosu (F0–F13-ens): val ve test tahminleri, kayıtlı ağırlıklarla."""
    run_config = deepcopy(config)
    run_config["forecast"]["xgboost"]["n_jobs"] = XGB_THREADS   # XGBoost iş parçacığı sabit
    for split in ("val", "test"):
        results = m["forecast"].all_predictions(df, run_config, split, torch.device("cpu"))
        save_forecasts(results, folder, f"household_forecast_{split}")


def household_anomaly(m, config, df, folder):
    """Final anomali protokolü ve A6: skorlar + kayıtlı seçimle seed başına metrikler."""
    device = torch.device("cpu")
    protocol = config["protocol"]
    tables_dir = m["results_io"].resolve_path(ROOT, config["paths"]["tables_dir"])
    splits = m["protocol"].protocol_splits(df, config)
    fitted = m["anomaly"].fit_detectors(splits, config, device, std_source=protocol["std_source"])
    sigma_table = m["inject"].train_sigma_table(splits["train"], config["preprocess"]["target"])
    combination = m["anomaly"].fit_combination(splits, config, fitted, device,
                                               protocol["combination_components"], reference_split="val_dev")
    chosen = pd.read_csv(tables_dir / "anomaly_protocol_selection.csv")
    windows = dict(zip(chosen["method"], chosen["smooth"].astype(int)))
    thresholds = dict(zip(chosen["method"], chosen["threshold"]))

    tables = []
    for split in ("val_dev", "val_hold", "test"):
        for scenario in ("main", "hard"):
            outputs = m["protocol"].score_copies(splits, config, split, scenario, fitted, device,
                                                 combination, sigma_table)
            save_scores(outputs, folder, f"household_anomaly_{split}_{scenario}")
            tables.append(m["protocol"].apply_fixed(outputs, windows, thresholds, split, scenario))
            if scenario == "main" and split in ("val_dev", "val_hold"):
                # A6: kayıtlı seçim (a6_selection.csv) aynen; test kümesinde A6 yok.
                settings = m["dual"].load_selection(tables_dir, splits["train"], config)
                copies = m["dual"].build_copies(outputs, config, settings["windows"])
                m["dual"].evaluate_methods(copies, settings, split).to_csv(
                    folder / f"table__a6_{split}.csv", index=False)
    pd.concat(tables, ignore_index=True).to_csv(folder / "table__household_anomaly_per_seed.csv", index=False)

    clean = m["anomaly"].score_frame(m["anomaly"].with_context(splits, "test", splits["test"], config),
                                     config, fitted, device)
    m["anomaly"].add_combinations(clean, combination)
    np.savez(folder / "household_anomaly_test_clean.npz", scores=clean.to_numpy(dtype=np.float64),
             columns=np.array(clean.columns, dtype=str), times=times_to_int(clean.index))


def eld(m, folder):
    """ELD: tahmin (val, test) ve anomali (val, test; kayıtlı seçim) – kayıtlı ELD ağırlıklarıyla."""
    device = torch.device("cpu")
    config = m["eld_data"].eld_config()
    config["forecast"]["xgboost"]["n_jobs"] = XGB_THREADS
    df = pd.read_parquet(m["results_io"].resolve_path(ROOT, config["paths"]["processed_file"]))
    eld_run = m["eld_run"]
    for split in ("val", "test"):
        results = m["forecast"].all_predictions(df, config, split, device, eld_run.ROSTER,
                                                eld_run.ENSEMBLE, prefix=eld_run.PREFIX)
        save_forecasts(results, folder, f"eld_forecast_{split}")

    tables_dir = m["results_io"].resolve_path(ROOT, config["paths"]["tables_dir"])
    splits = m["split"].chronological_split(df, config)
    fitted = m["anomaly"].fit_detectors(splits, config, device, std_source="oof")
    sigma_table = m["inject"].train_sigma_table(splits["train"], config["preprocess"]["target"])
    combination = m["anomaly"].fit_combination(splits, config, fitted, device,
                                               config["protocol"]["combination_components"],
                                               reference_split="val")
    chosen = pd.read_csv(tables_dir / "anomaly_selection.csv")
    windows = dict(zip(chosen["method"], chosen["smooth"].astype(int)))
    thresholds = dict(zip(chosen["method"], chosen["threshold"]))
    tables = []
    for split in ("val", "test"):
        outputs = m["protocol"].score_copies(splits, config, split, "main", fitted, device,
                                             combination, sigma_table)
        save_scores(outputs, folder, f"eld_anomaly_{split}_main")
        tables.append(m["protocol"].apply_fixed(outputs, windows, thresholds, split, "main"))
    pd.concat(tables, ignore_index=True).to_csv(folder / "table__eld_anomaly_per_seed.csv", index=False)


def demo(m, folder):
    """Demo servisleri: bir test günü tahmini (hane + ELD) ve tam dönem enjeksiyon analizi."""
    services, anomaly_services = m["services"], m["anomaly_services"]
    config, data_path, models_dir, tables_dir, features = anomaly_services.anomaly_demo_paths()
    frame = services.load_household_data(str(data_path))
    models = services.load_f13_models(str(models_dir), features)
    days = services.valid_test_days(frame, config)
    rows = {}
    for day in (days[0], days[len(days) // 2]):
        result = services.predict_test_day(frame, day, models)
        rows[f"household_day_{day}"] = np.concatenate([result["f13_kw"], result["sn168_kw"],
                                                       result["actual_kw"]])

    calibration = anomaly_services.load_anomaly_calibration(str(tables_dir))
    signature = anomaly_services.checkpoint_signature(models_dir)
    cases = [("A3-s2s-norm", "test"), ("A6-dual-floor", anomaly_services.EXPERIMENT_SPLIT)]
    for method, split in cases:
        period = anomaly_services.complete_test_days(frame, config, split)
        seed = int(config["protocol"]["scenarios"]["main"]["seeds"][split][0])
        result = anomaly_services.analyze_test_range(
            frame, config, pd.Timestamp(period[0]), pd.Timestamp(period[-1]) + pd.Timedelta(hours=23),
            True, "level_shift", "main", seed, calibration, signature, models, 1.0, split)
        outcomes, false_alarms = anomaly_services.event_outcomes(result, method)
        caught = int((outcomes["Durum"] == anomaly_services.CAUGHT).sum())
        rows[f"anomaly_{method}_{split}_scores"] = result["scores"][method].to_numpy(dtype=np.float64)
        rows[f"anomaly_{method}_{split}_counts"] = np.array([len(outcomes), caught, false_alarms],
                                                             dtype=np.float64)

    eld_config, eld_path = m["eld_services"].eld_paths()
    eld_frame = pd.read_parquet(eld_path)
    eld_results = m["eld_services"].eld_test_predictions(eld_frame, eld_config)
    eld_days = services.valid_test_days(eld_frame, eld_config)
    eld_day = m["eld_services"].predict_eld_day(eld_frame, eld_config, eld_results, eld_days[0])
    rows[f"eld_day_{eld_days[0]}"] = np.concatenate([eld_day["f13_kw"], eld_day["sn168_kw"]])
    np.savez(folder / "demo.npz", **{key: np.asarray(value, dtype=np.float64) for key, value in rows.items()})


def saved_predictions_check(config, m, out):
    """Bilgi amaçlı: yeniden üretilen hane tahminleri final koşuda kaydedilenlerle ne kadar aynı?

    results/predictions/ git'e girmez (yerel); yoksa atlanır. Sonuç yalnız
    info__saved_predictions.csv'ye yazılır, compare'de kullanılmaz (derin modeller
    final koşuda GPU'da üretildiği için ~1e-3 kW farklar beklenir).
    """
    saved_dir = m["results_io"].resolve_path(ROOT, config["paths"]["predictions_dir"])
    rows = []
    for path in sorted(out.glob("household_forecast_*__*.npz")):
        split = path.stem.split("__")[0].rsplit("_", 1)[1]
        exp_id = path.stem.split("__", 1)[1]
        saved = saved_dir / f"{exp_id}_{split}.npz"
        if not saved.exists():
            continue
        with np.load(path) as new, np.load(saved) as old:
            rows.append({"split": split, "exp_id": exp_id,
                         "same_windows": bool(np.array_equal(new["times"], old["target_times"])),
                         "max_abs_diff_kw": float(np.abs(new["y_pred"] - old["y_pred"]).max())})
    pd.DataFrame(rows).to_csv(out / "info__saved_predictions.csv", index=False)


def snapshot(out_dir):
    """Referans veya yeni anlık görüntüyü proje DIŞINDAKİ out_dir klasörüne yazar."""
    out = Path(out_dir).resolve()
    if out == ROOT or ROOT in out.parents:
        raise SystemExit("Çıktı klasörü proje içinde olamaz (results/ ve experiments.csv korunur).")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} boş değil; yeni bir klasör verin.")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(TORCH_THREADS)
    before = file_hashes()

    m = modules()
    config = m["config"].load_config()
    df = pd.read_parquet(m["results_io"].resolve_path(ROOT, config["paths"]["processed_file"]))
    steps = [("hane tahmin", lambda: household_forecasts(m, config, df, out)),
             ("hane anomali + A6", lambda: household_anomaly(m, config, df, out)),
             ("ELD", lambda: eld(m, out)),
             ("demo", lambda: demo(m, out))]
    for name, step in steps:
        print(f"[snapshot] {name} ...", flush=True)
        step()

    saved_predictions_check(config, m, out)
    after = file_hashes()
    if before != after:   # betik hiçbir izlenen dosyayı değiştirmemeli
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        raise SystemExit(f"Anlık görüntü sırasında proje dosyaları değişti: {changed}")
    manifest = {
        "hashes": before,
        "modules": {key: module.__name__ for key, module in m.items()},
        "threads": {"torch": TORCH_THREADS, "xgboost": XGB_THREADS},
        "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
        "xgboost": importlib.import_module("xgboost").__version__,
        "sklearn": importlib.import_module("sklearn").__version__,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[snapshot] {len(before)} dosya özeti, {len(list(out.glob('*.npz')))} dizi dosyası → {out}")


def compare_arrays(a, b):
    """İki .npz içeriğini karşılaştırır: anahtar, şekil, sayısal fark (atol), tam eşitlik (zaman/ad)."""
    problems, worst = [], 0.0
    if set(a.files) != set(b.files):
        return [f"anahtarlar farklı: {sorted(set(a.files) ^ set(b.files))}"], np.inf
    for key in a.files:
        x, y = a[key], b[key]
        if x.shape != y.shape:
            problems.append(f"{key}: şekil {x.shape} ≠ {y.shape}")
            continue
        if x.dtype.kind in "fc":
            nan_x, nan_y = np.isnan(x), np.isnan(y)
            if not np.array_equal(nan_x, nan_y):
                problems.append(f"{key}: NaN konumları farklı")
                continue
            diff = float(np.max(np.abs(x[~nan_x] - y[~nan_y]), initial=0.0))
            worst = max(worst, diff)
            if diff > ATOL:
                problems.append(f"{key}: en büyük fark {diff:.3g} > {ATOL}")
        elif not np.array_equal(x, y):
            problems.append(f"{key}: birebir değil")
    return problems, worst


def compare_tables(path_a, path_b):
    """İki CSV: aynı sütun/satır; sayısal sütunlarda atol, diğerlerinde tam eşitlik."""
    a, b = pd.read_csv(path_a), pd.read_csv(path_b)
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        return [f"şekil/sütun farklı: {a.shape} ≠ {b.shape}"], np.inf
    problems, worst = [], 0.0
    for column in a.columns:
        if pd.api.types.is_numeric_dtype(a[column]) and pd.api.types.is_numeric_dtype(b[column]):
            x, y = a[column].to_numpy(dtype=np.float64), b[column].to_numpy(dtype=np.float64)
            if not np.array_equal(np.isnan(x), np.isnan(y)):
                problems.append(f"{column}: NaN konumları farklı")
                continue
            diff = float(np.nanmax(np.abs(x - y), initial=0.0))
            worst = max(worst, diff)
            if diff > ATOL:
                problems.append(f"{column}: en büyük fark {diff:.3g}")
        elif not a[column].astype(str).equals(b[column].astype(str)):
            problems.append(f"{column}: metin farklı")
    return problems, worst


def compare(ref_dir, new_dir):
    """İki anlık görüntüyü karşılaştırır; özet tabloyu basar, uyuşmazlıkta çıkış kodu 1."""
    ref, new = Path(ref_dir), Path(new_dir)
    ref_manifest = json.loads((ref / "manifest.json").read_text(encoding="utf-8"))
    new_manifest = json.loads((new / "manifest.json").read_text(encoding="utf-8"))
    rows = []

    ref_hashes, new_hashes = ref_manifest["hashes"], new_manifest["hashes"]
    for path in sorted(set(ref_hashes) | set(new_hashes)):
        same = ref_hashes.get(path) == new_hashes.get(path)
        rows.append({"grup": "özet", "öğe": path, "durum": "TAMAM" if same else "FARKLI",
                     "en_büyük_fark": 0.0 if same else np.nan,
                     "not": "" if same else ("eksik" if path not in new_hashes or path not in ref_hashes
                                             else "SHA-256 farklı")})
    for key in ("threads", "torch", "numpy", "pandas", "xgboost", "sklearn"):
        if ref_manifest[key] != new_manifest[key]:
            rows.append({"grup": "ortam", "öğe": key, "durum": "FARKLI", "en_büyük_fark": np.nan,
                         "not": f"{ref_manifest[key]} ≠ {new_manifest[key]}"})

    names = sorted({p.name for p in ref.glob("*.npz")} | {p.name for p in new.glob("*.npz")})
    for name in names:
        if not (ref / name).exists() or not (new / name).exists():
            rows.append({"grup": "dizi", "öğe": name, "durum": "FARKLI", "en_büyük_fark": np.nan, "not": "eksik"})
            continue
        with np.load(ref / name) as a, np.load(new / name) as b:
            problems, worst = compare_arrays(a, b)
        rows.append({"grup": "dizi", "öğe": name, "durum": "FARKLI" if problems else "TAMAM",
                     "en_büyük_fark": worst, "not": "; ".join(problems)[:200]})
    for path in sorted(ref.glob("table__*.csv")):
        other = new / path.name
        problems, worst = (["eksik"], np.nan) if not other.exists() else compare_tables(path, other)
        rows.append({"grup": "tablo", "öğe": path.name, "durum": "FARKLI" if problems else "TAMAM",
                     "en_büyük_fark": worst, "not": "; ".join(problems)[:200]})

    table = pd.DataFrame(rows)
    table.to_csv(new / "compare_result.csv", index=False)
    summary = table.groupby("grup").agg(öğe=("öğe", "count"), farklı=("durum", lambda s: int((s != "TAMAM").sum())),
                                        en_büyük_fark=("en_büyük_fark", "max"))
    print(summary.to_string())
    failed = table[table["durum"] != "TAMAM"]
    if len(failed):
        print("\nUYUŞMAZLIK:")
        print(failed.to_string(index=False))
        raise SystemExit(1)
    print(f"\nTAMAM: {len(table)} öğe; özetler birebir, diziler ve tablolar atol = {ATOL} içinde.")


def main():
    """snapshot | compare alt komutları."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("snapshot").add_argument("--out", required=True)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("ref")
    compare_parser.add_argument("new")
    args = parser.parse_args()
    if args.command == "snapshot":
        snapshot(args.out)
    else:
        compare(args.ref, args.new)


if __name__ == "__main__":
    main()
