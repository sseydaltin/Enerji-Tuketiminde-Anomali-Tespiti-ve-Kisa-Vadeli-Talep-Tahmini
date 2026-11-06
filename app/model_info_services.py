"""Model Info sayfası için kaynak dosyalara dayalı, salt okunur özetler."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.load import PROJECT_ROOT


def _read_yaml(path):
    """YAML yapılandırmasını güvenli biçimde okur.

    Girdi: proje içindeki YAML dosyasının yolu.
    Çıktı: YAML içeriğini temsil eden sözlük.
    Neden: Sayfadaki teknik bilgiler yapılandırmadan doğrulanarak gösterilir.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Yapılandırma dosyası bulunamadı: {path}")
    with path.open(encoding="utf-8") as file:
        content = yaml.safe_load(file)
    if not isinstance(content, dict):
        raise ValueError(f"Yapılandırma sözlük biçiminde değil: {path}")
    return content


def _checkpoint_metadata(path):
    """Checkpoint'i güvenli modda okur, ağırlıkları dışarı vermeden metadata döndürür.

    Girdi: `.pt` checkpoint yolu.
    Çıktı: model adı, özellikler, ölçek istatistikleri ve forecast metadata'sı.
    Neden: Bilgi sayfası mimari kurmaz, ağırlıkları yükleyip inference çalıştırmaz.
    """
    if not path.is_file():
        return {"available": False, "path": str(path)}

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "model_name", "features", "feature_mean", "feature_scale",
        "target_mean", "target_scale", "forecast",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"{path} metadata alanları eksik: {sorted(missing)}")

    features = checkpoint["features"]
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float64)
    scale = np.asarray(checkpoint["feature_scale"], dtype=np.float64)
    target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float64)
    target_scale = np.asarray(checkpoint["target_scale"], dtype=np.float64)
    if (
        not isinstance(features, list)
        or mean.shape != (len(features),)
        or scale.shape != (len(features),)
        or target_mean.shape != (1,)
        or target_scale.shape != (1,)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
        or not np.isfinite(target_mean).all()
        or not np.isfinite(target_scale).all()
        or (target_scale <= 0).any()
        or not isinstance(checkpoint["forecast"], dict)
    ):
        raise ValueError(f"{path} checkpoint metadata biçimi geçersiz.")

    return {
        "available": True,
        "path": str(path),
        "model_name": checkpoint["model_name"],
        "features": features,
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "target_mean": target_mean.tolist(),
        "target_scale": target_scale.tolist(),
        "forecast": checkpoint["forecast"],
    }


def _read_thresholds(path, methods):
    """Validation seçim CSV'sinden istenen yöntemlerin eşiklerini doğrular.

    Girdi: seçim CSV'si, sayfada gösterilecek yöntemler ve final satırı filtresi.
    Çıktı: doğrulanmış, yalnız istenen yöntemlerden oluşan DataFrame.
    Neden: Test kümesinden eşik türetilmesini engeller ve eksik dosyayı görünür kılar.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Validation eşik dosyası bulunamadı: {path}")
    frame = pd.read_csv(path)
    required = {"method", "smooth", "threshold"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} gerekli sütunları içermiyor: {sorted(missing)}")

    selected = frame.loc[frame["method"].isin(methods)].copy()
    if selected["method"].duplicated().any() or set(selected["method"]) != set(methods):
        raise ValueError(
            f"{path} istenen yöntemler için tekil seçim içermiyor: {list(methods)}"
        )
    selected["threshold"] = pd.to_numeric(selected["threshold"], errors="raise")
    selected["smooth"] = pd.to_numeric(selected["smooth"], errors="raise")
    return selected.reset_index(drop=True)


def _validate_optuna_checkpoint(best_path, checkpoint):
    """Optuna'nın seçilmiş parametrelerini checkpoint metadata'sıyla karşılaştırır.

    Girdi: Optuna best CSV yolu ve ilgili tahmin checkpoint metadata'sı.
    Çıktı: Optuna satırındaki seçilmiş hiperparametreler.
    Neden: Deney parametresiyle varsayılan config'in karışmasını önler.
    """
    if not best_path.is_file():
        raise FileNotFoundError(f"Optuna best sonuç dosyası bulunamadı: {best_path}")
    frame = pd.read_csv(best_path)
    if len(frame) != 1:
        raise ValueError(f"{best_path} tek bir best satırı içermeli.")
    row = frame.iloc[0]
    fields = ("hidden", "layers", "lr", "batch_size", "dropout", "weight_decay", "loss")
    missing = set(fields) - set(frame.columns)
    if missing:
        raise ValueError(f"{best_path} gerekli sütunları içermiyor: {sorted(missing)}")

    forecast = checkpoint["forecast"]
    checkpoint_fields = {
        "hidden": "hidden",
        "layers": "layers",
        "lr": "lr",
        "batch_size": "batch_size",
        "dropout": "dropout",
        "weight_decay": "weight_decay",
        "loss": "loss",
    }
    for csv_field, checkpoint_field in checkpoint_fields.items():
        saved = row[csv_field]
        actual = forecast[checkpoint_field]
        if isinstance(actual, (float, int, np.number)):
            if not np.isclose(float(saved), float(actual), rtol=1e-5, atol=1e-8):
                raise ValueError(
                    f"{best_path} {csv_field} değeri checkpoint metadata'sıyla uyuşmuyor."
                )
        elif str(saved) != str(actual):
            raise ValueError(
                f"{best_path} {csv_field} değeri checkpoint metadata'sıyla uyuşmuyor."
            )
    return {field: row[field].item() if hasattr(row[field], "item") else row[field]
            for field in fields}


def load_model_info(project_root=PROJECT_ROOT):
    """Model Info için config, checkpoint metadata ve kayıtlı val eşiklerini toplar.

    Girdi: depo kökü (testlerde geçici depo kullanabilmek için).
    Çıktı: yalnız kaynak dosyalardan okunan, eğitim/inference içermeyen bilgi sözlüğü.
    Neden: Sayfanın tüm teknik iddiaları izlenebilir dosya kaynaklarına dayanır.
    """
    root = Path(project_root)
    household = _read_yaml(root / "configs/config.yaml")
    eld = _read_yaml(root / "configs/eld.yaml")

    checkpoints = {
        "F5-opt": _checkpoint_metadata(root / "models/F5-opt_best.pt"),
        "F10-opt": _checkpoint_metadata(root / "models/F10-opt_best.pt"),
        "ELD-F5-opt": _checkpoint_metadata(root / "models/eld/ELD-F5-opt_best.pt"),
        "ELD-F10-opt": _checkpoint_metadata(root / "models/eld/ELD-F10-opt_best.pt"),
    }
    for name in ("F5-opt", "F10-opt"):
        if not checkpoints[name]["available"]:
            continue
        checkpoints[name]["optuna"] = _validate_optuna_checkpoint(
            root / f"results/tables/optuna_{'lstm' if name == 'F5-opt' else 'seq2seq'}_best.csv",
            checkpoints[name],
        )

    model_dir = root / household["paths"]["models_dir"]
    xgb_artifacts = [
        path.relative_to(root).as_posix()
        for path in model_dir.rglob("*")
        if path.is_file()
        and (
            "xgb" in path.name.lower()
            or path.suffix.lower() in {".ubj", ".model"}
        )
    ]

    anomaly_methods = ("A0", "A3-s2s-norm", "A5-mean-z")
    household_thresholds = _read_thresholds(
        root / "results/tables/anomaly_protocol_selection.csv",
        anomaly_methods,
    )
    eld_thresholds = _read_thresholds(
        root / "results/eld/tables/anomaly_selection.csv",
        anomaly_methods,
    )

    household_processed = root / household["paths"]["processed_file"]
    eld_processed = root / eld["paths"]["processed_file"]
    household_raw = root / household["paths"]["raw_file"]
    eld_raw = root / eld["eld"]["raw_file"]

    return {
        "root": str(root),
        "household": household,
        "eld": eld,
        "checkpoints": checkpoints,
        "xgb_artifacts": xgb_artifacts,
        "thresholds": {
            "household": household_thresholds,
            "eld": eld_thresholds,
        },
        "availability": {
            "household_raw": household_raw.is_file(),
            "household_processed": household_processed.is_file(),
            "eld_raw": eld_raw.is_file(),
            "eld_processed": eld_processed.is_file(),
        },
        "paths": {
            "household_raw": str(household_raw),
            "household_processed": str(household_processed),
            "eld_raw": str(eld_raw),
            "eld_processed": str(eld_processed),
            "xgb_model_dir": str(model_dir),
        },
    }
