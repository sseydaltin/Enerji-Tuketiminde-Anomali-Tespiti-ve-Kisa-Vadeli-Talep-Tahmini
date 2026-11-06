"""Demo anomali kalibrasyonunu final protokolün kendi fonksiyonlarıyla dosyaya yazar.

Çalıştırma: python -m app.build_calibration
Çıktı: results/tables/demo_oof_std_{A3-s2s,A3-ens}.csv ve demo_a5_reference.csv

Neden: Eşikler OOF std ve val-dev referansıyla seçildi. Demo da aynı ölçeği kullanmalı;
ama her açılışta Isolation Forest ve kat modellerini çalıştırmamalı. Yeni hesap yoktur:
fit_detectors ve fit_combination, final testteki çağrıyla aynı biçimde çağrılır. Test
kümesi kullanılmaz; std yalnız eğitimden, referans yalnız enjeksiyonsuz val-dev'den gelir.
"""

import pandas as pd
import torch

from src.anomaly_protocol import protocol_splits
from src.data.load import PROJECT_ROOT, load_config
from src.evaluate_anomaly import fit_combination, fit_detectors
from src.utils.console import setup_console
from src.utils.results_io import resolve_path

STD_FILE = "demo_oof_std_{name}.csv"
REFERENCE_FILE = "demo_a5_reference.csv"


def build_calibration(config, device):
    """Protokol kalibrasyonunu üretir.

    Girdi: config, cihaz.
    Çıktı: ({"A3-s2s": tablo, "A3-ens": tablo}, [component, score] referans tablosu).
    """
    frame = pd.read_parquet(resolve_path(PROJECT_ROOT, config["paths"]["processed_file"]))
    protocol = config["protocol"]
    splits = protocol_splits(frame, config)
    fitted = fit_detectors(splits, config, device, std_source=protocol["std_source"])
    combination = fit_combination(
        splits, config, fitted, device, protocol["combination_components"], reference_split="val_dev",
    )
    reference = pd.concat(
        pd.DataFrame({"component": name, "score": values})
        for name, values in combination["references"].items()
    )
    return fitted["std_tables"], reference


def main():
    """Tabloları results/tables altına yazar ve özetini basar."""
    setup_console()
    config = load_config()
    tables_dir = resolve_path(PROJECT_ROOT, config["paths"]["tables_dir"])
    std_tables, reference = build_calibration(config, torch.device("cpu"))
    for name, table in std_tables.items():
        table.to_csv(tables_dir / STD_FILE.format(name=name), index=False)
        print(name, "ortalama std:", round(float(table["std"].mean()), 4))
    reference.to_csv(tables_dir / REFERENCE_FILE, index=False)
    print(reference.groupby("component")["score"].agg(["count", "mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
