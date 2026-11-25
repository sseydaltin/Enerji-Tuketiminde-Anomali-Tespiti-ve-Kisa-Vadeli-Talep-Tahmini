"""configs/config.yaml dosyasını okuyan yardımcı fonksiyon."""

from pathlib import Path

import yaml

# Proje kökü: bu dosya src/utils/config.py -> iki üst klasör yukarı
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """YAML yapılandırma dosyasını sözlük (dict) olarak döndürür.

    Girdi:
        path: config dosyasının yolu (varsayılan: configs/config.yaml).
    Çıktı:
        dict, ör. cfg["forecast"]["lookback"] -> 168

    Neden: Tüm yollar, tarihler ve hiperparametreler tek dosyada durur;
    böylece hiçbir sayı koda gömülmez ve her deney tekrarlanabilir.
    `safe_load` kullanılır çünkü YAML içinden kod çalıştırılmasına izin vermez.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
