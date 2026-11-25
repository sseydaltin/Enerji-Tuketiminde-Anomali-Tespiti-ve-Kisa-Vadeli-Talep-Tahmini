"""Proje kökü (PROJECT_ROOT) ve configs/config.yaml okuma: projedeki TEK tanım.

Eskiden iki ayrı kopya vardı (src/data/load.py ve src/utils/config.py); 2026-11-25
yeniden yapılandırmasında burada birleştirildi.
"""

from pathlib import Path

import yaml

# Proje kökünü tanıyan işaret dosyası (göreli yol).
ROOT_MARKER = Path("configs") / "config.yaml"


def find_project_root(start):
    """`start`tan yukarı doğru, içinde configs/config.yaml olan ilk klasörü döndürür.

    Girdi: bir dosya veya klasör yolu.
    Çıktı: proje kökü (Path).
    Neden: Kök, "bu dosyadan iki klasör yukarı" (parents[2]) gibi dosyanın derinliğine
    bağlı bir sayıyla bulunursa dosya taşındığında sessizce yanlış klasörü gösterir.
    İşaret dosyasını aramak dosyanın konumundan bağımsızdır; bulunamazsa açık hata verir.
    """
    start = Path(start).resolve()
    for folder in (start, *start.parents):
        if (folder / ROOT_MARKER).is_file():
            return folder
    raise FileNotFoundError(f"{start} ve üst klasörlerinde {ROOT_MARKER.as_posix()} bulunamadı.")


PROJECT_ROOT = find_project_root(Path(__file__).parent)
DEFAULT_CONFIG_PATH = PROJECT_ROOT / ROOT_MARKER


def load_config(config_path=None):
    """YAML yapılandırma dosyasını sözlük (dict) olarak döndürür.

    Girdi: config dosyasının yolu; None ise configs/config.yaml.
    Çıktı: dict, ör. cfg["forecast"]["horizon"] → 24.
    Neden: Bütün yollar, tarihler ve hiperparametreler tek dosyada durur; hiçbir sayı
    koda gömülmez. `safe_load`, YAML içinden kod çalıştırılmasına izin vermez.
    """
    path = DEFAULT_CONFIG_PATH if config_path is None else config_path
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
