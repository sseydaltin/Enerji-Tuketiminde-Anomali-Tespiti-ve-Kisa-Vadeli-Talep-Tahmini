"""Tekrarlanabilirlik: tüm rastgelelik kaynaklarını tek bir seed ile sabitler."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Python, NumPy ve PyTorch'un rastgele sayı üreteçlerini sabitler.

    Girdi:
        seed: tamsayı tohum (config.yaml'daki `seed`, varsayılan 42).
    Çıktı:
        Yok (global durumu değiştirir).

    Neden: Ağırlık başlatma, dropout ve DataLoader karıştırması rastgeledir.
    Seed sabitlenmezse aynı kod her çalıştırmada farklı sonuç verir ve
    deneyler karşılaştırılamaz (README §1, Tekrarlanabilirlik).
    """
    random.seed(seed)                    # Python'un random modülü
    np.random.seed(seed)                 # NumPy (ör. enjeksiyon, örnekleme)
    torch.manual_seed(seed)              # PyTorch CPU
    torch.cuda.manual_seed_all(seed)     # PyTorch GPU (GPU yoksa etkisiz)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # cuDNN bazı işlemlerde en hızlı (ama deterministik olmayan) algoritmayı seçer.
    # Biraz hız kaybı pahasına her seferinde aynı algoritmayı kullanmaya zorluyoruz.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(preference="cuda") -> torch.device:
    """Değerlendirme betiklerinin cihazı: CUDA varsa istenen cihaz, yoksa CPU.

    Girdi: --device argümanı (ör. "cuda", "cpu").
    Çıktı: torch.device(preference) CUDA varsa; yoksa torch.device("cpu").
    Neden: Aynı satır sekiz betikte tekrar yazılmıştı; tek fonksiyonda toplandı,
    davranış aynen korundu. Eğitim betiği (train_forecast) "auto"/"mps" seçeneği
    olduğu için kendi seçimini yapar; A6 (anomaly_dual) bilerek her zaman CPU kullanır.
    """
    return torch.device(preference if torch.cuda.is_available() else "cpu")
