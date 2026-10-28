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


def get_device() -> torch.device:
    """Eğitim/çıkarım cihazını döndürür: CUDA varsa GPU, yoksa CPU.

    Çıktı:
        torch.device("cuda") veya torch.device("cpu")
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
