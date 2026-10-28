"""Kurulum testleri: config okunuyor mu, split tarihleri tutarlı mı, seed çalışıyor mu."""

import numpy as np
import pandas as pd
import torch

from src.utils.config import load_config
from src.utils.seed import set_seed


def test_config_loads_required_sections():
    """config.yaml okunabiliyor ve ana bölümler mevcut."""
    cfg = load_config()
    for key in ["seed", "paths", "preprocess", "split", "features", "forecast", "anomaly", "injection"]:
        assert key in cfg


def test_split_dates_are_chronological_and_disjoint():
    """train < val < test sırası korunuyor ve tarih aralıkları kesişmiyor.

    Bitiş günleri dahil olduğu için bir sonraki kümenin başlangıcı,
    öncekinin bitişinden kesinlikle sonra olmalı.
    """
    split = load_config()["split"]
    previous_end = None
    for name in ["train", "val", "test"]:
        start = pd.Timestamp(split[name]["start"])
        end = pd.Timestamp(split[name]["end"])
        assert start <= end, f"{name}: başlangıç bitişten sonra"
        if previous_end is not None:
            assert start > previous_end, f"{name} önceki kümeyle kesişiyor"
        previous_end = end


def test_injection_seeds_differ():
    """Doğrulama ve test enjeksiyonu farklı seed kullanmalı (CLAUDE.md kural 7)."""
    inj = load_config()["injection"]
    assert inj["val_seed"] != inj["test_seed"]


def test_set_seed_is_reproducible():
    """Aynı seed ile NumPy ve PyTorch aynı rastgele sayıları üretmeli."""
    set_seed(42)
    a_np, a_torch = np.random.rand(5), torch.rand(5)
    set_seed(42)
    b_np, b_torch = np.random.rand(5), torch.rand(5)
    assert np.array_equal(a_np, b_np)
    assert torch.equal(a_torch, b_torch)
