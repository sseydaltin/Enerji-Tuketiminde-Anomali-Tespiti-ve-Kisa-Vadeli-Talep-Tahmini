"""PROJECT_ROOT dosya konumundan bağımsız mı? Tek tanım ve get_device davranışı."""

from pathlib import Path

import pytest
import torch

from src.core import config as core_config
from src.core.config import PROJECT_ROOT, find_project_root, load_config
from src.core.seed import get_device


def test_project_root_is_repository_root():
    """PROJECT_ROOT, bu test dosyasının bir üstündeki klasör (repo kökü) olmalı."""
    assert PROJECT_ROOT == Path(__file__).resolve().parents[1]
    assert (PROJECT_ROOT / "configs/config.yaml").is_file()


def test_find_project_root_does_not_depend_on_depth(tmp_path):
    """Aynı kök, farklı derinlikteki klasörlerden aranınca bulunur (parents[k] sayısına bağlı değil)."""
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/config.yaml").write_text("seed: 1\n", encoding="utf-8")
    shallow = tmp_path / "src"
    deep = tmp_path / "src/datasets/household/sub"
    deep.mkdir(parents=True)
    assert find_project_root(shallow) == tmp_path
    assert find_project_root(deep) == tmp_path
    assert find_project_root(deep / "load.py") == tmp_path      # dosya yolu da olur


def test_find_project_root_fails_loudly(tmp_path):
    """İşaret dosyası yoksa sessizce yanlış klasör değil, açık hata."""
    with pytest.raises(FileNotFoundError):
        find_project_root(tmp_path)


def test_single_definition_is_reexported_everywhere():
    """Veri modülleri kendi kopyalarını değil, core'daki tek tanımı kullanır."""
    import src.data.load as household_load
    assert household_load.PROJECT_ROOT is PROJECT_ROOT
    assert household_load.load_config is core_config.load_config
    assert load_config()["seed"] == 42


def test_get_device_matches_previous_inline_rule():
    """get_device(x) eski satırla aynı: CUDA varsa x, yoksa cpu."""
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert get_device("cuda").type == expected
    assert get_device("cpu").type == "cpu"
