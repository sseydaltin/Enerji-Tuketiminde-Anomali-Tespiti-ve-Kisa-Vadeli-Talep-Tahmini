"""setup_console testleri: Türkçe karakterler cp1252 konsolda da basılabilmeli."""

import importlib
import inspect
import io
import sys

import pytest

from src.utils.console import setup_console


def test_setup_console_switches_cp1252_to_utf8(monkeypatch):
    """cp1252 kodlamalı sahte bir konsol "İ" basamaz; setup_console sonrası basabilir."""
    fake_stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    fake_stderr = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", fake_stdout)
    monkeypatch.setattr(sys, "stderr", fake_stderr)

    # Önce: sorunun gerçekten var olduğunu gösteriyoruz.
    with pytest.raises(UnicodeEncodeError):
        fake_stdout.write("SAATLİK VERİ ÖZETİ")

    setup_console()

    assert fake_stdout.encoding == "utf-8"
    assert fake_stderr.encoding == "utf-8"

    fake_stdout.write("SAATLİK VERİ ÖZETİ")
    fake_stdout.flush()
    assert fake_stdout.buffer.getvalue().decode("utf-8") == "SAATLİK VERİ ÖZETİ"


@pytest.mark.parametrize(
    "module_name",
    [
        "src.data.load",
        "src.data.preprocess",
        "src.data.split",
        "src.data.windows",
        "src.models.baselines",
    ],
)
def test_scripts_call_setup_console(module_name):
    """python -m ile çalışan her betiğin main() fonksiyonu setup_console() çağırmalı."""
    module = importlib.import_module(module_name)
    assert "setup_console()" in inspect.getsource(module.main)
