"""A7 tablosunun başlığı ve notu betikten üretilir: yeniden üretim kayıtlı dosyayla birebir aynı olmalı."""

import pandas as pd

from experiments.archive.explore_tcn_ae import TABLE_NOTE, TABLE_TITLE
from src.anomaly_protocol import write_markdown
from src.data.load import PROJECT_ROOT


def test_a7_markdown_regenerates_saved_file(tmp_path):
    """Kayıtlı CSV'den yazılan Markdown, başlık ve not dahil kayıtlı .md ile aynı (satır sonu hariç)."""
    tables = PROJECT_ROOT / "results/tables"
    table = pd.read_csv(tables / "A7_explore_side_by_side.csv")
    out = tmp_path / "a7.md"
    write_markdown(table, out, title=TABLE_TITLE, note=TABLE_NOTE)
    saved = (tables / "A7_explore_side_by_side.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert out.read_text(encoding="utf-8").replace("\r\n", "\n") == saved
    assert "final sonuç değildir" in saved


def test_write_markdown_without_title_is_unchanged(tmp_path):
    """Başlık/not verilmezse çıktı eski biçimle aynı (yalnız tablo)."""
    out = tmp_path / "t.md"
    write_markdown(pd.DataFrame({"a": [1], "b": ["x"]}), out)
    assert out.read_text(encoding="utf-8").splitlines() == ["| a | b |", "|---|---|", "| 1 | x |"]
