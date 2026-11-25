"""Eski CLI yolları (geçici yönlendirmeler): hedef modül var mı, yönlendirme import edilince bir şey çalıştırıyor mu?"""

import ast
import importlib.util
from pathlib import Path

from src.core.config import PROJECT_ROOT

REDIRECTS = sorted(
    path for path in [*(PROJECT_ROOT / "src").glob("*.py"), *(PROJECT_ROOT / "src/data").glob("*.py")]
    if path.name != "__init__.py"
)


def _target(path):
    """Yönlendirme dosyasındaki TARGET sabitini AST ile okur (dosya çalıştırılmaz)."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and node.targets[0].id == "TARGET":
            return node.value.value
    raise AssertionError(f"{path.name}: TARGET yok")


def test_every_redirect_points_to_existing_module():
    """34 eski komutun her biri var olan yeni modüle gider."""
    assert len(REDIRECTS) == 34
    for path in REDIRECTS:
        assert importlib.util.find_spec(_target(path)) is not None, path


def test_redirects_have_no_library_code():
    """Yönlendirmeler yalnız runpy çağırır: fonksiyon/sınıf tanımı yok (import için kullanılamaz)."""
    for path in REDIRECTS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert not [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))], path
