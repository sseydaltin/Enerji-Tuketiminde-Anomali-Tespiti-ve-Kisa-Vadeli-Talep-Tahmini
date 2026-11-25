"""Model Info kaynak doğrulama ve salt-okunur sayfa testleri."""

from pathlib import Path

import numpy as np
import pytest
import torch

from app.model_info_services import _checkpoint_metadata, load_model_info
from app.pages.model_info import _flow_figure, _model_rows


def _render_model_info_page():
    """AppTest için yalnızca Model Info sayfasını çalıştırır."""
    from app.pages.model_info import render

    render()


def test_model_info_reads_metadata_safely_without_running_models(monkeypatch):
    """Tahmin checkpoint'lerinin güvenli metadata'sı okunur; model inference çağrılmaz."""
    root = Path(__file__).resolve().parents[1]
    required_checkpoints = (
        root / "models/F5-opt_best.pt",
        root / "models/F10-opt_best.pt",
    )
    if not all(path.is_file() for path in required_checkpoints):
        pytest.skip("Yerel Household checkpoint dosyaları bulunmuyor.")

    original_load = torch.load
    load_calls = []

    def safe_load(*args, **kwargs):
        load_calls.append(kwargs.copy())
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", safe_load)
    monkeypatch.setattr(torch.nn.Module, "forward", pytest.fail)

    info = load_model_info()

    # ELD (Şebeke) ağırlıkları isteğe bağlıdır ve repoya girmez (README §4.13); yoksa sayfa
    # bunu "kullanılamıyor" diye gösterir. Bu yüzden beklenen okuma sayısı yerel dosyalara göre.
    eld_present = sum((root / f"models/eld/{name}_best.pt").is_file()
                      for name in ("ELD-F5-opt", "ELD-F10-opt"))
    assert len(load_calls) == 2 + eld_present
    assert all(call == {"map_location": "cpu", "weights_only": True} for call in load_calls)
    f5 = info["checkpoints"]["F5-opt"]
    f10 = info["checkpoints"]["F10-opt"]
    assert f5["available"] and f10["available"]
    assert f5["model_name"] == "lstm"
    assert f10["model_name"] == "seq2seq"
    assert f5["features"] == f10["features"]
    assert f5["forecast"]["lookback"] == f10["forecast"]["lookback"] == 48
    assert f5["forecast"]["horizon"] == f10["forecast"]["horizon"] == 24
    assert np.isfinite(f5["feature_mean"]).all()
    assert np.isfinite(f10["feature_scale"]).all()
    assert _model_rows(info)[0]["Model"] == "LSTM (F5-opt)"


def test_model_info_reports_dataset_and_model_artifact_availability():
    """Veri dosyası ve XGBoost ağırlık durumu gerçek yerel dosya kaynaklarıyla eşleşir."""
    info = load_model_info()
    root = Path(info["root"])

    assert info["availability"]["household_raw"] == (
        root / "data/raw/household_power_consumption.txt"
    ).is_file()
    assert info["availability"]["household_processed"] == (
        root / "data/processed/hourly.parquet"
    ).is_file()
    # Şebeke verisi repoya girmez; bazı bilgisayarlarda var, bazılarında yoktur.
    assert info["availability"]["eld_raw"] == (root / info["eld"]["eld"]["raw_file"]).is_file()
    assert info["availability"]["eld_processed"] == (
        root / info["eld"]["paths"]["processed_file"]
    ).is_file()
    assert info["checkpoints"]["ELD-F5-opt"]["available"] == (
        root / "models/eld/ELD-F5-opt_best.pt"
    ).is_file()
    assert info["checkpoints"]["ELD-F10-opt"]["available"] == (
        root / "models/eld/ELD-F10-opt_best.pt"
    ).is_file()
    assert info["xgb_artifacts"] == []


def test_model_info_uses_only_saved_validation_threshold_sources():
    """Gösterilen üç yöntemin eşikleri kayıtlı validation seçim tablolarından gelir."""
    info = load_model_info()

    household = info["thresholds"]["household"].set_index("method")
    eld = info["thresholds"]["eld"].set_index("method")
    assert set(household.index) == {"A0", "A3-s2s-norm", "A5-mean-z"}
    assert set(eld.index) == {"A0", "A3-s2s-norm", "A5-mean-z"}
    assert np.isclose(household.loc["A3-s2s-norm", "threshold"], 3.557971)
    assert np.isclose(eld.loc["A3-s2s-norm", "threshold"], 8.416469)


def test_model_info_flow_diagram_has_readable_non_overlapping_nodes_and_true_edges():
    """Şema büyük, çakışmayan kutular ve yalnız gerçek işlem akışına ait oklar içerir."""
    figure = _flow_figure()

    meta = figure.layout.meta
    assert meta is not None
    node_names = set(meta["flow_nodes"])
    edges = {tuple(edge) for edge in meta["flow_edges"]}
    labels = {
        annotation.text
        for annotation in figure.layout.annotations
        if not annotation.showarrow
    }

    assert {"Saatlik veri", "Kronolojik bölme"} <= {
        label.split("</b>")[0].removeprefix("<b>") for label in labels
        if "</b>" in label
    }
    assert any("Yalnız eğitim" in label for label in labels)
    assert any("OOF artıklar" in label for label in labels)
    assert any("Doğrulama seçimi" in label for label in labels)
    assert any("Final metrikler" in label for label in labels)
    assert {
        ("train_scaler", "scaler_fit"),
        ("train_model", "train_windows"),
        ("train_windows", "fit_weights"),
        ("train_folds", "fold_models"),
        ("fold_models", "oof_residual"),
        ("val_scores", "val_select"),
        ("candidate", "frozen"),
        ("val_select", "frozen"),
        ("frozen", "test_apply"),
    } <= edges
    assert not any(
        source.startswith("test_") and target.startswith("val_")
        for source, target in edges
    )
    assert not any(
        source.startswith("val_") and target.startswith("test_")
        for source, target in edges
    )

    arrow_annotations = [
        annotation for annotation in figure.layout.annotations
        if annotation.showarrow
    ]
    assert len(arrow_annotations) == len(edges)
    node_shapes = list(figure.layout.shapes)[3:]
    assert len(node_shapes) == len(node_names)
    for shape in node_shapes:
        assert shape.x1 - shape.x0 >= 130
        assert shape.y1 - shape.y0 >= 50

    for index, first in enumerate(node_shapes):
        for second in node_shapes[index + 1:]:
            overlaps_x = first.x0 < second.x1 and second.x0 < first.x1
            overlaps_y = first.y0 < second.y1 and second.y0 < first.y1
            assert not (overlaps_x and overlaps_y)


def test_model_info_page_renders_without_errors():
    """Model Info kullanıcı arayüzü, Forecast sayfasını çalıştırmadan hatasız render olur."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_function(_render_model_info_page).run()

    assert not app.exception
    assert [element.value for element in app.title] == ["Model Bilgisi"]
    assert len(app.get("plotly_chart")) == 3
    assert len(app.table) == 2
    assert len(app.dataframe) == 3
    warning_text = " ".join(element.value for element in app.warning).lower()
    assert "kaynak notu" in warning_text
    assert "reference_split='val_dev'" in warning_text
    assert "bağımsız çalıştırma günlüğü bulunmadığından" in warning_text


def test_missing_checkpoint_metadata_is_reported(tmp_path):
    """Bozuk checkpoint sessizce atlanmaz, açık doğrulama hatası üretir."""
    path = tmp_path / "incomplete.pt"
    torch.save({"model_name": "lstm"}, path)

    with pytest.raises(ValueError, match="metadata alanları eksik"):
        _checkpoint_metadata(path)
