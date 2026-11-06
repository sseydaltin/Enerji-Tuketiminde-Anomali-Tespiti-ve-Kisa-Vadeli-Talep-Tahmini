"""Streamlit Forecast servisinin test penceresi entegrasyon testleri."""

import numpy as np
import pandas as pd
import pytest
import torch

from app.services import (
    demo_paths,
    load_f13_models,
    load_household_data,
    predict_test_day,
    valid_test_days,
)


@pytest.fixture(scope="module")
def household_demo():
    """Gerçek yerel household tablosu, config ve train checkpoint'lerini yükler."""
    config, data_path, models_dir, expected_features = demo_paths()
    frame = load_household_data(str(data_path))
    models = load_f13_models(str(models_dir), tuple(expected_features))
    return config, frame, models


def test_valid_days_are_complete_and_inside_chronological_test(household_demo):
    """Seçilebilir günler test bölmesinde tam ve imputasyonsuz olmalıdır."""
    config, frame, _ = household_demo
    days = valid_test_days(frame, config)
    test = frame.loc[config["split"]["test"]["start"]:config["split"]["test"]["end"]]
    assert days
    for day in days:
        target_times = pd.date_range(day, periods=24, freq="h")
        history_times = pd.date_range(
            pd.Timestamp(day) - pd.Timedelta(hours=48), periods=48, freq="h"
        )
        assert target_times.isin(test.index).all()
        assert history_times.isin(frame.index).all()
        assert not frame.loc[history_times.append(target_times), "is_imputed"].astype(bool).any()


def test_f13_inference_is_finite_and_aligned_on_a_real_test_window(household_demo):
    """Gerçek test günü üzerinde F13 ve SN168 24 saatlik hizalı tahmin üretmelidir."""
    config, frame, models = household_demo
    day = valid_test_days(frame, config)[0]
    result = predict_test_day(
        frame, day, models, config["evaluation"]["mape_min_y"]
    )

    assert result["history_kw"].shape == (48,)
    assert result["actual_kw"].shape == (24,)
    assert result["f13_kw"].shape == (24,)
    assert result["sn168_kw"].shape == (24,)
    assert len(result["target_times"]) == 24
    assert result["target_times"].equals(pd.date_range(day, periods=24, freq="h"))
    assert result["history_times"][-1] + pd.Timedelta(hours=1) == result["target_times"][0]
    assert result["target_times"][0] - pd.Timedelta(hours=168) in frame.index
    for values in (
        result["history_kw"],
        result["actual_kw"],
        result["f13_kw"],
        result["sn168_kw"],
    ):
        assert np.isfinite(values).all()
    assert np.isfinite(result["metrics"]["MAE"])
    assert np.isfinite(result["metrics"]["MAPE*"])


def test_checkpoint_feature_mismatch_fails_before_inference(tmp_path):
    """Beklenmeyen checkpoint özelliği yükleme aşamasında açık hata üretmelidir."""
    checkpoint = {
        "model_name": "seq2seq",
        "features": ["unexpected_feature"],
        "forecast": {},
        "feature_mean": [0.0],
        "feature_scale": [1.0],
        "target_mean": [0.0],
        "target_scale": [1.0],
        "state_dict": {},
    }
    torch.save(checkpoint, tmp_path / "F10-opt_best.pt")
    with pytest.raises(ValueError, match="özellikleri config ile uyuşmuyor"):
        load_f13_models(str(tmp_path), ("Global_active_power",))
