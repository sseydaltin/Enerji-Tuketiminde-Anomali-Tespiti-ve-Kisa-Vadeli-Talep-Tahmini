"""Aşama 4–5: CV katları, kayıp seçimi, Optuna parametreleri, topluluk, seed özeti."""

import numpy as np
import optuna
import pandas as pd
import pytest
import torch

from src.data.load import load_config
from src.ensemble import combine, search_weights, weight_grid
from src.seed_stability import seed_exp_id, summarize
from src.train_forecast import make_loss
from src.tune_optuna import apply_params, cv_folds, fold_config, suggest_params


def test_cv_folds_cover_last_12_train_months_in_order():
    folds = cv_folds("2006-12-16", "2009-09-30", cv_months=12, n_folds=3)

    assert [(f["val_start"], f["val_end"]) for f in folds] == [
        ("2008-10-01", "2009-01-31"),
        ("2009-02-01", "2009-05-31"),
        ("2009-06-01", "2009-09-30"),
    ]
    for fold in folds:
        # Genişleyen pencere: katın eğitimi doğrulamasından hemen önce biter.
        assert pd.Timestamp(fold["train_end"]) + pd.Timedelta(days=1) == pd.Timestamp(fold["val_start"])
        assert fold["train_start"] == "2006-12-16"


def test_cv_never_touches_real_validation_or_test():
    config = load_config()
    real_val_start = pd.Timestamp(config["split"]["val"]["start"])
    split = config["split"]["train"]
    for fold in cv_folds(split["start"], split["end"], 12, 3):
        fold_cfg = fold_config(config, fold)
        assert pd.Timestamp(fold_cfg["split"]["val"]["end"]) < real_val_start
        # Test tarihleri değişmez (pencere hazırlığı test'i zaten okumaz).
        assert fold_cfg["split"]["test"] == config["split"]["test"]
    # Orijinal config değişmemeli.
    assert config["split"]["val"]["start"] == str(real_val_start.date())


@pytest.mark.parametrize("name", ["mse", "l1", "huber"])
def test_make_loss_returns_zero_for_perfect_prediction(name):
    values = torch.tensor([[1.0, 2.0]])
    assert make_loss(name)(values, values).item() == pytest.approx(0.0)


def test_make_loss_rejects_unknown():
    with pytest.raises(ValueError):
        make_loss("mape")


def test_suggest_params_stay_inside_search_space():
    space = load_config()["optuna"]["space"]
    trial = optuna.trial.FixedTrial({
        "hidden": 64, "layers": 2, "lr": 0.001, "batch_size": 64, "dropout": 0.2,
        "weight_decay": 1e-5, "loss": "huber", "shuffle_train": True,
    })
    params = suggest_params(trial, space)
    config = apply_params(load_config(), params)

    assert config["forecast"]["loss"] == "huber"
    assert config["forecast"]["shuffle_train"] is True
    assert space["lr"][0] <= params["lr"] <= space["lr"][1]


def test_weight_grid_sums_to_one_and_has_expected_size():
    grid = weight_grid(3, 0.1)
    assert grid.shape == (66, 3)
    np.testing.assert_allclose(grid.sum(axis=1), 1.0)
    assert (grid >= 0).all()


def test_search_weights_picks_the_better_member():
    y_true = np.ones((5, 2))
    good = y_true + 0.1
    bad = y_true + 1.0
    weights, table = search_weights(y_true, [good, bad], step=0.5, mape_min_y=0.1)

    np.testing.assert_allclose(weights, [1.0, 0.0])
    assert table.loc[0, "MAE"] == pytest.approx(0.1)
    np.testing.assert_allclose(combine([good, bad], [0.5, 0.5]), y_true + 0.55)


def test_seed_summary_uses_naming_and_sample_std():
    results = pd.DataFrame(
        {"model": ["M"] * 3, "MAE": [1.0, 2.0, 3.0], "RMSE": [1.0] * 3,
         "sMAPE": [1.0] * 3, "R2": [0.5] * 3},
        index=["X", "X-s7", "X-s2026"],
    )
    assert seed_exp_id("X", 42, 42) == "X"
    assert seed_exp_id("X", 7, 42) == "X-s7"
    _, summary = summarize(results, ["X"], [42, 7, 2026], 42)
    assert summary.loc[0, "MAE_mean"] == pytest.approx(2.0)
    assert summary.loc[0, "MAE_std"] == pytest.approx(1.0)
