"""OOF std kalibrasyonu: kat modelinin eğitim ve early stopping verisi kattan önce biter."""

import pandas as pd

from src.data.load import load_config
from src.train_oof import inner_fold_config
from src.tune_optuna import cv_folds


def test_inner_fold_config_keeps_fold_hours_out_of_training_and_early_stopping():
    config = load_config("configs/optuna_seq2seq_best.yaml")
    split = config["split"]["train"]
    for fold in cv_folds(split["start"], split["end"], 12, 3):
        inner = inner_fold_config(config, fold, inner_val_months=1)
        train_end = pd.Timestamp(inner["split"]["train"]["end"])
        val_start = pd.Timestamp(inner["split"]["val"]["start"])
        val_end = pd.Timestamp(inner["split"]["val"]["end"])
        fold_start = pd.Timestamp(fold["val_start"])
        assert train_end + pd.Timedelta(days=1) == val_start         # bitişik, kesişmesiz
        assert val_end + pd.Timedelta(days=1) == fold_start          # iç val katın hemen öncesi
        assert (val_end - val_start).days in range(27, 31)           # ~1 ay
    assert config["split"]["train"]["end"] == "2009-09-30"           # orijinal config değişmedi
