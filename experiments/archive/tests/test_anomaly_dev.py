"""Geliştirme protokolü testleri (docs/gelistirme_protokolu.md): senaryolar ve yaz-dev sızıntı kontrolleri."""

import numpy as np
import pandas as pd

from src.anomaly_dev import dev_scenario_settings, dev_splits, set_config
from src.anomaly_protocol import scenario_settings
from src.data.load import load_config


def make_hourly(start="2006-12-16", end="2010-11-26 21:00"):
    """Gerçek tarih aralığında sentetik saatlik tablo (yalnız gerekli sütunlar)."""
    index = pd.date_range(start, end, freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame({"Global_active_power": 0.5 + rng.random(len(index)),
                         "is_imputed": np.zeros(len(index), dtype="int8")}, index=index)


def test_medium_scenario_equals_final_main_scenario():
    """Orta = final protokolün ana senaryosu (aynı ayarlar) → eski sonuçlarla birebir kıyas."""
    config = load_config()
    train = make_hourly(end="2009-09-30 23:00")
    assert dev_scenario_settings(config, "medium", train) == scenario_settings(config, "main")
    assert (config["development"]["scenarios"]["medium"]["seeds"]["winter_dev"]
            == config["protocol"]["scenarios"]["main"]["seeds"]["val_dev"])


def test_easy_spike_cap_comes_from_the_given_train_only():
    config = load_config()
    train = make_hourly(end="2009-05-31 23:00")
    train.iloc[10, 0] = 9.0                     # doldurulmamış en büyük değer
    train.iloc[20, 0] = 50.0
    train.iloc[20, 1] = 1                       # doldurulmuş saat sınıra girmez
    settings = dev_scenario_settings(config, "easy", train)
    assert settings["spike"]["max_kw"] == 9.0
    assert "cap_at_train_max" not in settings["spike"]


def test_summer_dev_split_ends_train_before_summer_and_does_not_touch_val_or_test():
    config = set_config(load_config(), "summer_dev")
    splits = dev_splits(make_hourly(), config, "summer_dev")
    assert splits["train"].index.max() == pd.Timestamp("2009-05-31 23:00")
    assert splits["summer_dev"].index.min() == pd.Timestamp("2009-06-01 00:00")
    assert splits["summer_dev"].index.max() == pd.Timestamp("2009-09-30 23:00")
    assert set(splits) == {"train", "summer_dev"}


def test_summer_config_uses_only_models_that_never_saw_fold_3():
    """Yaz-dev'de Optuna ayarlı (kat 3'ü görmüş) modeller ve tüm-train AE'leri kullanılmaz."""
    config = set_config(load_config(), "summer_dev")
    residual = config["anomaly"]["residual"]
    names = {residual["seq2seq"], *residual["ensemble"]}
    assert all(name.startswith("S-") for name in names)
    assert residual["oof_folds"] == 2
    assert config["anomaly"]["ae_experiments"] == ["A2"]
    assert config["anomaly"]["ae_checkpoints"] == {"A2": "S-A2"}
    # Kış-dev config'i değişmez.
    assert set_config(load_config(), "winter_dev")["anomaly"]["residual"]["seq2seq"] == "F10-opt"
