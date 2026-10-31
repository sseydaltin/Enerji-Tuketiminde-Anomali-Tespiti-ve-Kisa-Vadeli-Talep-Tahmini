"""Aşama 5: kayıtlı val tahminlerinin ağırlıklı ortalaması (topluluk)."""

import argparse
from datetime import date

import numpy as np
import pandas as pd

from src.data.load import PROJECT_ROOT, load_config
from src.utils.console import setup_console
from src.utils.metrics import calculate_metrics
from src.utils.results_io import (
    load_predictions,
    record_forecast_results,
    resolve_path,
    upsert_csv,
)


def weight_grid(n_models, step):
    """Toplamı 1 olan, `step` adımlı tüm negatif olmayan ağırlık kombinasyonları.

    Örn. 3 model, step=0.1 → 66 kombinasyon. Çıktı: [kombinasyon, n_models] dizi.
    Neden: Basit ve şeffaf ızgara; ağırlıklar optimizasyonla değil sayımla bulunur.
    """
    units = int(round(1 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError("1/step tamsayı olmalı.")
    combos = []

    def fill(prefix, remaining, slots):
        if slots == 1:
            combos.append(prefix + [remaining])
            return
        for value in range(remaining + 1):
            fill(prefix + [value], remaining - value, slots - 1)

    fill([], units, n_models)
    return np.array(combos, dtype=np.float64) / units


def combine(predictions, weights):
    """Tahminleri [model, N, ufuk] ağırlıklı toplar → [N, ufuk] (kW)."""
    return np.tensordot(np.asarray(weights), np.asarray(predictions), axes=1)


def search_weights(y_true, predictions, step, mape_min_y):
    """Izgaradaki her ağırlık kombinasyonu için val MAE'yi hesaplar.

    Çıktı: (en iyi ağırlıklar, tüm kombinasyonların tablosu).
    Not: Ağırlıklar val'de seçildiği için topluluğun val skoru iyimserdir;
    yansız ölçüm yalnız test'te (final) yapılabilir.
    """
    rows = []
    for weights in weight_grid(len(predictions), step):
        scores = calculate_metrics(y_true, combine(predictions, weights), mape_min_y)
        rows.append({"weights": weights.round(3).tolist(), "MAE": scores["MAE"],
                     "RMSE": scores["RMSE"]})
    table = pd.DataFrame(rows).sort_values("MAE", ignore_index=True)
    return np.array(table.loc[0, "weights"]), table


def main():
    """Üyelerin kayıtlı val tahminlerinden topluluk oluşturur ve kaydeder."""
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", nargs="+", required=True, help="Üye exp_id'leri")
    parser.add_argument("--exp", required=True, help="Topluluğun exp_id'si")
    parser.add_argument("--weights", default=None,
                        help="Sabit ağırlıklar (ör. 0.5,0.3,0.2); verilmezse val ızgarası")
    parser.add_argument("--seed", type=int, default=None, help="Kayıt için üyelerin seed'i")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    mape_min_y = config["evaluation"]["mape_min_y"]
    loaded = [load_predictions(PROJECT_ROOT, config, member) for member in args.members]
    y_true, _, times = loaded[0]
    for member, (member_true, _, member_times) in zip(args.members, loaded):
        # Üyeler aynı val pencerelerinde olmalı; aksi halde ortalama anlamsız.
        if not member_times.equals(times) or not np.allclose(member_true, y_true):
            raise RuntimeError(f"{member} farklı val pencereleri kullanıyor.")
    predictions = [member_pred for _, member_pred, _ in loaded]

    tables_dir = resolve_path(PROJECT_ROOT, config["paths"]["tables_dir"])
    if args.weights is None:
        weights, grid = search_weights(
            y_true, predictions, config["ensemble"]["weight_step"], mape_min_y
        )
        grid.insert(0, "members", [args.members] * len(grid))
        grid.to_csv(tables_dir / f"{args.exp}_weight_grid_val.csv", index=False)
        source = "val ızgarası"
    else:
        weights = np.array([float(value) for value in args.weights.split(",")])
        if len(weights) != len(args.members) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("Ağırlık sayısı üye sayısına eşit ve toplamı 1 olmalı.")
        source = "sabit (seed 42 val ızgarasından)"

    note = f"members={args.members}; weights={weights.round(3).tolist()} ({source})"
    scores = record_forecast_results(
        PROJECT_ROOT, config, "val", args.exp, "Ensemble",
        y_true, combine(predictions, weights), times,
        train_time_s=0.0, device="cpu", feature_set="-",
        lookback="-", note=note,
    )
    upsert_csv(
        resolve_path(PROJECT_ROOT, config["paths"]["experiments_file"]),
        [{
            "exp_id": args.exp, "date": date.today().isoformat(), "model": "Ensemble",
            "seed": args.seed if args.seed is not None else config["seed"],
            "val_mae": scores["MAE"], "val_rmse": scores["RMSE"],
            "test_mae": "", "test_rmse": "", "train_time_s": 0.0, "note": note,
        }],
        ["exp_id"],
    )
    print(f"{args.exp}: ağırlıklar {weights.round(3).tolist()} → val MAE {scores['MAE']:.4f} kW")
    for member, (_, member_pred, _) in zip(args.members, loaded):
        print(f"  {member}: val MAE {calculate_metrics(y_true, member_pred, mape_min_y)['MAE']:.4f}")


if __name__ == "__main__":
    main()
