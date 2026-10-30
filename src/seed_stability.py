"""F9: aynı konfigürasyonun farklı seed'lerdeki val sonuçlarını özetler."""

import argparse

import pandas as pd

from src.data.load import PROJECT_ROOT, load_config
from src.utils.console import setup_console
from src.utils.results_io import resolve_path

METRICS = ["MAE", "RMSE", "sMAPE", "R2"]


def seed_exp_id(base_id, seed, base_seed):
    """Seed tekrarlarının adlandırması: temel seed için base_id, diğerleri base_id-s{seed}."""
    return base_id if seed == base_seed else f"{base_id}-s{seed}"


def summarize(results, base_ids, seeds, base_seed):
    """Her temel konfigürasyon için seed başına metrikleri ve ortalama ± std'yi verir.

    Girdi: forecast_val_results tablosu (exp_id indeksli), temel exp_id'ler.
    Çıktı: (seed başına satırlar, özet satırları) DataFrame'leri.
    Neden: Tek seed'lik farkların gürültü olup olmadığını görmek için
    örneklem std'si (ddof=1, n=3) raporlanır.
    """
    per_seed, summary = [], []
    for base_id in base_ids:
        ids = [seed_exp_id(base_id, seed, base_seed) for seed in seeds]
        missing = [exp_id for exp_id in ids if exp_id not in results.index]
        if missing:
            raise KeyError(f"Eksik seed koşuları: {missing}")
        rows = results.loc[ids, ["model"] + METRICS].copy()
        rows.insert(0, "seed", seeds)
        rows.insert(0, "config", base_id)
        per_seed.append(rows.reset_index())
        entry = {"config": base_id, "model": rows["model"].iloc[0], "n_seeds": len(seeds)}
        for metric in METRICS:
            entry[f"{metric}_mean"] = rows[metric].mean()
            entry[f"{metric}_std"] = rows[metric].std(ddof=1)
        summary.append(entry)
    return pd.concat(per_seed, ignore_index=True), pd.DataFrame(summary)


def main():
    """Seed tablolarını results/tables/seed_stability_val*.csv olarak yazar."""
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True, help="Temel exp_id'ler")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seeds = config["multi_seeds"]
    tables_dir = resolve_path(PROJECT_ROOT, config["paths"]["tables_dir"])
    results = pd.read_csv(tables_dir / "forecast_val_results.csv").set_index("exp_id")
    per_seed, summary = summarize(results, args.configs, seeds, config["seed"])
    per_seed.to_csv(tables_dir / "seed_stability_val_runs.csv", index=False)
    summary.to_csv(tables_dir / "seed_stability_val.csv", index=False)
    print(per_seed.round(4).to_string(index=False))
    print()
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
