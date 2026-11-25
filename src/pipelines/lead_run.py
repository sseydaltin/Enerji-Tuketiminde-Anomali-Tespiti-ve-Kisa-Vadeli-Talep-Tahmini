"""LEAD1.0 hattı. L1: temizlik (--stage clean) ve keşifsel veri analizi (--stage eda).

Komutlar (proje kökünden):
    python -m src.pipelines.lead_run --stage clean   # data/processed/lead_clean.parquet + cleaning_log.csv
    python -m src.pipelines.lead_run --stage eda     # results/lead/tables/*.csv, results/lead/figures/*.png

EDA bütün etiketli veride (train.csv, 200 bina) yapılır: bu aşamada bölme, model, eşik veya
seçim YOKTUR; yalnız betimleme. Household ve ELD sonuçlarına dokunulmaz (ayrı config ve klasörler).
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.core.config import PROJECT_ROOT, load_config
from src.core.console import setup_console
from src.core.results_io import resolve_path
from src.datasets.lead.eda import (anomaly_events, backward_run_length, constant_run_length, coverage, gap_neighbors,
                                   length_bins, previous_missing, reading_flags, same_start_events)
from src.datasets.lead.load import clean, extract_sites, load_raw

CONFIG_FILE = PROJECT_ROOT / "configs/lead.yaml"
SCOPE = "LEAD train.csv, tüm etiketli veri (200 bina, 2016)"   # figür başlıklarında küme bilgisi
UNIT = "meter_reading (kWh)"


def paths(config):
    """Config'teki yolları mutlak yola çevirir; çıktı klasörlerini oluşturur."""
    out = {key: resolve_path(PROJECT_ROOT, value) for key, value in config["paths"].items()}
    for key in ("tables_dir", "figures_dir"):
        out[key].mkdir(parents=True, exist_ok=True)
    out["processed_file"].parent.mkdir(parents=True, exist_ok=True)
    return out


def stage_clean(config):
    """Ham train.csv → temiz tablo (parquet) + temizlik günlüğü (CSV)."""
    p = paths(config)
    cleaned, log = clean(load_raw(p["raw_file"]))
    cleaned.to_parquet(p["processed_file"], index=False)
    log.to_csv(p["tables_dir"] / "cleaning_log.csv", index=False)
    print(log.to_string(index=False))
    return cleaned


def save(figure, path):
    """Figürü dpi 200 kaydeder ve kapatır."""
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def summary_tables(raw, df, cov, events, flags, tables_dir):
    """Veri özeti, bina özeti, olay uzunlukları, sıfır/sabit–anomali ilişkisi ve ham eksiklerin etiketleri."""
    rate = df.groupby("building_id")["anomaly"].mean()
    building = cov.merge(pd.DataFrame({
        "anomaly_rate": rate,
        "n_anomaly_hours": df.groupby("building_id")["anomaly"].sum(),
        "n_events": events.groupby("building_id").size(),
        "median_reading_kwh": df.groupby("building_id")["meter_reading"].median(),
        "zero_share": flags["is_zero"].groupby(df["building_id"]).mean(),
        "frozen_share": flags["is_frozen"].groupby(df["building_id"]).mean(),
    }).reset_index(), on="building_id")
    building["n_events"] = building["n_events"].fillna(0).astype(int)
    building.to_csv(tables_dir / "building_summary.csv", index=False)

    raw_missing = raw["meter_reading"].isna()
    missing_labels = pd.DataFrame([{
        "raw_rows": len(raw), "missing_reading_rows": int(raw_missing.sum()),
        "missing_reading_anomaly_1": int((raw_missing & (raw["anomaly"] == 1)).sum()),
        "raw_anomaly_1": int((raw["anomaly"] == 1).sum()),
    }])
    missing_labels.to_csv(tables_dir / "raw_missing_reading_labels.csv", index=False)

    lengths = events["length_hours"]
    dist = pd.DataFrame({"bin": length_bins(lengths), "length": lengths}).groupby("bin", observed=False).agg(
        n_events=("length", "size"), anomaly_hours=("length", "sum")).reset_index()
    dist["event_share"] = dist["n_events"] / dist["n_events"].sum()
    dist["anomaly_hour_share"] = dist["anomaly_hours"] / dist["anomaly_hours"].sum()
    dist.to_csv(tables_dir / "event_length_distribution.csv", index=False)

    rows = []
    for label, name in ((1, "anomali"), (0, "normal")):
        part = flags[df["anomaly"].to_numpy() == label]
        rows.append({"etiket": name, "n_saat": len(part),
                     "sifir_orani": part["is_zero"].mean(), "sabit_orani": part["is_frozen"].mean(),
                     "sifir_veya_sabit_orani": (part["is_zero"] | part["is_frozen"]).mean()})
    for flag, name in (("is_zero", "sıfır okumalar"), ("is_frozen", "sabit okumalar")):
        mask = flags[flag].to_numpy()
        rows.append({"etiket": f"{name} içinde anomali oranı", "n_saat": int(mask.sum()),
                     "anomali_orani": float(df["anomaly"].to_numpy()[mask].mean()) if mask.any() else np.nan})
    zero_frozen = pd.DataFrame(rows)
    zero_frozen.to_csv(tables_dir / "anomaly_zero_frozen.csv", index=False)

    medians = building["median_reading_kwh"]
    summary = pd.DataFrame([
        ("ham satır", len(raw)), ("temiz satır", len(df)), ("bina", df["building_id"].nunique()),
        ("ilk saat", df["timestamp"].min()), ("son saat", df["timestamp"].max()),
        ("ızgara saati (bina başına)", int(cov["expected_hours"].iloc[0])),
        ("bina başına gözlem: min / medyan / maks",
         f"{cov['n_obs'].min()} / {int(cov['n_obs'].median())} / {cov['n_obs'].max()}"),
        ("toplam eksik bina-saat", int(cov["missing_hours"].sum())),
        ("anomali oranı (temiz, saat)", float(df["anomaly"].mean())),
        ("anomalisi olan bina", int((building["n_anomaly_hours"] > 0).sum())),
        ("bina başına anomali oranı: medyan / maks",
         f"{building['anomaly_rate'].median():.4f} / {building['anomaly_rate'].max():.4f}"),
        ("anomali olayı", len(events)),
        ("tek saatlik olay oranı", float((events["length_hours"] == 1).mean())),
        ("olay uzunluğu medyan / maks (saat)",
         f"{int(events['length_hours'].median())} / {int(events['length_hours'].max())}"),
        ("bina medyan tüketimi: min / medyan / maks (kWh)",
         f"{medians.min():.2f} / {medians.median():.2f} / {medians.max():.2f}"),
    ], columns=["ölçü", "değer"])
    summary.to_csv(tables_dir / "dataset_summary.csv", index=False)
    return building, dist, zero_frozen, summary, missing_labels


def frozen_sensitivity(df, run_length, tables_dir, min_runs=(2, 3, 6, 12, 24)):
    """"Sabit okuma" tanımının eşiğe duyarlılığı: anomali ve normal saatlerde sabit oranı.

    Neden: Ana tanım (≥3 saat) bir seçimdir; bulgunun eşiğe bağlı olup olmadığı gösterilir.
    """
    is_anomaly = df["anomaly"].to_numpy() == 1
    rows = [{"min_run_saat": k,
             "anomali_saatlerinde_sabit_orani": float((run_length[is_anomaly] >= k).mean()),
             "normal_saatlerde_sabit_orani": float((run_length[~is_anomaly] >= k).mean()),
             "sabit_saatlerde_anomali_orani": float(is_anomaly[run_length >= k].mean())}
            for k in min_runs]
    table = pd.DataFrame(rows)
    table.to_csv(tables_dir / "frozen_sensitivity.csv", index=False)
    print(table.round(4).to_string(index=False), "\n")


def gap_table(df, events, tables_dir):
    """Veri boşluğu kenarı ile etiket ilişkisi: anomali/normal saatlerin ve olayların ne kadarı eksik saate komşu?

    Olay, ilk saatinden önceki veya son saatinden sonraki saat eksikse "boşluğa komşu" sayılır
    (olayın içi boşluksuz olduğu için bu, ilk ve son satırın gap_neighbors bayrağına eşittir).
    Neden: Etiket incelemesinde tek saatlik etiketlerin sık sık boşluk kenarında durduğu görüldü.
    """
    near = gap_neighbors(df)
    is_anomaly = df["anomaly"].to_numpy() == 1
    row = pd.Series(np.arange(len(df)), index=pd.MultiIndex.from_frame(df[["building_id", "timestamp"]]))
    first = row.reindex(pd.MultiIndex.from_frame(events[["building_id", "start"]])).to_numpy()
    last = row.reindex(pd.MultiIndex.from_frame(events[["building_id", "end"]])).to_numpy()
    event_near = near[first] | near[last]
    single = events["length_hours"].to_numpy() == 1
    rows = [("anomali saatleri", is_anomaly.sum(), near[is_anomaly].mean()),
            ("normal saatler", (~is_anomaly).sum(), near[~is_anomaly].mean()),
            ("anomali olayları (tümü)", len(events), event_near.mean()),
            ("tek saatlik olaylar", single.sum(), event_near[single].mean()),
            ("çok saatlik olaylar", (~single).sum(), event_near[~single].mean()),
            ("eksik saate komşu saatlerde anomali oranı", near.sum(), is_anomaly[near].mean())]
    table = pd.DataFrame(rows, columns=["grup", "n", "oran"])
    table.to_csv(tables_dir / "gap_adjacency.csv", index=False)
    print(table.round(4).to_string(index=False))
    print()


def load_sites(path):
    """building_sites.csv → (building_id, site_id, primary_use); yoksa None (--stage sites ile üretilir)."""
    if not path.is_file():
        return None
    return pd.read_csv(path)


def site_table(df, sites, tables_dir):
    """LEAD binalarının kaçının site bilgisiyle eşleştiği ve site başına bina sayısı."""
    lead_buildings = pd.Series(df["building_id"].unique(), name="building_id")
    if sites is None:
        pd.DataFrame([{"durum": "building_sites.csv yok; site eşleşmesi yapılmadı (--stage sites)"}]).to_csv(
            tables_dir / "site_match.csv", index=False)
        print("Site bilgisi yok; eşleşme yapılmadı (--stage sites).")
        return None
    matched = lead_buildings.to_frame().merge(sites, on="building_id", how="left")
    per_site = matched.groupby("site_id", dropna=False).agg(
        n_bina=("building_id", "size"),
        kullanim_turleri=("primary_use", lambda s: ", ".join(sorted(s.dropna().unique())))).reset_index()
    per_site.to_csv(tables_dir / "site_match.csv", index=False)
    print(f"Site eşleşmesi: {matched['site_id'].notna().sum()} / {len(matched)} bina\n{per_site.to_string(index=False)}\n")
    return matched


def causal_table(df, events, tables_dir, ks=(2, 3, 6)):
    """Sabit okuma ve boşluk kenarı: iki yönlü (EDA) ve yalnız geçmişe bakan tanım yan yana.

    İki yönlü: sabit = serinin TAMAMI ≥ k saat; boşluk kenarı = önceki VEYA sonraki saat eksik.
    Nedensel: sabit = son k saat aynı (geriye dönük seri ≥ k); boşluk kenarı = önceki saat eksik.
    """
    is_anomaly = df["anomaly"].to_numpy() == 1
    two_run, back_run = constant_run_length(df), backward_run_length(df)
    flags = [("boşluk kenarı", "iki yönlü (önceki veya sonraki eksik)", "", gap_neighbors(df)),
             ("boşluk kenarı", "nedensel (önceki saat eksik) = R1", "", previous_missing(df))]
    for k in ks:
        flags += [("sabit okuma", "iki yönlü (seri ≥ k)", k, two_run >= k),
                  ("sabit okuma", "nedensel (son k saat aynı) = R2", k, back_run >= k)]
    rows = [{"özellik": name, "tanım": definition, "k": k, "işaretli_saat": int(mask.sum()),
             "anomali_saatlerinde_oran": float(mask[is_anomaly].mean()),
             "normal_saatlerde_oran": float(mask[~is_anomaly].mean()),
             "işaretli_saatlerde_anomali_oranı": float(is_anomaly[mask].mean())}
            for name, definition, k, mask in flags]
    table = pd.DataFrame(rows)
    table.to_csv(tables_dir / "causal_vs_twosided.csv", index=False)
    print(table.round(4).to_string(index=False))
    print()


def same_start_table(df, events, matched_sites, tables_dir):
    """Aynı saatte birden çok binada başlayan olaylar; site bilgisi varsa aynı sitede mi."""
    ev = same_start_events(events)
    shared = ev["n_buildings_same_start"] >= 2
    near = gap_neighbors(df)
    row = pd.Series(np.arange(len(df)), index=pd.MultiIndex.from_frame(df[["building_id", "timestamp"]]))
    first = row.reindex(pd.MultiIndex.from_frame(ev[["building_id", "start"]])).to_numpy()
    last = row.reindex(pd.MultiIndex.from_frame(ev[["building_id", "end"]])).to_numpy()
    gap_event = near[first] | near[last]
    summary = [
        ("olay", len(ev)),
        ("en az bir başka binada aynı saatte başlayan olay", int(shared.sum())),
        ("oranı", float(shared.mean())),
        ("bu olayların paylaşıldığı farklı başlangıç saati", int(ev.loc[shared, "start"].nunique())),
        ("bir saatte olay başlatan en çok bina", int(ev["n_buildings_same_start"].max())),
        ("paylaşılan olaylarda boşluk kenarı oranı", float(gap_event[shared].mean())),
        ("paylaşılmayan olaylarda boşluk kenarı oranı", float(gap_event[~shared].mean())),
        ("paylaşılan olaylarda medyan uzunluk (saat)", float(ev.loc[shared, "length_hours"].median())),
        ("paylaşılmayan olaylarda medyan uzunluk (saat)", float(ev.loc[~shared, "length_hours"].median())),
    ]
    if matched_sites is not None:
        site = ev["building_id"].map(matched_sites.set_index("building_id")["site_id"])
        n_sites = ev.assign(site=site).groupby("start")["site"].transform("nunique")
        summary.append(("paylaşılan olaylarda tüm binaların tek sitede olduğu oran",
                        float((n_sites[shared] == 1).mean())))
        same_site_peers = ev.assign(site=site).groupby(["start", "site"])["building_id"].transform("nunique")
        summary.append(("paylaşılan olaylarda en az bir başka binanın aynı sitede olduğu oran",
                        float((same_site_peers[shared] >= 2).mean())))
    else:
        summary.append(("site kontrolü", "yapılmadı (site bilgisi yok)"))
    pd.DataFrame(summary, columns=["ölçü", "değer"]).to_csv(tables_dir / "same_start_events.csv", index=False)
    top = (ev[shared].groupby("start").agg(n_bina=("building_id", "nunique"),
                                           medyan_uzunluk=("length_hours", "median"))
           .sort_values("n_bina", ascending=False).head(15).reset_index())
    top.to_csv(tables_dir / "same_start_top_hours.csv", index=False)
    print(pd.DataFrame(summary, columns=["ölçü", "değer"]).to_string(index=False))
    print(top.to_string(index=False))
    print()


def summary_figures(building, dist, zero_frozen, figures_dir):
    """Gözlem/eksik saat, anomali oranı, olay uzunluğu, sıfır/sabit ve ölçek dağılımı figürleri."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(building["n_obs"], bins=40, color="#1f4e9c")
    axes[0].set(xlabel="Bina başına gözlem (saat)", ylabel="Bina sayısı", title="Gözlem sayısı")
    axes[1].hist(building["missing_hours"], bins=40, color="#888888")
    axes[1].set(xlabel="Bina başına eksik saat (8.784 saatlik ızgaraya göre)", ylabel="Bina sayısı",
                title="Eksik saat")
    figure.suptitle(SCOPE)
    save(figure, figures_dir / "lead_coverage.png")

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.hist(100 * building["anomaly_rate"], bins=40, color="#c0392b")
    axis.set(xlabel="Bina başına anomali oranı (%, saat)", ylabel="Bina sayısı",
             title=f"Bina başına anomali oranı – {SCOPE}")
    save(figure, figures_dir / "lead_anomaly_rate_by_building.png")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = dist["bin"].astype(str)
    axes[0].bar(labels, dist["n_events"], color="#c0392b")
    axes[0].set(xlabel="Olay uzunluğu (saat)", ylabel="Olay sayısı", title="Olay sayısı")
    axes[1].bar(labels, 100 * dist["anomaly_hour_share"], color="#7f1d1d")
    axes[1].set(xlabel="Olay uzunluğu (saat)", ylabel="Anomali saatlerinin payı (%)",
                title="Anomali saatlerinin olay uzunluğuna dağılımı")
    figure.suptitle(f"Anomali olayları (ardışık etiketli saatler) – {SCOPE}")
    save(figure, figures_dir / "lead_event_lengths.png")

    part = zero_frozen.iloc[:2]
    figure, axis = plt.subplots(figsize=(8, 4))
    x = np.arange(3)
    for i, (_, row) in enumerate(part.iterrows()):
        values = 100 * row[["sifir_orani", "sabit_orani", "sifir_veya_sabit_orani"]].astype(float).to_numpy()
        axis.bar(x + (i - 0.5) * 0.38, values, width=0.38, label=f"{row['etiket']} saatler",
                 color="#c0392b" if row["etiket"] == "anomali" else "#888888")
    axis.set_xticks(x, ["Sıfır okuma", "Sabit okuma (≥3 s)", "Sıfır veya sabit"])
    axis.set(ylabel="Saatlerin oranı (%)", title=f"Sıfır ve sabit okumalar – {SCOPE}")
    axis.legend()
    save(figure, figures_dir / "lead_zero_frozen.png")

    figure, axis = plt.subplots(figsize=(8, 4))
    medians = building["median_reading_kwh"].clip(lower=1e-3)
    axis.hist(np.log10(medians), bins=40, color="#1f4e9c")
    axis.set(xlabel="log10(bina medyan tüketimi, kWh)", ylabel="Bina sayısı",
             title=f"Bina ölçek farkı – {SCOPE}")
    save(figure, figures_dir / "lead_building_scale.png")


def sample_review(df, events, config):
    """Etiket incelemesi örneklemi: rastgele anomali olayları ve tamamı normal etiketli pencereler.

    Çıktı: DataFrame (kind, building_id, start, end). Seed config'ten; olaylar uzunluktan bağımsız,
    eşit olasılıkla seçilir. Normal pencere: aynı binada boşluksuz `normal_window_hours` saat, hiç anomali yok.
    """
    eda = config["eda"]
    rng = np.random.default_rng(config["seed"])
    chosen = events.iloc[rng.choice(len(events), size=eda["review_anomaly_segments"], replace=False)]
    rows = [{"kind": "anomali", "building_id": int(r.building_id), "start": r.start, "end": r.end}
            for r in chosen.sort_values(["building_id", "start"]).itertuples()]

    width = eda["normal_window_hours"]
    indexed = df.set_index(["building_id", "timestamp"])["anomaly"]
    normals = []
    while len(normals) < eda["review_normal_windows"]:
        row = df.iloc[int(rng.integers(len(df)))]
        hours = pd.date_range(row["timestamp"], periods=width, freq="h")
        keys = pd.MultiIndex.from_arrays([np.full(width, row["building_id"]), hours])
        labels = indexed.reindex(keys)
        if labels.notna().all() and (labels == 0).all():
            normals.append({"kind": "normal", "building_id": int(row["building_id"]),
                            "start": hours[0], "end": hours[-1]})
    return pd.DataFrame(rows + sorted(normals, key=lambda r: (r["building_id"], r["start"])))


def review_figures(df, sample, config, figures_dir):
    """Her örnek için ±context saatlik okuma grafiği; etiketli saatler kırmızı, seçilen aralık gölgeli."""
    context = pd.Timedelta(hours=config["eda"]["review_context_hours"])
    by_building = {b: g.set_index("timestamp") for b, g in df.groupby("building_id")}
    for kind, per_figure in (("anomali", 6), ("normal", 5)):
        part = sample[sample["kind"] == kind].reset_index(drop=True)
        for page in range(int(np.ceil(len(part) / per_figure))):
            chunk = part.iloc[page * per_figure:(page + 1) * per_figure]
            figure, axes = plt.subplots(len(chunk), 1, figsize=(12, 2.3 * len(chunk)), squeeze=False)
            for axis, r in zip(axes[:, 0], chunk.itertuples()):
                grid = pd.date_range(r.start - context, r.end + context, freq="h")
                window = by_building[r.building_id].reindex(grid)          # eksik saat → boşluk
                axis.plot(grid, window["meter_reading"], color="#333333", linewidth=1)
                labeled = window["anomaly"] == 1
                axis.scatter(grid[labeled], window.loc[labeled, "meter_reading"], color="#c0392b", s=10,
                             zorder=3, label="anomaly = 1")
                axis.axvspan(r.start, r.end + pd.Timedelta(minutes=59), color="#f5b7b1" if kind == "anomali"
                             else "#d6eaf8", alpha=0.6)
                length = int((r.end - r.start) / pd.Timedelta(hours=1)) + 1
                axis.set_title(f"Bina {r.building_id} · {r.start:%Y-%m-%d %H:%M} – {r.end:%Y-%m-%d %H:%M} "
                               f"({length} s, {kind})", fontsize=9)
                axis.set_ylabel("kWh", fontsize=8)
                axis.tick_params(labelsize=7)
            axes[-1, 0].set_xlabel("Zaman (saat)")
            figure.suptitle(f"Etiket incelemesi – {kind} örnekleri ±{config['eda']['review_context_hours']} s "
                            f"({UNIT}; kırmızı nokta = etiketli saat) – {SCOPE}", fontsize=10)
            save(figure, figures_dir / f"lead_review_{'anomaly' if kind == 'anomali' else 'normal'}_{page + 1}.png")


def stage_eda(config):
    """Temiz tablodan bütün EDA tablolarını ve figürlerini üretir."""
    p = paths(config)
    raw = load_raw(p["raw_file"])
    if not p["processed_file"].is_file():
        raise FileNotFoundError(f"{p['processed_file']} yok; önce --stage clean çalıştırın.")
    df = pd.read_parquet(p["processed_file"])
    cov = coverage(df)
    events = anomaly_events(df)
    flags = reading_flags(df, config["eda"]["frozen_min_run"])
    building, dist, zero_frozen, summary, missing = summary_tables(raw, df, cov, events, flags, p["tables_dir"])
    events.to_csv(p["tables_dir"] / "anomaly_events.csv", index=False)
    frozen_sensitivity(df, flags["run_length"].to_numpy(), p["tables_dir"])
    gap_table(df, events, p["tables_dir"])
    matched = site_table(df, load_sites(p["sites_file"]), p["tables_dir"])
    causal_table(df, events, p["tables_dir"])
    same_start_table(df, events, matched, p["tables_dir"])
    summary_figures(building, dist, zero_frozen, p["figures_dir"])
    sample = sample_review(df, events, config)
    sample.to_csv(p["tables_dir"] / "label_review_sample.csv", index=False)
    review_figures(df, sample, config, p["figures_dir"])
    for table in (summary, missing, dist, zero_frozen):
        print(table.to_string(index=False), "\n")


def stage_sites(config, features_path):
    """train_features.csv (veya zip) → data/raw/lead/building_sites.csv (yalnız site_id, primary_use)."""
    p = paths(config)
    sites = extract_sites(features_path)
    sites.to_csv(p["sites_file"], index=False)
    print(f"{len(sites)} bina → {p['sites_file']}")


def main():
    """--stage clean | eda."""
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("clean", "eda", "sites", "splits", "l0", "l1", "l4", "deep", "l5", "fe", "l1fe", "l2fe", "l5fe", "sitefe", "l1off", "improve",
                                            "if", "site", "summary"), required=True)
    parser.add_argument("--model", choices=("L2", "L3"), default="L2", help="--stage deep")
    parser.add_argument("--sets", default="full,behavior", help="öznitelik setleri (virgülle)")
    parser.add_argument("--seeds", default=None, help="bölme seed'leri (virgülle); varsayılan config")
    parser.add_argument("--trials", type=int, default=None, help="Optuna deneme sayısı (bütçe)")
    parser.add_argument("--features", default=None,
                        help="--stage sites: Kaggle train_features.csv ya da energy-anomaly-detection.zip yolu")
    parser.add_argument("--config", default=str(CONFIG_FILE))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage in ("splits", "l0", "l1", "l4", "deep", "l5", "if", "site", "summary", "fe", "l1fe", "l2fe", "l5fe", "sitefe", "l1off", "improve"):
        from src.pipelines import lead_models as lm      # model kütüphaneleri yalnız gerektiğinde yüklenir
        ctx = lm.LeadData(config)
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else config["split"]["seeds"]
        sets = args.sets.split(",")
        if args.stage == "splits":
            lm.write_splits(ctx)
        if args.stage in ("fe", "l1fe", "l2fe", "l5fe", "sitefe", "l1off", "improve"):  # L-FE (§17–§20)
            from src.pipelines import lead_fe
            if args.stage == "fe":
                print(lead_fe.run_fe(ctx, seeds))
            elif args.stage == "l1fe":
                lead_fe.run_l1_fe(ctx, seeds)
            elif args.stage == "l2fe":
                lead_fe.run_l2_fe(ctx, seeds, sets, n_trials=args.trials)
            elif args.stage == "improve":                # §20: I-turu (train CV, val en sonda bir kez)
                from src.pipelines import lead_improve
                print(lead_improve.run_improve(ctx, seeds))
            elif args.stage == "l1off":                  # §19: yalnız literatür kıyası (offline)
                lead_fe.run_l1_offline(ctx, seeds)
            elif args.stage == "l5fe":
                lead_fe.run_l5_fe(ctx, seeds, sets)
            else:
                lead_fe.run_site_fe(ctx, seeds, sets)
        for seed in seeds if args.stage not in ("splits", "site", "summary", "fe", "l1fe", "l2fe", "l5fe",
                                                "sitefe", "l1off", "improve") else []:
            if args.stage == "l0":
                lm.run_rules(ctx, "bina", seed)
            elif args.stage == "l1":
                for s in sets:
                    lm.run_lightgbm(ctx, "bina", seed, s)
            elif args.stage == "l4":
                lm.run_a3(ctx, "bina", seed)
                lm.run_a3(ctx, "bina", seed, semi=True)
            elif args.stage == "deep":
                for s in sets:
                    lm.run_deep(ctx, args.model, "bina", seed, s, n_trials=args.trials)
            elif args.stage == "l5":
                for s in sets:
                    lm.run_ensemble(ctx, "bina", seed, s)
            elif args.stage == "if":
                lm.run_isolation_forest(ctx, "bina", seed)
        if args.stage == "site":
            lm.run_site_check(ctx, seeds)
        if args.stage == "summary":
            lm.summarize(ctx)
        return
    if args.stage == "clean":
        stage_clean(config)
    elif args.stage == "sites":
        stage_sites(config, args.features)
    else:
        stage_eda(config)


if __name__ == "__main__":
    main()
