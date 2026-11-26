"""LEAD L-FE: öznitelik gruplarının ileri ablasyonu ve L1-FE val değerlendirmesi (docs/lead_protokol.md §17).

Seçim YALNIZ train binalarında yapılır; val'e bakılmaz:
- Her bölme seed'i ayrı (bir seed'in train binaları başka seed'in val binaları olabilir).
- Binaya göre gruplu 3 katlı CV; LightGBM o seed/varyant için L1'in sabit ayarlarıyla (Optuna yok).
- Eğitimde negatif alt örnekleme (bütün pozitifler + negatiflerin %20'si); kat değerlendirmesi tam veriyle.
- G7 (L4-semi artığı) train binalarında katlar dışı (OOF), aynı katlarla; val binalarında bütün train
  binalarıyla eğitilmiş L4-semi'nin skoru.
Çıktılar: results/lead/tables/fe_ablation_folds.csv, fe_ablation.csv/.md, fe_importance.csv, fe_shap_slices.csv.
"""

import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import torch
from sklearn.metrics import average_precision_score

from src.datasets.lead.evaluate import evaluate, f1_max_threshold
from src.datasets.lead.features import CALENDAR, fe_columns, fe_group
from src.pipelines.lead_models import (a3_score_grid, device, fit_lgb, log_run, negative_subsample, record,
                                       run_a3, run_deep, run_ensemble, run_rules, sequence_array, train_a3)

GROUP_ORDER = ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]
CV_SLICES = ("genel", "gap_edge", "frozen", "other")


# ----------------------------------------------------------------------------- öznitelik matrisi
class FeatureStore:
    """Grup başına öznitelikler, yalnız gözlenen satırlar için [N, F] (float32, önbellekli).

    G7 seed'e bağlıdır (OOF); stacked(seed) ile ayrıca verilir.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.cache = {}
        uses = sorted(ctx.sites["primary_use"].unique())
        self.primary_use = dict(zip(ctx.sites["building_id"],
                                    ctx.sites["primary_use"].map({u: i for i, u in enumerate(uses)})))

    def group(self, group, variant, stacked=None):
        """[N, grup sütunları] DataFrame (gözlenen satırlar)."""
        key = (group, variant)
        if group == "G7":
            return fe_group(self.ctx.grid, "G7", variant, stacked=stacked).iloc[self.ctx.obs].reset_index(drop=True)
        if key not in self.cache:
            X = fe_group(self.ctx.grid, group, variant, primary_use=self.primary_use)
            self.cache[key] = X.iloc[self.ctx.obs].reset_index(drop=True)
        return self.cache[key]

    def matrix(self, groups, variant, stacked=None):
        """Seçilen grupların birleşik matrisi. Çıktı: (X [N, F] float32, sütun adları, kategorik indeksler)."""
        parts = [self.group(g, variant, stacked) for g in groups]
        X = pd.concat(parts, axis=1)
        assert list(X.columns) == fe_columns(groups, variant)
        categorical = [i for i, c in enumerate(X.columns) if c == "primary_use"] or "auto"
        return X.to_numpy(np.float32), list(X.columns), categorical


def l1_params(ctx, seed, variant):
    """L1'in o seed ve varyant (full/behavior) için Optuna ile bulduğu ayarlar (bina bölmesi)."""
    runs = pd.read_csv(ctx.tables.parent / "experiments.csv")
    row = runs[(runs["model"] == "L1") & (runs["set"] == variant) & (runs["bolme"] == "bina") & (runs["seed"] == seed)]
    return json.loads(row["params"].iloc[0])


def cv_folds(ctx, seed, kind="bina"):
    """Train binalarında binaya göre gruplu 3 kat. Çıktı: (train satır indeksleri, [(fit, tutulan) kat indeksleri],
    kat başına tutulan binalar). GroupKFold karıştırmaz: aynı seed → aynı katlar (G7 OOF ile ortak)."""
    split = ctx.split(kind, seed)
    train_idx = np.flatnonzero(ctx.rows(split["train"]))
    k = ctx.config["budget"]["cv_folds"]
    folds = list(GroupKFold(k).split(train_idx, groups=ctx.building[train_idx]))
    held = [np.unique(ctx.building[train_idx[b]]) for _, b in folds]
    return train_idx, folds, held


# ----------------------------------------------------------------------------- G7: L4-semi OOF
def stacked_l4semi(ctx, seed, kind="bina"):
    """G7 skoru ızgara sırasında [B·H]: train binaları OOF, val binaları tam-train L4-semi, test binaları NaN.

    OOF: her kat için L4-semi, o katın binaları HARİÇ diğer train binalarıyla eğitilir (ölçek de onlardan);
    tutulan bina kullanılmadığı için erken durdurma yok, epoch sayısı tam-train L4-semi'nin en iyi epoch'u.
    Val binalarının skoru kayıtlı L4-semi koşusundan (results/lead/scores/, val satır sırası). Sonuç önbelleğe
    yazılır (results/lead/scores/g7_{kind}_{seed}.npy). kind="site": siteye göre bölmenin kendi train/val'i.
    Not: tam-train L4-semi'nin val erken durdurması val etiketli saatleri kayıptan çıkarır (val etiketi görür).
    """
    cache = ctx.scores_dir / f"g7_{kind}_{seed}.npy"
    if cache.is_file():
        return np.load(cache)
    started = time.time()
    split = ctx.split(kind, seed)
    runs = pd.read_csv(ctx.tables.parent / "experiments.csv")
    epochs = int(runs[(runs["model"] == "L4-semi") & (runs["bolme"] == kind) & (runs["seed"] == seed)]
                 ["best_epoch"].iloc[0])
    B, H = len(ctx.buildings), ctx.n_hours
    grid_scores = np.full((B, H), np.nan, dtype=np.float32)
    _, _, held = cv_folds(ctx, seed, kind)
    for f, held_b in enumerate(held):
        fit_b = np.setdiff1d(split["train"], held_b)
        model, array, _, _ = train_a3(ctx, fit_b, seed + f, semi=True, fixed_epochs=epochs)
        scores, rows = a3_score_grid(ctx, model, array, held_b)
        grid_scores[rows] = scores
        print(f"  G7 OOF {kind} seed {seed} kat {f}: {len(fit_b)} binayla eğitildi, {len(held_b)} bina skorlandı "
              f"({time.time() - started:.0f} s)", flush=True)
    val = ctx.rows(split["val"])
    val_scores = np.load(ctx.scores_dir / f"{kind}_{seed}_L4-semi_-.npy")
    grid_scores.reshape(-1)[ctx.obs[val]] = val_scores        # val satırları: kayıtlı L4-semi skoru
    flat = grid_scores.reshape(-1)
    np.save(cache, flat)
    log_run(ctx, {"model": "G7-OOF", "set": "-", "bolme": kind, "seed": seed,
                  "params": json.dumps({"fixed_epochs": epochs}), "best_epoch": epochs,
                  "train_time_s": round(time.time() - started, 1)})
    return flat


# ----------------------------------------------------------------------------- ablasyon
def cv_evaluate(ctx, X, categorical, train_idx, folds, params, seed, keep_models=False):
    """Sabit ayarlarla 3 katlı CV. Çıktı: (kat × dilim metrik tablosu, [kat modelleri ve tutulan satırlar])."""
    settings, budget = ctx.config["lightgbm"], ctx.config["budget"]
    rows, models = [], []
    for f, (a, b) in enumerate(folds):
        fit_rows = negative_subsample(train_idx[a], ctx.y, budget["negative_fraction"], seed + f)
        hold = train_idx[b]
        model = fit_lgb(X[fit_rows], ctx.y[fit_rows], X[hold], ctx.y[hold], params, seed, settings, categorical)
        scores = model.predict_proba(X[hold])[:, 1]
        table = evaluate(ctx.y[hold], scores, 0.5, ctx.labels.iloc[hold].reset_index(drop=True))
        for r in table[table["dilim"].isin(CV_SLICES)].itertuples():
            rows.append({"kat": f, "dilim": r.dilim, "pr_auc": r.pr_auc, "roc_auc": r.roc_auc,
                         "best_iteration": int(model.best_iteration_ or 0)})
        if keep_models:
            models.append((model, hold))
    return pd.DataFrame(rows), models


def noise_test(new, old):
    """Kat başına eşleştirilmiş fark Δ = new − old. Çıktı: (ort, 2·std/√k, anlamlı mı)."""
    diff = np.asarray(new) - np.asarray(old)
    bound = 2 * diff.std(ddof=1) / np.sqrt(len(diff))
    return float(diff.mean()), float(bound), bool(abs(diff.mean()) > bound)


def decide(step, accepted):
    """§17.3 karar kuralı: genel PR-AUC'de anlamlı artış → kalır; anlamlı düşüş → çıkar; gürültü düzeyindeyse
    'other' PR-AUC'de anlamlı artış varsa kalır, yoksa çıkar (her grubun maliyeti var)."""
    def metric(table, dilim):
        return table[table["dilim"] == dilim].sort_values("kat")["pr_auc"].to_numpy()

    g_mean, g_bound, g_sig = noise_test(metric(step, "genel"), metric(accepted, "genel"))
    o_mean, o_bound, o_sig = noise_test(metric(step, "other"), metric(accepted, "other"))
    if g_sig:
        keep = g_mean > 0
        reason = "genel anlamlı artış" if keep else "genel anlamlı düşüş"
    else:
        keep = o_sig and o_mean > 0
        reason = "genel gürültü, other anlamlı artış" if keep else "genel gürültü, other artışı yok"
    return keep, {"genel_delta": g_mean, "genel_2se": g_bound, "other_delta": o_mean, "other_2se": o_bound,
                  "karar": "kalır" if keep else "çıkar", "gerekce": reason}


def summarize_step(table):
    """Kat tablosu → dilim başına ortalama ± std (PR-AUC, ROC-AUC)."""
    agg = table.groupby("dilim").agg(pr_auc=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
                                     roc_auc=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"))
    out = {}
    for dilim in CV_SLICES:
        for m in ("pr_auc", "pr_auc_std", "roc_auc", "roc_auc_std"):
            out[f"{dilim}_{m}"] = agg.loc[dilim, m]
    out["best_iteration"] = table["best_iteration"].mean()
    return out


def importance(ctx, X, models, columns, seed):
    """Seçilen setin kat modellerinden gain (normalize, kat ortalaması) ve tutulan katlarda SHAP özeti.

    SHAP: LightGBM pred_contrib (TreeSHAP). Örneklem: tutulan katın bütün pozitifleri + 20 000 negatif.
    Dilim önemi: o dilimin anomali saatlerinde ortalama |SHAP|. X: seçilen setin matrisi [N, F].
    Çıktı: (gain/SHAP tablosu, dilim tablosu).
    """
    rng = np.random.default_rng(seed)
    gains, shap_rows = [], []
    for model, hold in models:
        gain = model.booster_.feature_importance("gain")
        gains.append(gain / gain.sum())
        pos = hold[ctx.y[hold] == 1]
        neg = rng.choice(hold[ctx.y[hold] == 0], size=min(20000, int((ctx.y[hold] == 0).sum())), replace=False)
        rows = np.concatenate([pos, neg])
        contrib = model.booster_.predict(X[rows], pred_contrib=True)[:, :-1]          # son sütun = sabit terim
        subtype = ctx.labels["subtype"].to_numpy()[rows]
        shap_rows.append(pd.DataFrame(np.abs(contrib), columns=columns).assign(subtype=subtype))
    shap = pd.concat(shap_rows, ignore_index=True)
    overall = pd.DataFrame({"oznitelik": columns, "gain": np.mean(gains, axis=0),
                            "shap_ort_abs": shap[columns].mean().to_numpy(), "seed": seed})
    slices = []
    for dilim in ("gap_edge", "frozen", "other"):
        mean = shap.loc[shap["subtype"] == dilim, columns].mean().sort_values(ascending=False).head(10)
        slices += [{"seed": seed, "dilim": dilim, "sira": i + 1, "oznitelik": c, "shap_ort_abs": v}
                   for i, (c, v) in enumerate(mean.items())]
    return overall.sort_values("gain", ascending=False), pd.DataFrame(slices)


def run_ablation(ctx, seed, store):
    """Bir seed için ileri ablasyon (full) + behavior varyantı + öznitelik önemi. Çıktı: seçilen gruplar."""
    started = time.time()
    train_idx, folds, _ = cv_folds(ctx, seed)
    stacked = stacked_l4semi(ctx, seed)
    params = l1_params(ctx, seed, "full")
    fold_rows, step_rows = [], []

    def run(groups, variant, step_name, prm, keep_models=False):
        X, columns, categorical = store.matrix(groups, variant, stacked if "G7" in groups else None)
        t0 = time.time()
        table, models = cv_evaluate(ctx, X, categorical, train_idx, folds, prm, seed, keep_models)
        fold_rows.append(table.assign(seed=seed, adim=step_name, varyant=variant))
        print(f"  seed {seed} {variant:8s} {step_name:28s} F={len(columns):3d} genel PR-AUC "
              f"{table[table.dilim == 'genel'].pr_auc.mean():.4f} other {table[table.dilim == 'other'].pr_auc.mean():.4f}"
              f" ({time.time() - t0:.0f} s)", flush=True)
        return table, models, columns, X

    accepted_groups = ["G0"]
    accepted, models, columns, accepted_X = run(accepted_groups, "full", "G0", params, keep_models=True)
    step_rows.append({"seed": seed, "varyant": "full", "adim": "G0", "eklenen": "G0",
                      "gruplar": "G0", "n_oznitelik": len(columns), **summarize_step(accepted),
                      "karar": "temel", "gerekce": ""})
    for group in GROUP_ORDER:
        trial_groups = accepted_groups + [group]
        table, trial_models, trial_columns, trial_X = run(trial_groups, "full", "+".join(trial_groups), params,
                                                 keep_models=True)
        keep, info = decide(table, accepted)
        step_rows.append({"seed": seed, "varyant": "full", "adim": "+".join(trial_groups), "eklenen": group,
                          "gruplar": "+".join(trial_groups), "n_oznitelik": len(trial_columns),
                          **summarize_step(table), **info})
        if keep:
            accepted_groups, accepted, models, columns, accepted_X = (trial_groups, table, trial_models,
                                                                      trial_columns, trial_X)
        del trial_X                                              # bellek: reddedilen matris tutulmaz
    gain, shap_slices = importance(ctx, accepted_X, models, columns, seed)
    del accepted_X

    b_params = l1_params(ctx, seed, "behavior")
    b_table, _, b_columns, _ = run(accepted_groups, "behavior", "behavior:" + "+".join(accepted_groups), b_params)
    step_rows.append({"seed": seed, "varyant": "behavior", "adim": "behavior:" + "+".join(accepted_groups),
                      "eklenen": "-", "gruplar": "+".join(accepted_groups), "n_oznitelik": len(b_columns),
                      **summarize_step(b_table), "karar": "seçilen setin behavior varyantı", "gerekce": ""})
    log_run(ctx, {"model": "L-FE-ablasyon", "set": "full", "bolme": "bina", "seed": seed,
                  "params": json.dumps({"secilen": accepted_groups}), "best_epoch": None,
                  "train_time_s": round(time.time() - started, 1)})
    return accepted_groups, pd.concat(fold_rows), pd.DataFrame(step_rows), gain, shap_slices


def write_table(frame, path, title, note):
    """CSV + başlıklı/notlu Markdown."""
    frame.to_csv(path.with_suffix(".csv"), index=False)
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |")
    path.with_suffix(".md").write_text(f"# {title}\n\n{note}\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def run_fe(ctx, seeds):
    """Bütün seed'ler için ablasyon; tabloları yazar. Çıktı: {seed: seçilen gruplar}."""
    store = FeatureStore(ctx)
    selected, folds, steps, gains, slices = {}, [], [], [], []
    for seed in seeds:
        groups, f, s, g, sl = run_ablation(ctx, seed, store)
        selected[seed] = groups
        folds.append(f); steps.append(s); gains.append(g); slices.append(sl)
        tables = ctx.tables
        pd.concat(folds).to_csv(tables / "fe_ablation_folds.csv", index=False)
        note = ("Yalnız train binaları, binaya göre gruplu 3 katlı CV, L1'in sabit ayarları; val ve test "
                "kullanılmadı. Final sonuç değildir. Karar kuralı: docs/lead_protokol.md §17.3.")
        write_table(pd.concat(steps), tables / "fe_ablation", "LEAD L-FE ileri ablasyon (train CV)", note)
        write_table(pd.concat(gains), tables / "fe_importance", "LEAD L-FE öznitelik önemi (seçilen set, train CV)",
                    "gain: kat modellerinin normalize gain ortalaması; shap_ort_abs: tutulan katlarda ort. |SHAP|.")
        write_table(pd.concat(slices), tables / "fe_shap_slices", "LEAD L-FE dilim başına en önemli 10 öznitelik",
                    "Tutulan katlarda o dilimin anomali saatlerinde ortalama |SHAP| (train CV).")
    return selected


# ----------------------------------------------------------------------------- L1-FE (val, seed başına bir kez)
def selected_groups(ctx, seed):
    """Bina bölmesinde o seed'in ablasyonla seçtiği gruplar (site kontrolü de bunları kullanır)."""
    runs = pd.read_csv(ctx.tables.parent / "experiments.csv")
    row = runs[(runs["model"] == "L-FE-ablasyon") & (runs["bolme"] == "bina") & (runs["seed"] == seed)]
    return json.loads(row["params"].iloc[0])["secilen"]


def run_l1_fe(ctx, seeds, variants=("full", "behavior"), kind="bina", store=None):
    """Ablasyonda seçilen setle L1-FE: L1 ayarları, bütün train binalarında eğitim, val'de erken durdurma ve eşik."""
    store = store or FeatureStore(ctx)
    settings, budget = ctx.config["lightgbm"], ctx.config["budget"]
    for seed in seeds:
        groups = selected_groups(ctx, seed)
        split = ctx.split(kind, seed)
        train_idx = np.flatnonzero(ctx.rows(split["train"]))
        val = ctx.rows(split["val"])
        stacked = stacked_l4semi(ctx, seed, kind) if "G7" in groups else None
        for variant in variants:
            started = time.time()
            X, columns, categorical = store.matrix(groups, variant, stacked)
            params = l1_params(ctx, seed, variant)
            fit_rows = negative_subsample(train_idx, ctx.y, budget["negative_fraction"], seed)
            model = fit_lgb(X[fit_rows], ctx.y[fit_rows], X[val], ctx.y[val], params, seed, settings, categorical)
            scores = model.predict_proba(X[val])[:, 1]
            record(ctx, "L1-FE", variant, kind, seed, f1_max_threshold(ctx.y[val], scores), val, scores,
                   {"gruplar": groups, "best_iteration": int(model.best_iteration_ or 0)})
            log_run(ctx, {"model": "L1-FE", "set": variant, "bolme": kind, "seed": seed,
                          "params": json.dumps({"gruplar": groups, **params}),
                          "best_epoch": int(model.best_iteration_ or 0),
                          "train_time_s": round(time.time() - started, 1)})


# ----------------------------------------------------------------------------- L2-FE, L5-FE, site kontrolü (§18.3)
def composite_metric(ctx):
    """Optuna amacı ve erken durdurma ölçütü: (genel PR-AUC + `other` PR-AUC) / 2.

    `other` PR-AUC: bütün normal saatler + `other` alt türündeki anomali saatleri (değerlendirmedeki dilim tanımı).
    """
    subtype = ctx.labels["subtype"].to_numpy()

    def metric(rows, scores):
        y, st = ctx.y[rows], subtype[rows]
        other = (st == "normal") | (st == "other")
        return 0.5 * (average_precision_score(y, scores) + average_precision_score(y[other], scores[other]))
    return metric


def fe_sequence_array(ctx, store, kind, seed, variant, groups, stacked):
    """L2-FE girdisi [B, H, C] (GPU, float32): L2'nin kanalları + seçilen grupların öznitelikleri (takvim hariç).

    Sızıntı önlemleri: kırpma sınırları (%1–%99) ve ortalama/std YALNIZ train binalarının gözlenen saatlerinden;
    NaN → 0 (= train ortalaması). primary_use (G6) sıra bilgisi taşımasın diye tek-sıcak (one-hot) kanallara açılır.
    """
    split = ctx.split(kind, seed)
    base = sequence_array(ctx, split["train"], variant)               # [B, H, 1(+1)+7]
    B, H = len(ctx.buildings), ctx.n_hours
    columns = [c for c in fe_columns(groups, variant) if c not in CALENDAR and c != "primary_use"]
    parts = [fe_group(ctx.grid, g, variant, primary_use=store.primary_use, stacked=stacked) for g in groups]
    X = pd.concat(parts, axis=1)[columns].to_numpy(np.float32)       # [B·H, F]
    train_rows = (np.isin(ctx.grid["building_id"].to_numpy(), split["train"])
                  & (ctx.grid["observed"].to_numpy() == 1))
    low = np.nanpercentile(X[train_rows], 1, axis=0)
    high = np.nanpercentile(X[train_rows], 99, axis=0)
    X = np.clip(X, low, high)                                        # aşırı z değerleri (IQR≈0) GRU'yu bozmasın
    mean, std = np.nanmean(X[train_rows], axis=0), np.nanstd(X[train_rows], axis=0) + 1e-6
    X = np.nan_to_num((X - mean) / std, nan=0.0).astype(np.float32)
    extra = [X.reshape(B, H, -1)]
    if "G6" in groups:
        codes = ctx.grid["building_id"].map(store.primary_use).to_numpy().reshape(B, H)
        n_use = max(store.primary_use.values()) + 1
        extra.append(np.eye(n_use, dtype=np.float32)[codes])         # [B, H, kullanım türü sayısı]
    extra = torch.tensor(np.concatenate(extra, axis=2), device=device())
    return torch.cat([base, extra], dim=2)


def memory_estimate_gb(ctx, channels, window, batch=512):
    """Tahmini bellek (GB): dizi [B, H, C] + bir eğitim partisinin pencereleri + değerlendirme partisi (8192)."""
    array = len(ctx.buildings) * ctx.n_hours * channels * 4
    windows = (batch + 8192) * window * channels * 4
    return (array + windows) / 1e9


def run_l2_fe(ctx, seeds, variants=("full", "behavior"), kind="bina", n_trials=None, store=None):
    """L2-FE: GRU, seed'in seçtiği öznitelikler zaman adımı başına. bina: Optuna (bileşik amaç); site: bina
    bölmesinin aynı seed/varyant ayarları (Optuna yok)."""
    store = store or FeatureStore(ctx)
    runs = pd.read_csv(ctx.tables.parent / "experiments.csv")
    for seed in seeds:
        groups = selected_groups(ctx, seed)
        stacked = stacked_l4semi(ctx, seed, kind) if "G7" in groups else None
        for variant in variants:
            array = fe_sequence_array(ctx, store, kind, seed, variant, groups, stacked)
            gb = memory_estimate_gb(ctx, array.shape[2], ctx.config["deep"]["window"])
            print(f"  L2-FE {kind}/{seed}/{variant}: kanal {array.shape[2]}, tahmini bellek {gb:.2f} GB", flush=True)
            if gb > 6.0:                                             # §18.3: 6 GB'ı aşarsa pencere 24
                ctx.config["deep"]["window"] = 24
                print("  bellek > 6 GB: pencere 24 saate düşürüldü", flush=True)
            params = None
            if kind != "bina":
                row = runs[(runs["model"] == "L2-FE") & (runs["set"] == variant) & (runs["bolme"] == "bina")
                           & (runs["seed"] == seed)]
                params = json.loads(row["params"].iloc[0])
            run_deep(ctx, "L2", kind, seed, variant, params=params, n_trials=n_trials, array=array,
                     name="L2-FE", metric=composite_metric(ctx), use_loader=True,
                     extra={"gruplar": groups, "kanal": int(array.shape[2]), "bellek_gb": round(gb, 2)})
            del array
            torch.cuda.empty_cache()


def run_l5_fe(ctx, seeds, variants=("full", "behavior"), kind="bina"):
    """L5-FE: w·L1-FE + (1−w)·L2-FE, aynı varyant; w val PR-AUC ile (§8 L5 kuralı)."""
    for seed in seeds:
        for variant in variants:
            run_ensemble(ctx, kind, seed, variant, members=("L1-FE", "L2-FE"), name="L5-FE")


def run_site_fe(ctx, seeds, variants=("full", "behavior")):
    """Siteye göre bölme kontrolü (§18.3): L0 kuralları (A0 dahil), L4, L4-semi (G7 için), L1-FE, L2-FE, L5-FE.
    Öznitelik grupları ve ayarlar bina bölmesinin aynı seed'inden; eşik ve erken durdurma site val'inde."""
    store = FeatureStore(ctx)
    for seed in seeds:
        run_rules(ctx, "site", seed)
        run_a3(ctx, "site", seed)
        run_a3(ctx, "site", seed, semi=True)                          # G7'nin site val skoru ve OOF epoch'u
    run_l1_fe(ctx, seeds, variants, kind="site", store=store)
    run_l2_fe(ctx, seeds, variants, kind="site", store=store)
    run_l5_fe(ctx, seeds, variants, kind="site")


# ----------------------------------------------------------------------------- L1-OFF (§19, yalnız literatür kıyası)
def run_l1_offline(ctx, seeds):
    """L1-OFF: L1-FE (full) seçilen seti + iki yönlü (GELECEĞE BAKAN) öznitelikler; ablasyon yok, val'de bir kez.

    Ana (çevrimiçi) sonuç DEĞİLDİR; seçim yarışına girmez. Yalnız Kaggle/literatür (iki yönlü öznitelik kullanan
    çözümler) ile kıyas için. Ayarlar, eğitim, erken durdurma ve eşik L1-FE ile aynı.
    """
    from src.datasets.lead.features import offline_features
    store = FeatureStore(ctx)
    offline = offline_features(ctx.grid).iloc[ctx.obs].reset_index(drop=True).to_numpy(np.float32)
    settings, budget = ctx.config["lightgbm"], ctx.config["budget"]
    for seed in seeds:
        started = time.time()
        groups = selected_groups(ctx, seed)
        split = ctx.split("bina", seed)
        train_idx = np.flatnonzero(ctx.rows(split["train"]))
        val = ctx.rows(split["val"])
        stacked = stacked_l4semi(ctx, seed) if "G7" in groups else None
        X, columns, categorical = store.matrix(groups, "full", stacked)
        X = np.hstack([X, offline])                                   # [N, F + 5]
        params = l1_params(ctx, seed, "full")
        fit_rows = negative_subsample(train_idx, ctx.y, budget["negative_fraction"], seed)
        model = fit_lgb(X[fit_rows], ctx.y[fit_rows], X[val], ctx.y[val], params, seed, settings, categorical)
        scores = model.predict_proba(X[val])[:, 1]
        record(ctx, "L1-OFF", "full", "bina", seed, f1_max_threshold(ctx.y[val], scores), val, scores,
               {"gruplar": groups + ["OFFLINE"], "best_iteration": int(model.best_iteration_ or 0)})
        log_run(ctx, {"model": "L1-OFF", "set": "full", "bolme": "bina", "seed": seed,
                      "params": json.dumps({"gruplar": groups + ["OFFLINE"], **params}),
                      "best_epoch": int(model.best_iteration_ or 0), "train_time_s": round(time.time() - started, 1)})
