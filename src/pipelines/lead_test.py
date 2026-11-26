"""LEAD final test (docs/lead_protokol.md §13, §21.5–§21.6): TEK SEFER, yalnız kullanıcının açık onayıyla.

Testte HİÇBİR ŞEY SEÇİLMEZ. Ayarlar, öznitelik grupları, eşikler (val F1-max), R3'ün k'sı ve L5-FE ağırlıkları
val kayıtlarından (`val_results.csv`, `experiments.csv`) aynen okunur.

Val aşamasında model nesneleri kaydedilmediği için her model, val'deki prosedürle (aynı seed, ayar, veri ve
erken durdurma) yeniden eğitilir. Önce VAL skorları yeniden üretilip kayıtlı val skorlarıyla karşılaştırılır
(`test_val_reproduction.csv`); sonra aynı model test binalarını bir kez skorlar. Test etiketi yalnız ölçümde.
Kapsam: bina bölmesi, 3 seed; site bölmesinde test yok.
Çıktılar: results/lead/tables/test_results.csv, test_runs/*.csv, test_val_reproduction.csv, test_building_f1.csv.
"""

import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

from src.datasets.lead.eda import backward_run_length, previous_missing
from src.datasets.lead.evaluate import evaluate, fill_scores
from src.datasets.lead.features import offline_features
from src.models.anomaly_baselines import rolling_zscore
from src.pipelines.lead_fe import (FeatureStore, composite_metric, fe_sequence_array, l1_params, selected_groups)
from src.pipelines.lead_models import (KEYS, a3_score_grid, device, fit_lgb, model_factory, negative_subsample,
                                       predict_logits, train_a3, train_classifier)

TEST_MODELS = [("L1", "behavior"), ("L1-FE", "full"), ("L1-FE", "behavior"), ("L0-R3", "-"), ("L0-A0", "-"),
               ("L4", "-"), ("L4-semi", "-"), ("L2-FE", "full"), ("L2-FE", "behavior"), ("L5-FE", "full"),
               ("L5-FE", "behavior"), ("IF", "behavior"), ("L1-OFF", "full")]
FINAL_MODEL = ("L1", "behavior")                   # §21.2


class TestRun:
    """Bir seed'in test koşusu: val kayıtları, maskeler ve sonuç yazımı."""

    def __init__(self, ctx, seed):
        self.ctx, self.seed = ctx, seed
        self.split = ctx.split("bina", seed)
        self.val, self.test = ctx.rows(self.split["val"]), ctx.rows(self.split["test"])
        self.train_idx = np.flatnonzero(ctx.rows(self.split["train"]))
        self.val_results = pd.read_csv(ctx.tables / "val_results.csv")
        self.runs = pd.read_csv(ctx.tables.parent / "experiments.csv")
        self.repro, self.test_scores = [], {}

    def val_row(self, model, fset):
        """O modelin bu seed'deki val kaydı (genel dilim): eşik ve ek bilgi."""
        r = self.val_results[(self.val_results["model"] == model) & (self.val_results["set"] == fset)
                             & (self.val_results["bolme"] == "bina") & (self.val_results["seed"] == self.seed)
                             & (self.val_results["dilim"] == "genel")]
        return r.iloc[0]

    def check_val(self, model, fset, val_scores):
        """Yeniden üretilen val skorları ↔ kayıtlı val skorları: en büyük |fark| ve PR-AUC farkı."""
        saved = np.load(self.ctx.scores_dir / f"bina_{self.seed}_{model}_{fset}.npy")
        y = self.ctx.y[self.val]
        a, b = fill_scores(saved), fill_scores(np.asarray(val_scores, dtype=np.float32))
        self.repro.append({"model": model, "set": fset, "seed": self.seed,
                           "max_abs_fark": float(np.nanmax(np.abs(a - b))),
                           "pr_auc_kayitli": average_precision_score(y, a), "pr_auc_yeniden": average_precision_score(y, b)})

    def record(self, model, fset, test_scores):
        """Test metrikleri (val eşiğiyle) → test_runs/{model}_{set}_bina_{seed}.csv; bina bazında F1."""
        ctx, row = self.ctx, self.val_row(model, fset)
        threshold = float(row["threshold"])
        labels = ctx.labels[self.test].reset_index(drop=True)
        y = ctx.y[self.test]
        table = evaluate(y, test_scores, threshold, labels)
        out = [{"model": model, "set": fset, "bolme": "bina", "seed": self.seed, "threshold": threshold, **r}
               for r in table.to_dict("records")]
        run_dir = ctx.tables / "test_runs"
        run_dir.mkdir(exist_ok=True)
        pd.DataFrame(out).to_csv(run_dir / f"{model}_{fset}_bina_{self.seed}.csv", index=False)
        np.save(ctx.scores_dir / f"test_bina_{self.seed}_{model}_{fset}.npy", np.asarray(test_scores, np.float32))
        self.test_scores[(model, fset)] = np.asarray(test_scores, dtype=np.float64)
        pred = fill_scores(test_scores) >= threshold
        buildings = ctx.building[self.test]
        f1s = []
        for b in np.unique(buildings):
            m = buildings == b
            tp = int((pred[m] & (y[m] == 1)).sum()); fp = int((pred[m] & (y[m] == 0)).sum())
            fn = int((~pred[m] & (y[m] == 1)).sum())
            if tp + fn:                                            # anomalisi olan binalar
                f1s.append({"model": model, "set": fset, "seed": self.seed, "building_id": int(b),
                            "f1": 2 * tp / (2 * tp + fp + fn)})
        genel = table.set_index("dilim").loc["genel"]
        other = table.set_index("dilim").loc["other"]
        print(f"  TEST {model:8s} {fset:9s} seed {self.seed}: PR-AUC {genel.pr_auc:.4f} ROC {genel.roc_auc:.4f} "
              f"F1 {genel.f1:.4f} | other PR-AUC {other.pr_auc:.4f}", flush=True)
        return pd.DataFrame(f1s)


# ----------------------------------------------------------------------------- modeller
def run_rules(run):
    """L0-R3 (k val kaydından) ve L0-A0 (aynı tanım): eğitim yok."""
    ctx = run.ctx
    prev = previous_missing(ctx.clean).astype(float)
    length = backward_run_length(ctx.clean).astype(float)
    k = json.loads(run.val_row("L0-R3", "-")["extra"])["k"]
    r3 = ((prev > 0) | (length >= k)).astype(float)
    run.check_val("L0-R3", "-", r3[run.val])
    rules = ctx.config["rules"]
    z = np.log1p(ctx.grid["meter_reading"]).groupby(ctx.grid["building_id"], sort=False).transform(
        lambda s: rolling_zscore(s, rules["a0_window"], rules["a0_min_std"])).to_numpy()[ctx.obs]
    run.check_val("L0-A0", "-", z[run.val])
    return [run.record("L0-R3", "-", r3[run.test]), run.record("L0-A0", "-", z[run.test])]


def lgb_val_test(run, X, params, categorical="auto"):
    """L1 prosedürü: alt örneklenmiş train, val'de erken durdurma (val aşamasıyla aynı). Çıktı: (val, test) skor."""
    ctx = run.ctx
    fit_rows = negative_subsample(run.train_idx, ctx.y, ctx.config["budget"]["negative_fraction"], run.seed)
    model = fit_lgb(X[fit_rows], ctx.y[fit_rows], X[run.val], ctx.y[run.val], params, run.seed,
                    ctx.config["lightgbm"], categorical)
    return model.predict_proba(X[run.val])[:, 1], model.predict_proba(X[run.test])[:, 1]


def l4_models(run):
    """L4 ve L4-semi: val'deki eğitim (val MAE ile erken durdurma) yeniden; val ve test binalarını skorlar.
    L4-semi'nin test skoru ızgara sırasında döner (G7'nin test değeri)."""
    ctx, out = run.ctx, []
    B, H = len(ctx.buildings), ctx.n_hours
    g7_test = np.full(B * H, np.nan, dtype=np.float32)
    for name, semi in (("L4", False), ("L4-semi", True)):
        model, array, _, _ = train_a3(ctx, run.split["train"], run.seed, semi, stop_buildings=run.split["val"])
        for part, mask in (("val", run.val), ("test", run.test)):
            grid, rows = a3_score_grid(ctx, model, array, run.split[part])
            scores = grid[np.searchsorted(rows, ctx.b_pos[mask]), ctx.t_pos[mask]]
            if part == "val":
                run.check_val(name, "-", scores)
            else:
                out.append(run.record(name, "-", scores))
                if semi:
                    g7_test[ctx.obs[mask]] = scores
        del model, array
        torch.cuda.empty_cache()
    return out, g7_test


def full_stacked(ctx, seed, g7_test):
    """G7 ızgarası: train OOF + val (val aşamasındaki kayıtlı dosya) + test (yeniden eğitilen L4-semi)."""
    stacked = np.load(ctx.scores_dir / f"g7_bina_{seed}.npy").copy()
    fill = np.isnan(stacked) & ~np.isnan(g7_test)
    stacked[fill] = g7_test[fill]
    return stacked


def gru_val_test(run, array, params):
    """L2-FE son eğitimi (val'de bileşik ölçütle erken durdurma, val aşamasıyla aynı). Çıktı: (val, test) olasılık."""
    ctx, dev = run.ctx, device()
    fit_rows = negative_subsample(run.train_idx, ctx.y, ctx.config["budget"]["negative_fraction"], run.seed)
    val_idx, test_idx = np.flatnonzero(run.val), np.flatnonzero(run.test)
    make = model_factory("L2", array.shape[2], params)
    deep = ctx.config["deep"]
    _, state, _ = train_classifier(make, array, fit_rows, val_idx, ctx, params, deep["max_epochs"], deep["patience"],
                                   run.seed, metric=composite_metric(ctx), use_loader=True)
    model = make().to(dev)
    model.load_state_dict(state)

    def prob(idx):
        logits = predict_logits(model, array, torch.tensor(ctx.b_pos[idx], device=dev),
                                torch.tensor(ctx.t_pos[idx], device=dev), deep["window"])
        return 1 / (1 + np.exp(-logits))
    return prob(val_idx), prob(test_idx)


def run_seed(ctx, seed, store):
    """Bir seed'in bütün test modelleri. Çıktı: (bina F1 tablosu, yeniden üretim satırları)."""
    run = TestRun(ctx, seed)
    f1_tables = run_rules(run)

    # L1 behavior (final) — L1'in val aşamasındaki ayarları
    params = json.loads(run.runs[(run.runs["model"] == "L1") & (run.runs["set"] == "behavior")
                                 & (run.runs["bolme"] == "bina") & (run.runs["seed"] == seed)]["params"].iloc[0])
    v, t = lgb_val_test(run, ctx.tabular("behavior").to_numpy(), params)
    run.check_val("L1", "behavior", v)
    f1_tables.append(run.record("L1", "behavior", t))

    # L4, L4-semi (ve G7'nin test değeri)
    tables, g7_test = l4_models(run)
    f1_tables += tables
    groups = selected_groups(ctx, seed)
    stacked = full_stacked(ctx, seed, g7_test) if "G7" in groups else None

    # L1-FE (full, behavior) ve L1-OFF
    fe_scores = {}
    for variant in ("full", "behavior"):
        X, _, categorical = store.matrix(groups, variant, stacked)
        v, t = lgb_val_test(run, X, l1_params(ctx, seed, variant), categorical)
        run.check_val("L1-FE", variant, v)
        f1_tables.append(run.record("L1-FE", variant, t))
        fe_scores[("L1-FE", variant)] = (v, t)
        if variant == "full":
            offline = offline_features(ctx.grid).iloc[ctx.obs].reset_index(drop=True).to_numpy(np.float32)
            v, t = lgb_val_test(run, np.hstack([X, offline]), l1_params(ctx, seed, "full"), categorical)
            run.check_val("L1-OFF", "full", v)
            f1_tables.append(run.record("L1-OFF", "full", t))
        del X

    # L2-FE (full, behavior) — L2-FE'nin val aşamasındaki ayarları
    for variant in ("full", "behavior"):
        array = fe_sequence_array(ctx, store, "bina", seed, variant, groups, stacked)
        params = json.loads(run.runs[(run.runs["model"] == "L2-FE") & (run.runs["set"] == variant)
                                     & (run.runs["bolme"] == "bina") & (run.runs["seed"] == seed)]["params"].iloc[0])
        v, t = gru_val_test(run, array, params)
        run.check_val("L2-FE", variant, v)
        f1_tables.append(run.record("L2-FE", variant, t))
        fe_scores[("L2-FE", variant)] = (v, t)
        del array
        torch.cuda.empty_cache()

    # L5-FE: ağırlık val kaydından
    for variant in ("full", "behavior"):
        w = json.loads(run.val_row("L5-FE", variant)["extra"])["w_L1"]
        (v1, t1), (v2, t2) = fe_scores[("L1-FE", variant)], fe_scores[("L2-FE", variant)]
        run.check_val("L5-FE", variant, w * v1 + (1 - w) * v2)
        f1_tables.append(run.record("L5-FE", variant, w * t1 + (1 - w) * t2))

    # Isolation Forest (aynı tanım: behavior öznitelikleri, train medyanı ile doldurma)
    cfg = ctx.config["isolation_forest"]
    X = ctx.tabular("behavior")
    train = ctx.rows(run.split["train"])
    X = X.fillna(X[train].median()).to_numpy()
    rng = np.random.default_rng(seed)
    fit_rows = rng.choice(np.flatnonzero(train), size=min(cfg["train_rows"], train.sum()), replace=False)
    forest = IsolationForest(n_estimators=cfg["n_estimators"], max_samples=cfg["max_samples"], random_state=seed,
                             n_jobs=-1).fit(X[fit_rows])
    run.check_val("IF", "behavior", -forest.score_samples(X[run.val]))
    f1_tables.append(run.record("IF", "behavior", -forest.score_samples(X[run.test])))
    return pd.concat(f1_tables, ignore_index=True), run.repro


def run_test(ctx, seeds):
    """Final test: seed başına bir kez. Tabloları yazar ve test_results.csv'yi toplar."""
    started = time.time()
    store = FeatureStore(ctx)
    f1_all, repro_all = [], []
    for seed in seeds:
        f1, repro = run_seed(ctx, seed, store)
        f1_all.append(f1); repro_all += repro
        pd.DataFrame(repro_all).to_csv(ctx.tables / "test_val_reproduction.csv", index=False)
        pd.concat(f1_all).to_csv(ctx.tables / "test_building_f1.csv", index=False)
        files = sorted((ctx.tables / "test_runs").glob("*.csv"))
        pd.concat([pd.read_csv(f) for f in files]).drop_duplicates(KEYS, keep="last").sort_values(KEYS).to_csv(
            ctx.tables / "test_results.csv", index=False)
        print(f"  seed {seed} bitti ({time.time() - started:.0f} s)", flush=True)
