"""LEAD I-turu (docs/lead_protokol.md §20): davranışsal (`other`) anomali odaklı iyileştirme.

Bütün kararlar train binalarında, binaya göre gruplu 3 katlı CV'nin katlar dışı (OOF) skorlarıyla, her seed ayrı.
Ana ölçüt `other` PR-AUC, ikincil genel PR-AUC; gürültü kuralı |ort(Δ)| ≤ 2·std(Δ)/√3 (kat başına eşleştirilmiş).
Val'e yalnız en sonda bir kez bakılır: son modeller val'de erken durdurma YAPMAZ (ağaç / epoch sayısı = CV katlarındaki
en iyi değerlerin ortalaması); val yalnız son ölçüm ve F1-max eşiği için.

Hatlar:
- LightGBM (L1-I): temel L1-FE → P1 (L1-other + kalibre max) → P4 (akran sapması) → P5 (geriye dönük yumuşatma).
- GRU (L2-I): temel L2-FE → P1 (L2-other + kalibre max) → P5. Optuna yok; L2-FE'nin en iyi ayarları.
Varyant: full (uygulama kararı, günlükte).
"""

import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.core.seed import set_seed
from src.datasets.lead.evaluate import evaluate, f1_max_threshold
from src.datasets.lead.features import P4_COLS, peer_features
from src.pipelines.lead_fe import (CV_SLICES, FeatureStore, composite_metric, cv_folds, fe_sequence_array,
                                   importance, l1_params, noise_test, selected_groups, stacked_l4semi, write_table)
from src.pipelines.lead_models import (device, fit_lgb, lgb_model, log_run, model_factory, negative_subsample,
                                       predict_logits, record, train_classifier, windows)

SMOOTH_K = (1, 3, 6)
GRU_CV_EPOCHS, GRU_CV_PATIENCE = 10, 3          # GRU OOF eğitimi: tutulan katta erken durdurma (CV içinde)


# ----------------------------------------------------------------------------- etiketler ve ortak yardımcılar
def other_targets(ctx):
    """y_other [N]: "other" anomalisi mi; drop [N]: gap_edge/frozen pozitifi mi (L1-other eğitiminden çıkar)."""
    subtype = ctx.labels["subtype"].to_numpy()
    return (subtype == "other").astype(int), np.isin(subtype, ("gap_edge", "frozen"))


def fold_metrics(ctx, train_idx, folds, scores):
    """OOF skor vektörü (train_idx sırası) → kat × dilim PR-AUC/ROC-AUC (gerçek etiketlerle, tam veri)."""
    rows = []
    for f, (_, b) in enumerate(folds):
        hold = train_idx[b]
        table = evaluate(ctx.y[hold], scores[b], 0.5, ctx.labels.iloc[hold].reset_index(drop=True))
        for r in table[table["dilim"].isin(CV_SLICES)].itertuples():
            rows.append({"kat": f, "dilim": r.dilim, "pr_auc": r.pr_auc, "roc_auc": r.roc_auc})
    return pd.DataFrame(rows)


def decide_other(step, base):
    """§20.1: `other` PR-AUC anlamlı artış → kabul; anlamlı düşüş → ret; gürültüyse genel PR-AUC anlamlı artış → kabul."""
    def metric(table, dilim):
        return table[table["dilim"] == dilim].sort_values("kat")["pr_auc"].to_numpy()

    o_mean, o_bound, o_sig = noise_test(metric(step, "other"), metric(base, "other"))
    g_mean, g_bound, g_sig = noise_test(metric(step, "genel"), metric(base, "genel"))
    if o_sig:
        keep, reason = o_mean > 0, "other anlamlı " + ("artış" if o_mean > 0 else "düşüş")
    else:
        keep = g_sig and g_mean > 0
        reason = "other gürültü, genel anlamlı artış" if keep else "other gürültü, genel artışı yok"
    return keep, {"other_delta": o_mean, "other_2se": o_bound, "genel_delta": g_mean, "genel_2se": g_bound,
                  "karar": "kabul" if keep else "ret", "gerekce": reason}


def step_summary(table):
    """Kat tablosu → dilim başına ort ± std (PR-AUC, ROC-AUC)."""
    out = {}
    for dilim in CV_SLICES:
        part = table[table["dilim"] == dilim]
        out.update({f"{dilim}_pr_auc": part["pr_auc"].mean(), f"{dilim}_pr_auc_std": part["pr_auc"].std(),
                    f"{dilim}_roc_auc": part["roc_auc"].mean(), f"{dilim}_roc_auc_std": part["roc_auc"].std()})
    return out


def crossfit_calibrate(scores, target, fold_of):
    """CV içi kalibrasyon: j. katın skoru, diğer katların OOF skorlarıyla fit edilen isotonic ile [0, 1]'e."""
    out = np.empty(len(scores))
    for f in np.unique(fold_of):
        m = fold_of == f
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(scores[~m], target[~m])
        out[m] = iso.predict(scores[m])
    return out


def fit_calibrator(scores, target):
    """Val için: bütün train OOF skorlarıyla isotonic (val etiketi kullanılmaz)."""
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(scores, target)


def smooth(ctx, rows, scores, k):
    """P5: bina içinde geriye dönük k saatlik ortalama (ızgara saatleri; eksik saat ve skorsuz satır atlanır).

    rows: skorların gözlenen satır indeksleri [n]; çıktı aynı sırada [n]. k = 1 → değişiklik yok. Nedensel.
    """
    if k == 1:
        return scores
    B, H = len(ctx.buildings), ctx.n_hours
    grid = np.full(B * H, np.nan)
    grid[ctx.obs[rows]] = scores
    wide = pd.DataFrame(grid.reshape(B, H).T).rolling(k, min_periods=1).mean()   # t-k+1 … t
    return wide.to_numpy().T.reshape(-1)[ctx.obs[rows]]


# ----------------------------------------------------------------------------- LightGBM hattı
def lgb_oof(ctx, X, categorical, train_idx, folds, params, seed, target, drop=None):
    """OOF olasılıkları (train_idx sırası) + kat başına en iyi ağaç sayısı + kat modelleri (tutulan satırlarla).

    drop verilirse (L1-other) o satırlar eğitimden ve erken durdurma kümesinden çıkarılır; OOF skoru yine bütün
    tutulan satırlar için üretilir (değerlendirme tam veriyle).
    """
    settings, budget = ctx.config["lightgbm"], ctx.config["budget"]
    oof = np.full(len(train_idx), np.nan)
    iters, models = [], []
    for f, (a, b) in enumerate(folds):
        pool, hold = train_idx[a], train_idx[b]
        stop = hold
        if drop is not None:
            pool, stop = pool[~drop[pool]], hold[~drop[hold]]
        fit_rows = negative_subsample(pool, target, budget["negative_fraction"], seed + f)
        model = fit_lgb(X[fit_rows], target[fit_rows], X[stop], target[stop], params, seed, settings, categorical)
        oof[b] = model.predict_proba(X[hold])[:, 1]
        iters.append(int(model.best_iteration_ or settings["n_estimators"]))
        models.append((model, hold))
    return oof, iters, models


def lgb_final(ctx, X, categorical, train_idx, params, seed, target, n_estimators, drop=None):
    """Bütün train binalarında, sabit ağaç sayısıyla (val'de erken durdurma YOK)."""
    pool = train_idx if drop is None else train_idx[~drop[train_idx]]
    fit_rows = negative_subsample(pool, target, ctx.config["budget"]["negative_fraction"], seed)
    y = target[fit_rows]
    model = lgb_model(params, (y == 0).sum() / max((y == 1).sum(), 1), seed, n_estimators)
    model.fit(X[fit_rows], y, categorical_feature=categorical)
    return model


def p4_matrix(ctx, store, split):
    """P4 öznitelikleri (gözlenen satırlar) [N, 4]: akranlar aynı bölme parçasından (train/val/test)."""
    part_of = {b: p for p in ("train", "val", "test") for b in split[p]}
    use_of = dict(zip(ctx.sites["building_id"], ctx.sites["primary_use"]))
    X = peer_features(ctx.grid, part_of, use_of)
    return X.iloc[ctx.obs].reset_index(drop=True).to_numpy(np.float32)


def run_lgb_line(ctx, seed, store, step_rows, fold_rows, shap_rows):
    """L1-I: açgözlü adımlar CV'de; sonunda val'de bir kez. Çıktı: kabul edilen yapılandırma."""
    split = ctx.split("bina", seed)
    groups = selected_groups(ctx, seed)
    stacked = stacked_l4semi(ctx, seed) if "G7" in groups else None
    X, columns, categorical = store.matrix(groups, "full", stacked)
    params = l1_params(ctx, seed, "full")
    train_idx, folds, _ = cv_folds(ctx, seed)
    fold_of = np.zeros(len(train_idx), dtype=int)
    for f, (_, b) in enumerate(folds):
        fold_of[b] = f
    y_other, drop = other_targets(ctx)
    y_tr, o_tr = ctx.y[train_idx], y_other[train_idx]

    def log_step(name, table, info):
        fold_rows.append(table.assign(seed=seed, hat="L1", adim=name))
        step_rows.append({"seed": seed, "hat": "L1", "adim": name, **step_summary(table), **info})
        print(f"  L1 seed {seed} {name:28s} other PR-AUC {table[table.dilim == 'other'].pr_auc.mean():.4f} "
              f"genel {table[table.dilim == 'genel'].pr_auc.mean():.4f} → {info.get('karar', 'temel')}", flush=True)

    def score_config(Xc, use_p1):
        """OOF skorlar: L1-FE (+ L1-other ve kalibre max). Çıktı: (skor, (fe oof, iters), (other oof, iters, modeller))."""
        fe, fe_it, fe_models = lgb_oof(ctx, Xc, categorical, train_idx, folds, params, seed, ctx.y)
        if not use_p1:
            return fe, (fe, fe_it, fe_models), None
        ot, ot_it, ot_models = lgb_oof(ctx, Xc, categorical, train_idx, folds, params, seed, y_other, drop)
        combined = np.maximum(crossfit_calibrate(fe, y_tr, fold_of), crossfit_calibrate(ot, o_tr, fold_of))
        return combined, (fe, fe_it, fe_models), (ot, ot_it, ot_models)

    # temel: L1-FE (sabit L1 ayarları, aynı katlar)
    base_scores, base_fe, _ = score_config(X, False)
    current = fold_metrics(ctx, train_idx, folds, base_scores)
    log_step("temel L1-FE", current, {})
    config = {"p1": False, "p4": False, "k": 1}
    scores, fe_part, ot_part, Xc, cols = base_scores, base_fe, None, X, columns

    # P1
    s1, fe1, ot1 = score_config(X, True)
    t1 = fold_metrics(ctx, train_idx, folds, s1)
    keep, info = decide_other(t1, current)
    log_step("+P1 (L1-other, max)", t1, info)
    if keep:
        config["p1"], current, scores, fe_part, ot_part = True, t1, s1, fe1, ot1

    # P4
    P4 = p4_matrix(ctx, store, split)
    X4 = np.hstack([X, P4])
    s4, fe4, ot4 = score_config(X4, config["p1"])
    t4 = fold_metrics(ctx, train_idx, folds, s4)
    keep, info = decide_other(t4, current)
    log_step("+P4 (akran sapması)", t4, info)
    if keep:
        config["p4"], current, scores, fe_part, ot_part, Xc, cols = True, t4, s4, fe4, ot4, X4, columns + P4_COLS
    del X4

    # P5: k ∈ {3, 6} açgözlü, k = 1'e (ya da kabul edilen k'ye) karşı
    for k in SMOOTH_K[1:]:
        sk = smooth(ctx, train_idx, scores, k)
        tk = fold_metrics(ctx, train_idx, folds, sk)
        keep, info = decide_other(tk, current)
        log_step(f"+P5 (k={k})", tk, info)
        if keep:
            config["k"], current = k, tk

    # SHAP: davranışsal dilim, en önemli 10 (L1-other kabul edildiyse onun, değilse L1-FE'nin kat modelleri)
    models = (ot_part or fe_part)[2]
    _, slices = importance(ctx, Xc, models, cols, seed)
    shap_rows.append(slices[slices["dilim"] == "other"].assign(hat="L1", model="L1-other" if ot_part else "L1-FE"))

    # val: bir kez
    val = ctx.rows(split["val"])
    fe_model = lgb_final(ctx, Xc, categorical, train_idx, params, seed, ctx.y, int(np.mean(fe_part[1])))
    val_scores = fe_model.predict_proba(Xc[val])[:, 1]
    if config["p1"]:
        ot_model = lgb_final(ctx, Xc, categorical, train_idx, params, seed, y_other, int(np.mean(ot_part[1])), drop)
        cal_fe, cal_ot = fit_calibrator(fe_part[0], y_tr), fit_calibrator(ot_part[0], o_tr)
        val_scores = np.maximum(cal_fe.predict(val_scores), cal_ot.predict(ot_model.predict_proba(Xc[val])[:, 1]))
    val_idx = np.flatnonzero(val)
    val_scores = smooth(ctx, val_idx, val_scores, config["k"])
    record(ctx, "L1-I", "full", "bina", seed, f1_max_threshold(ctx.y[val], val_scores), val, val_scores,
           {"config": config, "gruplar": groups})
    return config


# ----------------------------------------------------------------------------- GRU hattı
def gru_fixed(ctx, make_model, array, fit_rows, params, epochs, seed, target):
    """Sabit epoch'lu GRU eğitimi (val'de erken durdurma YOK); ağırlıklı BCE, DataLoader (num_workers=0)."""
    set_seed(seed)
    dev = device()
    model = make_model().to(dev)
    L = ctx.config["deep"]["window"]
    b_fit = torch.tensor(ctx.b_pos[fit_rows], device=dev)
    t_fit = torch.tensor(ctx.t_pos[fit_rows], device=dev)
    y_fit = torch.tensor(target[fit_rows], dtype=torch.float32, device=dev)
    pos_weight = torch.tensor((y_fit == 0).sum().item() / max((y_fit == 1).sum().item(), 1), device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=1e-5)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(epochs):
        model.train()
        loader = DataLoader(TensorDataset(torch.arange(len(y_fit))), batch_size=params["batch_size"],
                            shuffle=True, generator=generator, num_workers=0)
        for (idx,) in loader:
            idx = idx.to(dev)
            optimizer.zero_grad()
            loss = loss_fn(model(windows(array, b_fit[idx], t_fit[idx], L)), y_fit[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), ctx.config["deep"]["grad_clip"])
            optimizer.step()
    return model


def gru_oof(ctx, array, params, train_idx, folds, seed, target, metric, drop=None):
    """GRU OOF olasılıkları + kat başına en iyi epoch (tutulan katta erken durdurma, CV içinde)."""
    dev = device()
    oof = np.full(len(train_idx), np.nan)
    epochs = []
    make = model_factory("L2", array.shape[2], params)
    for f, (a, b) in enumerate(folds):
        pool, hold = train_idx[a], train_idx[b]
        stop = hold
        if drop is not None:
            pool, stop = pool[~drop[pool]], hold[~drop[hold]]
        fit_rows = negative_subsample(pool, target, ctx.config["budget"]["negative_fraction"], seed + f)
        _, state, epoch = train_classifier(make, array, fit_rows, stop, ctx, params, GRU_CV_EPOCHS, GRU_CV_PATIENCE,
                                           seed, metric=metric, use_loader=True, target=target)
        model = make().to(dev)
        model.load_state_dict(state)
        logits = predict_logits(model, array, torch.tensor(ctx.b_pos[hold], device=dev),
                                torch.tensor(ctx.t_pos[hold], device=dev), ctx.config["deep"]["window"])
        oof[b] = 1 / (1 + np.exp(-logits))
        epochs.append(epoch)
    return oof, epochs


def run_gru_line(ctx, seed, store, step_rows, fold_rows):
    """L2-I: temel L2-FE → P1 (L2-other, kalibre max) → P5; val'de bir kez (sabit epoch)."""
    split = ctx.split("bina", seed)
    groups = selected_groups(ctx, seed)
    stacked = stacked_l4semi(ctx, seed) if "G7" in groups else None
    array = fe_sequence_array(ctx, store, "bina", seed, "full", groups, stacked)
    runs = pd.read_csv(ctx.tables.parent / "experiments.csv")
    params = json.loads(runs[(runs["model"] == "L2-FE") & (runs["set"] == "full") & (runs["bolme"] == "bina")
                             & (runs["seed"] == seed)]["params"].iloc[0])
    train_idx, folds, _ = cv_folds(ctx, seed)
    fold_of = np.zeros(len(train_idx), dtype=int)
    for f, (_, b) in enumerate(folds):
        fold_of[b] = f
    y_other, drop = other_targets(ctx)
    y_tr, o_tr = ctx.y[train_idx], y_other[train_idx]

    def other_metric(rows, scores):
        return average_precision_score(y_other[rows], scores)

    def log_step(name, table, info):
        fold_rows.append(table.assign(seed=seed, hat="L2", adim=name))
        step_rows.append({"seed": seed, "hat": "L2", "adim": name, **step_summary(table), **info})
        print(f"  L2 seed {seed} {name:28s} other PR-AUC {table[table.dilim == 'other'].pr_auc.mean():.4f} "
              f"genel {table[table.dilim == 'genel'].pr_auc.mean():.4f} → {info.get('karar', 'temel')}", flush=True)

    fe, fe_ep = gru_oof(ctx, array, params, train_idx, folds, seed, ctx.y, composite_metric(ctx))
    current = fold_metrics(ctx, train_idx, folds, fe)
    log_step("temel L2-FE", current, {})
    config, scores = {"p1": False, "k": 1}, fe
    ot, ot_ep = gru_oof(ctx, array, params, train_idx, folds, seed, y_other, other_metric, drop)
    combined = np.maximum(crossfit_calibrate(fe, y_tr, fold_of), crossfit_calibrate(ot, o_tr, fold_of))
    t1 = fold_metrics(ctx, train_idx, folds, combined)
    keep, info = decide_other(t1, current)
    log_step("+P1 (L2-other, max)", t1, info)
    if keep:
        config["p1"], current, scores = True, t1, combined
    for k in SMOOTH_K[1:]:
        tk = fold_metrics(ctx, train_idx, folds, smooth(ctx, train_idx, scores, k))
        keep, info = decide_other(tk, current)
        log_step(f"+P5 (k={k})", tk, info)
        if keep:
            config["k"], current = k, tk

    # val: bir kez; epoch = CV en iyi epoch ortalaması (en az 1)
    dev = device()
    val = ctx.rows(split["val"])
    val_idx = np.flatnonzero(val)
    budget = ctx.config["budget"]
    make = model_factory("L2", array.shape[2], params)

    def val_prob(model):
        logits = predict_logits(model, array, torch.tensor(ctx.b_pos[val_idx], device=dev),
                                torch.tensor(ctx.t_pos[val_idx], device=dev), ctx.config["deep"]["window"])
        return 1 / (1 + np.exp(-logits))

    fit_rows = negative_subsample(train_idx, ctx.y, budget["negative_fraction"], seed)
    val_scores = val_prob(gru_fixed(ctx, make, array, fit_rows, params, max(1, round(np.mean(fe_ep))), seed, ctx.y))
    if config["p1"]:
        pool = train_idx[~drop[train_idx]]
        fit_o = negative_subsample(pool, y_other, budget["negative_fraction"], seed)
        ot_val = val_prob(gru_fixed(ctx, make, array, fit_o, params, max(1, round(np.mean(ot_ep))), seed, y_other))
        val_scores = np.maximum(fit_calibrator(fe, y_tr).predict(val_scores), fit_calibrator(ot, o_tr).predict(ot_val))
    val_scores = smooth(ctx, val_idx, val_scores, config["k"])
    record(ctx, "L2-I", "full", "bina", seed, f1_max_threshold(ctx.y[val], val_scores), val, val_scores,
           {"config": config, "gruplar": groups, "epoch_fe": fe_ep, "epoch_other": ot_ep})
    del array
    torch.cuda.empty_cache()
    return config


# ----------------------------------------------------------------------------- giriş
def run_improve(ctx, seeds, lines=("L1", "L2")):
    """I-turu: her seed için hatlar; tablolar results/lead/tables/i_*.csv/.md."""
    store = FeatureStore(ctx)
    step_rows, fold_rows, shap_rows, configs = [], [], [], {}
    note = ("Yalnız train binaları, binaya göre gruplu 3 katlı CV (OOF skorlar); val ve test kullanılmadı. "
            "Karar kuralı: docs/lead_protokol.md §20.1. Final sonuç değildir.")
    for seed in seeds:
        started = time.time()
        for line in lines:
            if line == "L1":
                configs[(seed, "L1")] = run_lgb_line(ctx, seed, store, step_rows, fold_rows, shap_rows)
            else:
                configs[(seed, "L2")] = run_gru_line(ctx, seed, store, step_rows, fold_rows)
        log_run(ctx, {"model": "I-turu", "set": "full", "bolme": "bina", "seed": seed,
                      "params": json.dumps({k[1]: v for k, v in configs.items() if k[0] == seed}),
                      "best_epoch": None, "train_time_s": round(time.time() - started, 1)})
        tables = ctx.tables
        pd.concat(fold_rows).to_csv(tables / "i_cv_folds.csv", index=False)
        write_table(pd.DataFrame(step_rows), tables / "i_cv_steps", "LEAD I-turu adımları (train CV)", note)
        if shap_rows:
            write_table(pd.concat(shap_rows), tables / "i_shap_other",
                        "LEAD I-turu: davranışsal (other) dilimde en önemli 10 öznitelik (SHAP, train CV)",
                        "Tutulan katlarda other anomali saatlerinde ortalama |SHAP|.")
    return configs
