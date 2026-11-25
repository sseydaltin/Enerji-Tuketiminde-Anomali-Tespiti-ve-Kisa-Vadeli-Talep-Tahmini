"""LEAD model hatları (docs/lead_protokol.md): L0 kurallar, L1 LightGBM, L2 GRU, L3 CNN+GRU, L4 A3, L5 topluluk,
Isolation Forest. YALNIZ VAL: test binaları bu modülde hiçbir zaman skorlanmaz (test ayrı onayla yapılır).

Ortak kurallar:
- Bölme bina düzeyinde; ölçek istatistikleri ve modeller yalnız TRAIN binalarına fit edilir.
- Optuna yalnız train binalarında, binaya göre gruplu CV ile; val binaları yalnız erken durdurma, eşik ve
  topluluk ağırlığı için (protokol §9–§10).
- Eğitimde negatif alt örnekleme (bütün pozitifler + negatiflerin %20'si, seed'li); değerlendirme tam veri.
- Sonuçlar results/lead/tables/val_results.csv (anahtar: model, set, bölme, seed, dilim); val skorları
  results/lead/scores/ (git dışı) altında topluluk için saklanır.
"""

import json
import time

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.core.config import PROJECT_ROOT
from src.core.results_io import resolve_path
from src.core.seed import set_seed
from src.datasets.lead.eda import backward_run_length, previous_missing
from src.datasets.lead.evaluate import evaluate, f1_max_threshold
from src.datasets.lead.features import (FEATURE_SETS, calendar, evaluation_labels, full_grid, normalized_residual,
                                        tabular_features)
from src.datasets.lead.splits import building_split, site_split, split_table
from src.models.anomaly_baselines import rolling_zscore
from src.models.lead_classifiers import CNNGRUClassifier, GRUClassifier
from src.models.seq2seq import Seq2SeqForecaster

KEYS = ["model", "set", "bolme", "seed", "dilim"]
SIMPLICITY = ["L0", "L4", "L1", "L2", "L3", "L5"]          # protokol §12 (küçük = basit)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ----------------------------------------------------------------------------- ortak veri
class LeadData:
    """Bütün aşamaların ortak verisi: temiz tablo, saat ızgarası, değerlendirme etiketleri, bölmeler."""

    def __init__(self, config):
        self.config = config
        self.paths = {k: resolve_path(PROJECT_ROOT, v) for k, v in config["paths"].items()}
        self.tables = self.paths["tables_dir"]
        self.scores_dir = self.tables.parent / "scores"
        self.scores_dir.mkdir(parents=True, exist_ok=True)
        clean = pd.read_parquet(self.paths["processed_file"])
        self.clean = clean.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
        self.grid = full_grid(self.clean)
        self.obs = np.flatnonzero(self.grid["observed"].to_numpy() == 1)   # temiz satırların ızgara konumu
        assert len(self.obs) == len(self.clean)
        self.labels = evaluation_labels(self.clean, config["evaluation"]["frozen_min_run"])
        self.y = self.clean["anomaly"].to_numpy().astype(int)
        self.building = self.clean["building_id"].to_numpy()
        self.buildings = np.sort(self.clean["building_id"].unique())
        self.n_hours = len(self.grid) // len(self.buildings)
        self.b_pos = np.searchsorted(self.buildings, self.building)        # [N] bina sırası
        self.t_pos = self.obs - self.b_pos * self.n_hours                  # [N] saat sırası
        sites_file = self.paths["sites_file"]
        self.sites = pd.read_csv(sites_file) if sites_file.is_file() else None
        self._tabular = {}

    def tabular(self, feature_set):
        """Gözlenen satırlar için tablo öznitelikleri [N, F] (önbellekli)."""
        if feature_set not in self._tabular:
            X = tabular_features(self.grid, feature_set).iloc[self.obs].reset_index(drop=True)
            self._tabular[feature_set] = X
        return self._tabular[feature_set]

    def split(self, kind, seed):
        """kind: "bina" (ana) veya "site" (sağlamlık)."""
        s = self.config["split"]
        if kind == "bina":
            return building_split(self.buildings, seed, s["n_val"], s["n_test"])
        return site_split(self.sites, seed, tuple(s["site_fractions"]))

    def rows(self, building_ids):
        """Verilen binaların gözlenen satırları (bool maske [N])."""
        return np.isin(self.building, building_ids)


def negative_subsample(index, y, fraction, seed):
    """Bütün pozitifler + negatiflerin `fraction` kadarı (seed'li). Girdi/çıktı: satır indeksleri."""
    rng = np.random.default_rng(seed)
    pos, neg = index[y[index] == 1], index[y[index] == 0]
    keep = rng.choice(neg, size=int(round(fraction * len(neg))), replace=False)
    return np.sort(np.concatenate([pos, keep]))


def record(ctx, model, feature_set, kind, seed, threshold, val_mask, scores, extra=None):
    """Val metriklerini hesaplar, val_results.csv'ye yazar, val skorlarını saklar. Çıktı: metrik tablosu."""
    table = evaluate(ctx.y[val_mask], scores, threshold, ctx.labels[val_mask].reset_index(drop=True))
    rows = [{"model": model, "set": feature_set, "bolme": kind, "seed": seed, "threshold": threshold,
             **row, "extra": json.dumps(extra or {}, ensure_ascii=False)} for row in table.to_dict("records")]
    # Koşu başına ayrı dosya: farklı modeller aynı anda koşabilir (ortak CSV'ye eşzamanlı yazma yarışı yok).
    run_dir = ctx.tables / "val_runs"
    run_dir.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(run_dir / f"{model}_{feature_set}_{kind}_{seed}.csv", index=False)
    collect_results(ctx)
    np.save(ctx.scores_dir / f"{kind}_{seed}_{model}_{feature_set}.npy", np.asarray(scores, dtype=np.float32))
    genel = table.set_index("dilim").loc["genel"]
    other = table.set_index("dilim").loc["other"]
    print(f"  {model:10s} {feature_set:9s} {kind}/{seed}: PR-AUC {genel.pr_auc:.4f} ROC {genel.roc_auc:.4f} "
          f"F1 {genel.f1:.4f} | other PR-AUC {other.pr_auc:.4f} F1 {other.f1:.4f}", flush=True)
    return table


def collect_results(ctx):
    """val_runs/*.csv → val_results.csv (bütün koşular, anahtar başına son kayıt)."""
    files = sorted((ctx.tables / "val_runs").glob("*.csv"))
    if files:
        table = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates(KEYS, keep="last")
        table.sort_values(KEYS).to_csv(ctx.tables / "val_results.csv", index=False)


def log_run(ctx, row):
    """Koşu kaydı (süre, seçilen ayarlar) → results/lead/runs/*.json → results/lead/experiments.csv."""
    run_dir = ctx.tables.parent / "runs"
    run_dir.mkdir(exist_ok=True)
    (run_dir / f"{row['model']}_{row['set']}_{row['bolme']}_{row['seed']}.json").write_text(
        json.dumps(row, ensure_ascii=False), encoding="utf-8")
    rows = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(run_dir.glob("*.json"))]
    pd.DataFrame(rows).to_csv(ctx.tables.parent / "experiments.csv", index=False)


# ----------------------------------------------------------------------------- L0
def run_rules(ctx, kind, seed):
    """L0: (a) hepsi normal, (b) R1, (c) R2 (k val'de), (d) R3 = R1 ∨ R2 (k val'de), (e) A0 z-skoru."""
    split = ctx.split(kind, seed)
    val = ctx.rows(split["val"])
    y = ctx.y[val]
    prev = previous_missing(ctx.clean)[val].astype(float)
    run = backward_run_length(ctx.clean)[val].astype(float)
    ks = ctx.config["rules"]["r2_k_values"]

    record(ctx, "L0-normal", "-", kind, seed, np.inf, val, np.zeros(val.sum()))
    record(ctx, "L0-R1", "-", kind, seed, 0.5, val, prev)

    def f1(pred):
        tp = (pred & (y == 1)).sum()
        return 2 * tp / (pred.sum() + (y == 1).sum())

    k2 = max(ks, key=lambda k: f1(run >= k))                      # R2: k val F1 ile
    record(ctx, "L0-R2", "-", kind, seed, float(k2), val, run, {"k": k2})
    k3 = max(ks, key=lambda k: f1((prev > 0) | (run >= k)))       # R3: k val F1 ile
    r3 = ((prev > 0) | (run >= k3)).astype(float)
    record(ctx, "L0-R3", "-", kind, seed, 0.5, val, r3, {"k": k3})

    rules = ctx.config["rules"]
    log_reading = np.log1p(ctx.grid["meter_reading"])
    z = log_reading.groupby(ctx.grid["building_id"], sort=False).transform(
        lambda s: rolling_zscore(s, rules["a0_window"], rules["a0_min_std"]))
    a0 = z.to_numpy()[ctx.obs][val]
    record(ctx, "L0-A0", "-", kind, seed, f1_max_threshold(y, a0), val, a0)


# ----------------------------------------------------------------------------- L1
def lgb_model(params, scale_pos_weight, seed, n_estimators):
    """LightGBM sınıflandırıcı (erken durdurma metriği: average precision = PR-AUC)."""
    # metric yalnız average_precision: varsayılan binary_logloss da izlenirse erken durdurma, scale_pos_weight'in
    # şişirdiği olasılıklar yüzünden logloss ilk ağaçtan kötüleştiği için 1. ağaçta durur (2026-11-25 hatası).
    return lgb.LGBMClassifier(objective="binary", metric="average_precision", n_estimators=n_estimators,
                              random_state=seed, verbosity=-1, scale_pos_weight=scale_pos_weight,
                              subsample_freq=1, **params)


def fit_lgb(X, y, X_stop, y_stop, params, seed, settings, categorical="auto"):
    """Alt örneklenmiş eğitim kümesiyle eğitir; X_stop'ta PR-AUC ile erken durdurur.

    categorical: kategorik sütunların indeksleri (L-FE G6 primary_use); varsayılan "auto" = yok (L1).
    """
    model = lgb_model(params, (y == 0).sum() / max((y == 1).sum(), 1), seed, settings["n_estimators"])
    model.fit(X, y, eval_set=[(X_stop, y_stop)], categorical_feature=categorical,
              callbacks=[lgb.early_stopping(settings["early_stopping_rounds"], first_metric_only=True, verbose=False)])
    return model


def lgb_space(trial):
    """L1 arama uzayı."""
    return {"learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)}


def run_lightgbm(ctx, kind, seed, feature_set, params=None):
    """L1: Optuna (train binalarında gruplu CV) → train binalarında son eğitim, val'de erken durdurma ve eşik.

    params verilirse (site bölmesi) Optuna atlanır.
    """
    started = time.time()
    split = ctx.split(kind, seed)
    X = ctx.tabular(feature_set).to_numpy()
    train_idx = np.flatnonzero(ctx.rows(split["train"]))
    val = ctx.rows(split["val"])
    settings, budget = ctx.config["lightgbm"], ctx.config["budget"]

    if params is None:
        folds = list(GroupKFold(budget["cv_folds"]).split(train_idx, groups=ctx.building[train_idx]))

        def objective(trial):
            p = lgb_space(trial)
            scores = []
            for f, (a, b) in enumerate(folds):
                fit_rows = negative_subsample(train_idx[a], ctx.y, budget["negative_fraction"], seed + f)
                hold = train_idx[b]
                m = fit_lgb(X[fit_rows], ctx.y[fit_rows], X[hold], ctx.y[hold], p, seed, settings)
                scores.append(average_precision_score(ctx.y[hold], m.predict_proba(X[hold])[:, 1]))
                trial.report(float(np.mean(scores)), f)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
        study.optimize(objective, n_trials=settings["optuna_trials"])
        params = study.best_params

    fit_rows = negative_subsample(train_idx, ctx.y, budget["negative_fraction"], seed)
    model = fit_lgb(X[fit_rows], ctx.y[fit_rows], X[val], ctx.y[val], params, seed, settings)
    scores = model.predict_proba(X[val])[:, 1]
    record(ctx, "L1", feature_set, kind, seed, f1_max_threshold(ctx.y[val], scores), val, scores,
           {"best_iteration": int(model.best_iteration_ or 0)})
    log_run(ctx, {"model": "L1", "set": feature_set, "bolme": kind, "seed": seed,
                  "params": json.dumps(params), "best_epoch": int(model.best_iteration_ or 0),
                  "train_time_s": round(time.time() - started, 1)})
    return params


# ----------------------------------------------------------------------------- derin modeller: ortak tensörler
def device():
    """CUDA varsa GPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sequence_array(ctx, train_buildings, feature_set):
    """[bina, saat, kanal] tensörü (GPU). Kanallar: ölçekli log okuma (ileri doldurulmuş), [maske], takvim (7).

    Ölçek (ortalama/std) yalnız TRAIN binalarının gözlenen log okumalarından. Eksik okuma bina bazında son
    geçerli değerle doldurulur (yalnız geçmiş); binanın ilk gözleminden önceki saatler 0 (train ortalaması).
    full setinde eksiklik maskesi kanalı vardır; behavior setinde yoktur (§7).
    """
    B, H = len(ctx.buildings), ctx.n_hours
    logs = np.log1p(ctx.grid["meter_reading"].to_numpy()).reshape(B, H)
    train = np.isin(ctx.buildings, train_buildings)
    mean, std = np.nanmean(logs[train]), np.nanstd(logs[train])
    x = (logs - mean) / std
    missing = np.isnan(x)
    last = np.where(~missing, np.arange(H)[None, :], 0)
    np.maximum.accumulate(last, axis=1, out=last)                  # son geçerli saatin indeksi (geçmiş)
    x = np.nan_to_num(np.take_along_axis(x, last, axis=1), nan=0.0)
    cal = np.broadcast_to(calendar(ctx.grid["timestamp"].iloc[:H]).to_numpy(np.float32), (B, H, 7))
    parts = [x[..., None]] + ([missing[..., None].astype(np.float32)] if feature_set == "full" else []) + [cal]
    return torch.tensor(np.concatenate(parts, axis=2), dtype=torch.float32, device=device())


def windows(array, b, t, length):
    """[B,H,C] tensöründen (b, t) çiftleri için son `length` saat [n, length, C] (t dahil, yalnız geçmiş)."""
    offsets = torch.arange(-length + 1, 1, device=array.device)
    idx = (t[:, None] + offsets[None, :]).clamp(min=0)             # binanın başından önce: ilk saat tekrar
    return array[b[:, None], idx]


def predict_logits(model, array, b, t, length, batch=8192):
    """Gradyansız, parça parça logit."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(b), batch):
            out.append(model(windows(array, b[i:i + batch], t[i:i + batch], length)).float().cpu())
    return torch.cat(out).numpy()


def train_classifier(make_model, array, fit_rows, stop_rows, ctx, params, max_epochs, patience, seed,
                     trial=None, step=0, metric=None, use_loader=False):
    """Ağırlıklı BCE ile eğitir; stop_rows'ta ölçütle erken durdurur. Çıktı: (en iyi ölçüt, durum, epoch).

    metric(rows, scores): erken durdurma ölçütü; None → genel PR-AUC (L2). use_loader=True (L2-FE): karışık
    satır indeksleri DataLoader'dan (num_workers=0) gelir, pencereler her parti için ayrı üretilir.
    """
    set_seed(seed)
    dev = device()
    model = make_model().to(dev)
    L = ctx.config["deep"]["window"]
    b_fit = torch.tensor(ctx.b_pos[fit_rows], device=dev); t_fit = torch.tensor(ctx.t_pos[fit_rows], device=dev)
    y_fit = torch.tensor(ctx.y[fit_rows], dtype=torch.float32, device=dev)
    b_stop = torch.tensor(ctx.b_pos[stop_rows], device=dev); t_stop = torch.tensor(ctx.t_pos[stop_rows], device=dev)
    pos_weight = torch.tensor((y_fit == 0).sum().item() / max((y_fit == 1).sum().item(), 1), device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=1e-5)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, best_epoch, wait = -1.0, None, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        if use_loader:
            loader = DataLoader(TensorDataset(torch.arange(len(y_fit))), batch_size=params["batch_size"],
                                shuffle=True, generator=generator, num_workers=0)
            batches = (idx.to(dev) for (idx,) in loader)
        else:
            order = torch.randperm(len(y_fit), generator=generator).to(dev)
            batches = (order[i:i + params["batch_size"]] for i in range(0, len(order), params["batch_size"]))
        for batch in batches:
            optimizer.zero_grad()
            loss = loss_fn(model(windows(array, b_fit[batch], t_fit[batch], L)), y_fit[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), ctx.config["deep"]["grad_clip"])
            optimizer.step()
        logits = predict_logits(model, array, b_stop, t_stop, L)
        score = metric(stop_rows, logits) if metric else average_precision_score(ctx.y[stop_rows], logits)
        if trial is not None:
            trial.report(score, step + epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if score > best:
            best, best_epoch, wait = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    return best, best_state, best_epoch


def deep_space(trial, architecture):
    """L2/L3 arama uzayı."""
    p = {"hidden": trial.suggest_categorical("hidden", [32, 64, 128]),
         "layers": trial.suggest_int("layers", 1, 2),
         "dropout": trial.suggest_float("dropout", 0.0, 0.3),
         "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
         "batch_size": trial.suggest_categorical("batch_size", [256, 512])}
    if architecture == "L3":
        p["filters"] = trial.suggest_categorical("filters", [16, 32, 64])
        p["kernel"] = trial.suggest_categorical("kernel", [3, 5])
    return p


def model_factory(architecture, channels, p):
    """L2 → GRUClassifier, L3 → CNNGRUClassifier."""
    if architecture == "L2":
        return lambda: GRUClassifier(channels, p["hidden"], p["layers"], p["dropout"])
    return lambda: CNNGRUClassifier(channels, p["hidden"], p["layers"], p["dropout"], p["filters"], p["kernel"])


def run_deep(ctx, architecture, kind, seed, feature_set, params=None, n_trials=None, array=None, name=None,
             metric=None, use_loader=False, extra=None):
    """L2 / L3: Optuna (train binalarında gruplu CV, MedianPruner, epoch sınırı) → son eğitim → val.

    L2-FE (§18.3) için: array = öznitelik kanallı dizi, name = "L2-FE", metric = bileşik ölçüt (Optuna amacı ve
    erken durdurma), use_loader = DataLoader ile parti; extra = kayda eklenecek bilgi (öznitelik grupları).
    """
    started = time.time()
    split = ctx.split(kind, seed)
    name = name or architecture
    if array is None:
        array = sequence_array(ctx, split["train"], feature_set)
    channels = array.shape[2]
    deep, budget = ctx.config["deep"], ctx.config["budget"]
    train_idx = np.flatnonzero(ctx.rows(split["train"]))
    val_idx = np.flatnonzero(ctx.rows(split["val"]))

    if params is None:
        folds = list(GroupKFold(budget["cv_folds"]).split(train_idx, groups=ctx.building[train_idx]))

        def objective(trial):
            p = deep_space(trial, architecture)
            scores = []
            for f, (a, b) in enumerate(folds):
                fit_rows = negative_subsample(train_idx[a], ctx.y, budget["negative_fraction"], seed + f)
                best, _, _ = train_classifier(model_factory(architecture, channels, p), array, fit_rows,
                                              train_idx[b], ctx, p, deep["optuna_max_epochs"], 2, seed,
                                              trial=trial, step=f * deep["optuna_max_epochs"],
                                              metric=metric, use_loader=use_loader)
                scores.append(best)
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=2))
        study.optimize(objective, n_trials=n_trials or deep["optuna_trials"])
        params = study.best_params

    fit_rows = negative_subsample(train_idx, ctx.y, budget["negative_fraction"], seed)
    _, state, epoch = train_classifier(model_factory(architecture, channels, params), array, fit_rows, val_idx,
                                       ctx, params, deep["max_epochs"], deep["patience"], seed,
                                       metric=metric, use_loader=use_loader)
    model = model_factory(architecture, channels, params)().to(device())
    model.load_state_dict(state)
    dev = device()
    logits = predict_logits(model, array, torch.tensor(ctx.b_pos[val_idx], device=dev),
                            torch.tensor(ctx.t_pos[val_idx], device=dev), deep["window"])
    scores = 1 / (1 + np.exp(-logits))
    val = ctx.rows(split["val"])
    record(ctx, name, feature_set, kind, seed, f1_max_threshold(ctx.y[val], scores), val, scores,
           {"best_epoch": epoch, **(extra or {})})
    log_run(ctx, {"model": name, "set": feature_set, "bolme": kind, "seed": seed,
                  "params": json.dumps(params), "best_epoch": epoch, "train_time_s": round(time.time() - started, 1)})
    return params


# ----------------------------------------------------------------------------- L4
def train_a3(ctx, train_buildings, seed, semi=False, stop_buildings=None, fixed_epochs=None):
    """L4 (A3) Seq2Seq tahmin modelini eğitir. Çıktı: (model, array [B,H,9], en iyi epoch, en iyi kayıp).

    train_buildings: eğitim binaları (ölçek de yalnız bunlardan). semi=True: bu binaların etiketli anomali saatleri
    kaybın hedeflerinden çıkarılır. stop_buildings verilirse erken durdurma ve ReduceLROnPlateau onların tahmin
    MAE'si ile (L4/L4-semi, val binaları); verilmezse (G7 OOF, §17.2) `fixed_epochs` epoch eğitilir, LR
    planlayıcısı eğitim kaybıyla çalışır ve hiçbir tutulan bina kullanılmaz.
    """
    cfg = ctx.config["a3"]
    set_seed(seed)
    dev = device()
    array = sequence_array(ctx, train_buildings, "full")            # [B, H, 9]
    B, H = len(ctx.buildings), ctx.n_hours
    L, horizon = ctx.config["deep"]["window"], cfg["horizon"]
    observed = torch.tensor(ctx.grid["observed"].to_numpy().reshape(B, H) == 1, device=dev)
    target_ok = observed.clone()
    if semi:
        labels = ctx.grid["anomaly"].fillna(0).to_numpy().reshape(B, H) == 1
        target_ok &= ~torch.tensor(labels, device=dev)
    x = array[..., 0]
    cal = array[..., 2:]                                           # [B, H, 7]
    train_b = np.flatnonzero(np.isin(ctx.buildings, train_buildings))
    rng = np.random.default_rng(seed)

    def sample(buildings, n):
        return (torch.tensor(rng.choice(buildings, n), device=dev),
                torch.tensor(rng.integers(L - 1, H - horizon, n), device=dev))

    def batch_loss(model, b, t):
        steps = torch.arange(1, horizon + 1, device=dev)
        future_t = t[:, None] + steps[None, :]
        pred = model(windows(array, b, t, L), cal[b[:, None], future_t])
        target, ok = x[b[:, None], future_t], target_ok[b[:, None], future_t]
        return ((pred - target).abs() * ok).sum() / ok.sum().clamp(min=1)

    model = Seq2SeqForecaster(array.shape[2], 7, cfg["hidden"], cfg["layers"], cfg["dropout"]).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    if stop_buildings is not None:
        vb, vt = sample(np.flatnonzero(np.isin(ctx.buildings, stop_buildings)), cfg["val_windows"])
    max_epochs = cfg["max_epochs"] if fixed_epochs is None else fixed_epochs
    best, best_state, best_epoch, wait = np.inf, None, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        b, t = sample(train_b, cfg["windows_per_epoch"])
        train_losses = []
        for i in range(0, len(b), cfg["batch_size"]):
            optimizer.zero_grad()
            loss = batch_loss(model, b[i:i + cfg["batch_size"]], t[i:i + cfg["batch_size"]])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        if stop_buildings is None:                                  # G7 OOF: sabit epoch, tutulan bina yok
            scheduler.step(float(np.mean(train_losses)))
            best, best_epoch = float(np.mean(train_losses)), epoch
            continue
        model.eval()
        with torch.no_grad():
            val_loss = float(np.mean([batch_loss(model, vb[i:i + 4096], vt[i:i + 4096]).item()
                                      for i in range(0, len(vb), 4096)]))
        scheduler.step(val_loss)
        if val_loss < best:
            best, best_epoch, wait = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, array, best_epoch, best


def a3_score_grid(ctx, model, array, buildings):
    """Verilen binaların her saati için normalize 1 adım hatası. Çıktı: [len(buildings), H] (ilk 48 saat NaN).

    t saatinin tahmini t-1'de biten pencereden; hata eksik saatlerde NaN; ölçek son 168 saatin medyan hatası.
    """
    cfg = ctx.config["a3"]
    dev = array.device
    H, L = ctx.n_hours, ctx.config["deep"]["window"]
    B = len(ctx.buildings)
    observed = torch.tensor(ctx.grid["observed"].to_numpy().reshape(B, H) == 1, device=dev)
    x, cal = array[..., 0], array[..., 2:]
    rows = np.flatnonzero(np.isin(ctx.buildings, buildings))
    errors = np.full((len(rows), H), np.nan, dtype=np.float32)
    with torch.no_grad():
        for row, building in enumerate(rows):
            t = torch.arange(L - 1, H - 1, device=dev)
            b = torch.full_like(t, int(building))
            preds = []
            for i in range(0, len(t), 4096):
                bb, tt = b[i:i + 4096], t[i:i + 4096]
                preds.append(model(windows(array, bb, tt, L), cal[bb[:, None], tt[:, None] + 1])[:, 0])
            pred = torch.cat(preds)
            err = (pred - x[building, L:]).abs()
            err[~observed[building, L:]] = float("nan")
            errors[row, L:] = err.cpu().numpy()
    return normalized_residual(errors.T, cfg["residual_scale_hours"]).to_numpy().T, rows


def run_a3(ctx, kind, seed, semi=False):
    """L4 (A3): Seq2Seq tahmin modeli train binalarının saatleriyle; skor = |h=1 hatası| / son 168 saatin medyanı.

    semi=True (L4-semi): train binalarının etiketli anomali saatleri kaybın hedeflerinden çıkarılır.
    Girdi kanalları: ölçekli log okuma (ileri doldurulmuş), eksiklik maskesi, takvim. Decoder: hedef saatlerin
    takvimi (önceden bilinir). Kayıp yalnız gözlenen hedef saatlerde (L1, hanedeki F10-opt gibi).
    Erken durdurma val binalarının tahmin MAE'si ile.
    """
    started = time.time()
    split = ctx.split(kind, seed)
    model, array, best_epoch, best = train_a3(ctx, split["train"], seed, semi, stop_buildings=split["val"])
    score_grid, val_b = a3_score_grid(ctx, model, array, split["val"])     # [val bina, H]
    val = ctx.rows(split["val"])
    rows_b = np.searchsorted(val_b, ctx.b_pos[val])
    scores = score_grid[rows_b, ctx.t_pos[val]]
    name = "L4-semi" if semi else "L4"
    record(ctx, name, "-", kind, seed, f1_max_threshold(ctx.y[val], scores), val, scores,
           {"best_epoch": best_epoch, "val_mae": best})
    log_run(ctx, {"model": name, "set": "-", "bolme": kind, "seed": seed, "params": json.dumps(ctx.config["a3"]),
                  "best_epoch": best_epoch, "train_time_s": round(time.time() - started, 1)})


# ----------------------------------------------------------------------------- L5 ve Isolation Forest
def run_ensemble(ctx, kind, seed, feature_set, members=("L1", "L2"), name="L5"):
    """L5: w·L1 + (1−w)·L2 (aynı öznitelik seti); w val PR-AUC ile {0, 0,1, …, 1} ızgarasından.

    L5-FE: members=("L1-FE", "L2-FE"), name="L5-FE".
    """
    val = ctx.rows(ctx.split(kind, seed)["val"])
    l1 = np.load(ctx.scores_dir / f"{kind}_{seed}_{members[0]}_{feature_set}.npy")
    l2 = np.load(ctx.scores_dir / f"{kind}_{seed}_{members[1]}_{feature_set}.npy")
    grid = np.round(np.arange(0, 1.01, 0.1), 1)
    w = max(grid, key=lambda w: average_precision_score(ctx.y[val], w * l1 + (1 - w) * l2))
    scores = w * l1 + (1 - w) * l2
    record(ctx, name, feature_set, kind, seed, f1_max_threshold(ctx.y[val], scores), val, scores,
           {"w_L1": float(w)})


def run_isolation_forest(ctx, kind, seed):
    """Ek: Isolation Forest (etiketsiz) behavior öznitelikleriyle; train binalarında fit, eşik val'de.

    Eksik öznitelikler (ilk 168 saat) train medyanıyla doldurulur. Yalnız literatür referansı.
    """
    cfg = ctx.config["isolation_forest"]
    split = ctx.split(kind, seed)
    X = ctx.tabular("behavior")
    train = ctx.rows(split["train"])
    medians = X[train].median()
    X = X.fillna(medians).to_numpy()
    rng = np.random.default_rng(seed)
    fit_rows = rng.choice(np.flatnonzero(train), size=min(cfg["train_rows"], train.sum()), replace=False)
    forest = IsolationForest(n_estimators=cfg["n_estimators"], max_samples=cfg["max_samples"], random_state=seed,
                             n_jobs=-1).fit(X[fit_rows])
    val = ctx.rows(split["val"])
    scores = -forest.score_samples(X[val])
    record(ctx, "IF", "behavior", kind, seed, f1_max_threshold(ctx.y[val], scores), val, scores)


def write_splits(ctx):
    """Bütün bölmeleri results/lead/tables/splits.csv'ye yazar (protokol §5)."""
    tables = []
    for kind in (["bina"] + (["site"] if ctx.sites is not None else [])):
        for seed in ctx.config["split"]["seeds"]:
            tables.append(split_table(ctx.split(kind, seed), kind, seed))
    out = pd.concat(tables, ignore_index=True)
    out.to_csv(ctx.tables / "splits.csv", index=False)
    print(out.groupby(["bolme", "seed", "parca"]).size().unstack().to_string())


# ----------------------------------------------------------------------------- özet, seçim ve site kontrolü
def family(model):
    """Model adından aile: L0-R2 → L0, L4-semi → L4."""
    return model.split("-")[0]


def summary_table(results, kind="bina"):
    """Seed ortalaması ± std: genel ve dilim metrikleri, model × set satırı başına."""
    part = results[results["bolme"] == kind]
    agg = part.groupby(["model", "set", "dilim"]).agg(
        n_seed=("seed", "nunique"), pr_auc=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
        roc_auc=("roc_auc", "mean"), f1=("f1", "mean"), precision=("precision", "mean"),
        recall=("recall", "mean"), event_recall=("event_recall", "mean")).reset_index()
    wide = agg.pivot_table(index=["model", "set"], columns="dilim",
                           values=["pr_auc", "pr_auc_std", "roc_auc", "f1", "precision", "recall", "event_recall"])
    wide.columns = [f"{d}_{m}" for m, d in wide.columns]
    wide["n_seed"] = agg[agg["dilim"] == "genel"].set_index(["model", "set"])["n_seed"]
    return wide.reset_index()


def select_model(results, kind="bina"):
    """Protokol §12: val genel PR-AUC; en iyiye karşı eşleştirilmiş fark |ort| ≤ 2·std/√n → gürültü düzeyi;
    bunlardan en basit aile (L0 < L4 < L1 < L2 < L3 < L5), aynı ailede ortalaması yüksek olan.
    Isolation Forest yalnız literatür referansıdır; yarışa girmez."""
    genel = results[(results["bolme"] == kind) & (results["dilim"] == "genel") & (results["model"] != "IF")]
    pivot = genel.pivot_table(index="seed", columns=["model", "set"], values="pr_auc")
    pivot = pivot.loc[:, pivot.notna().all()]                       # bütün seed'lerde koşmuş olanlar
    means = pivot.mean()
    best = means.idxmax()
    rows = []
    for key in pivot.columns:
        diff = pivot[key] - pivot[best]
        n = len(diff)
        se2 = 2 * diff.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        rows.append({"model": key[0], "set": key[1], "pr_auc_mean": means[key], "fark_ort": diff.mean(),
                     "fark_2se": se2, "gurultu_duzeyinde": abs(diff.mean()) <= se2 + 1e-12,
                     "aile": family(key[0])})
    table = pd.DataFrame(rows)
    table["basitlik"] = table["aile"].map({f: i for i, f in enumerate(SIMPLICITY)})
    tied = table[table["gurultu_duzeyinde"]].sort_values(["basitlik", "pr_auc_mean"], ascending=[True, False])
    table["secilen"] = False
    table.loc[tied.index[0], "secilen"] = True
    return table.sort_values("pr_auc_mean", ascending=False).reset_index(drop=True)


def top_models(results, k=2, kind="bina"):
    """Site kontrolü için val genel PR-AUC'si en yüksek k öğrenen model (kurallar ve IF hariç)."""
    table = select_model(results, kind)
    learned = table[~table["aile"].isin(["L0"])]
    return list(zip(learned["model"].head(k), learned["set"].head(k)))


def summarize(ctx):
    """val_summary.csv/.md ve val_selection.csv üretir; tabloyu basar."""
    results = pd.read_csv(ctx.tables / "val_results.csv")
    summary = summary_table(results)
    selection = select_model(results)
    summary.to_csv(ctx.tables / "val_summary.csv", index=False)
    selection.to_csv(ctx.tables / "val_selection.csv", index=False)
    if (results["bolme"] == "site").any():
        summary_table(results, "site").to_csv(ctx.tables / "val_summary_site.csv", index=False)
    cols = ["model", "set", "n_seed", "genel_pr_auc", "genel_pr_auc_std", "genel_roc_auc", "genel_f1",
            "genel_precision", "genel_recall", "genel_event_recall", "gap_edge_recall", "frozen_recall",
            "other_pr_auc", "other_f1", "other_event_recall", "paylasilan_pr_auc", "tekil_pr_auc"]
    short = summary[[c for c in cols if c in summary.columns]].sort_values("genel_pr_auc", ascending=False)
    lines = ["| " + " | ".join(short.columns) + " |", "|" + "---|" * len(short.columns)]
    for row in short.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |")
    (ctx.tables / "val_summary.md").write_text(
        "# LEAD val özeti (bina bölmesi, 3 seed ortalaması) — test değil\n\n" + "\n".join(lines) + "\n",
        encoding="utf-8")
    pd.set_option("display.width", 250)
    print(short.round(4).to_string(index=False))
    print(selection.round(4).to_string(index=False))


def run_site_check(ctx, seeds):
    """Siteye göre bölme (sağlamlık): kurallar + val'de en iyi 2 öğrenen model. Ayarlar bina bölmesinin aynı
    seed'inden (Optuna tekrarlanmaz); eşik ve erken durdurma site bölmesinin kendi val'inde."""
    results = pd.read_csv(ctx.tables / "val_results.csv")
    chosen = top_models(results)
    runs = pd.read_csv(ctx.tables.parent / "experiments.csv")
    print("Site kontrolü modelleri:", chosen)
    for seed in seeds:
        run_rules(ctx, "site", seed)
        for model, feature_set in chosen:
            row = runs[(runs["model"] == model) & (runs["set"] == feature_set) & (runs["bolme"] == "bina")
                       & (runs["seed"] == seed)]
            params = json.loads(row["params"].iloc[0]) if len(row) else None
            if model == "L1":
                run_lightgbm(ctx, "site", seed, feature_set, params=params)
            elif model in ("L2", "L3"):
                run_deep(ctx, model, "site", seed, feature_set, params=params)
            elif model in ("L4", "L4-semi"):
                run_a3(ctx, "site", seed, semi=model == "L4-semi")
            elif model == "L5":
                for member, member_set in (("L1", feature_set), ("L2", feature_set)):
                    member_row = runs[(runs["model"] == member) & (runs["set"] == member_set)
                                      & (runs["bolme"] == "bina") & (runs["seed"] == seed)]
                    member_params = json.loads(member_row["params"].iloc[0])
                    if member == "L1":
                        run_lightgbm(ctx, "site", seed, member_set, params=member_params)
                    else:
                        run_deep(ctx, "L2", "site", seed, member_set, params=member_params)
                run_ensemble(ctx, "site", seed, feature_set)
