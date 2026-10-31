"""A2: LSTM-Autoencoder'ı yalnız train'de eğitir (early stopping: val yeniden yapılandırma kaybı)."""

import argparse
from copy import deepcopy
from datetime import date
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data.load import PROJECT_ROOT, load_config
from src.data.split import chronological_split
from src.models.autoencoder import LSTMAutoencoder, make_windows, reconstruct
from src.utils.console import setup_console
from src.utils.results_io import resolve_path, upsert_csv
from src.utils.seed import set_seed


def clean_windows(frame, target, window, mean, scale, exclude=None):
    """Bir kümeden ölçeklenmiş, doldurulmuş saat içermeyen pencereler üretir.

    Girdi: küme tablosu, hedef, pencere uzunluğu, train ölçek istatistikleri,
    isteğe bağlı (başlangıç, bitiş) dışlama aralığı (kontaminasyon ablasyonu).
    Çıktı: [N, window] float32 dizi.
    Neden: AE "normal" deseni öğrenmeli; doldurulmuş (yapay) saatler ve
    bilinen olağandışı dönem eğitimden uzak tutulabilir.
    """
    values = (frame[target].to_numpy(dtype=np.float64) - mean) / scale
    windows, starts = make_windows(values, window)
    bad = frame["is_imputed"].to_numpy(dtype=bool)
    if exclude is not None:
        start, end = pd.Timestamp(exclude[0]), pd.Timestamp(exclude[1]) + pd.Timedelta(days=1)
        bad = bad | ((frame.index >= start) & (frame.index < end))
    # Penceredeki herhangi bir saat kötü ise pencere atılır.
    bad_cumsum = np.concatenate(([0], np.cumsum(bad)))
    keep = (bad_cumsum[starts + window] - bad_cumsum[starts]) == 0
    return windows[keep].astype(np.float32)


def train_autoencoder(df, config, exp_id, device_name, exclude_contamination=False):
    """LSTM-AE'yi eğitir, checkpoint/geçmiş/eğri ve experiments satırını yazar.

    Girdi: saatlik tablo, config (anomaly.ae), exp_id, cihaz, ablasyon bayrağı.
    Çıktı: (en iyi val kaybı, epoch geçmişi).
    Neden: Ölçek istatistikleri yalnız train'den; test kümesi hiç kullanılmaz.
    """
    settings = config["anomaly"]["ae"]
    set_seed(config["seed"])
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    splits = chronological_split(df, config)
    target = config["preprocess"]["target"]
    window = settings["window"]
    mean = splits["train"][target].mean()
    scale = splits["train"][target].std(ddof=0)
    exclude = config["anomaly"]["contamination_exclude"] if exclude_contamination else None

    train_windows = clean_windows(splits["train"], target, window, mean, scale, exclude)
    val_windows = clean_windows(splits["val"], target, window, mean, scale)
    X_train = torch.from_numpy(train_windows[:, :, None]).to(device)  # [N, 24, 1]
    X_val = torch.from_numpy(val_windows[:, :, None]).to(device)

    model = LSTMAutoencoder(1, settings["hidden"], settings["latent"], settings["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"],
                                 weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=settings["scheduler_factor"],
        patience=settings["scheduler_patience"],
    )
    loss_function = nn.MSELoss()
    batch_size = settings["batch_size"]
    best_loss, best_state, best_epoch, waited = float("inf"), None, 0, 0
    history = []
    start_time = time.perf_counter()

    for epoch in range(1, settings["max_epochs"] + 1):
        model.train()
        # AE pencereleri bağımsız örnekler gibi ele alınır; eğitim sırası karıştırılır.
        order = torch.randperm(len(X_train), device=device)
        total = torch.zeros((), device=device)
        for batch_start in range(0, len(X_train), batch_size):
            batch = X_train[order[batch_start:batch_start + batch_size]]
            optimizer.zero_grad()
            loss = loss_function(model(batch), batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"])
            optimizer.step()
            total += loss.detach() * len(batch)
        model.eval()
        with torch.no_grad():
            val_total = 0.0
            for batch_start in range(0, len(X_val), batch_size):
                batch = X_val[batch_start:batch_start + batch_size]
                val_total += loss_function(model(batch), batch).item() * len(batch)
        train_loss = total.item() / len(X_train)
        val_loss = val_total / len(X_val)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"{exp_id} epoch {epoch}: train_loss={train_loss:.5f}, val_loss={val_loss:.5f}")
        # Early stopping ve model seçimi: val yeniden yapılandırma kaybı (README §4.7).
        if val_loss < best_loss:
            best_loss, best_epoch, waited = val_loss, epoch, 0
            best_state = deepcopy(model.state_dict())
        else:
            waited += 1
        if waited >= settings["early_stopping_patience"]:
            break
    elapsed = time.perf_counter() - start_time

    models_dir = resolve_path(PROJECT_ROOT, config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {k: v.cpu() for k, v in best_state.items()},
        "target_mean": float(mean), "target_scale": float(scale),
        "settings": settings, "exclude": exclude,
    }, models_dir / f"{exp_id}_best.pt")

    tables_dir = resolve_path(PROJECT_ROOT, config["paths"]["tables_dir"])
    figures_dir = resolve_path(PROJECT_ROOT, config["paths"]["figures_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(tables_dir / f"{exp_id}_training_history.csv", index=False)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], label="Train")
    axis.plot([h["epoch"] for h in history], [h["val_loss"] for h in history], label="Val (enjeksiyonsuz)")
    axis.set_title(f"{exp_id} LSTM-AE öğrenme eğrisi (train/val)")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Yeniden yapılandırma MSE (ölçekli, boyutsuz)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures_dir / f"{exp_id}_learning_curve.png", dpi=200)
    plt.close(figure)

    note = (
        f"LSTM-AE window={window}; hidden={settings['hidden']}->latent={settings['latent']}; "
        f"best_val_recon_mse={best_loss:.6f}; train_windows={len(train_windows)}; "
        f"exclude={exclude}; device={device.type}; detection metrics in anomaly_val_results.csv"
    )
    upsert_csv(resolve_path(PROJECT_ROOT, config["paths"]["experiments_file"]), [{
        "exp_id": exp_id, "date": date.today().isoformat(), "model": "LSTM-AE",
        "feature_set": "A", "lookback": window, "hidden": settings["hidden"],
        "layers": 2, "dropout": settings["dropout"], "lr": settings["lr"],
        "batch": batch_size, "epochs_run": len(history), "best_epoch": best_epoch,
        "seed": config["seed"], "val_mae": "", "val_rmse": "", "test_mae": "",
        "test_rmse": "", "train_time_s": elapsed, "note": note,
    }], ["exp_id"])
    return best_loss, pd.DataFrame(history)


def main():
    """CLI: --exp, --exclude_contamination, --max_epochs (smoke test için), --device."""
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", default="A2")
    parser.add_argument("--exclude_contamination", action="store_true",
                        help="config'teki olağandışı dönemi train'den çıkar")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--latent", type=int, default=None,
                        help="Gizil boyut (tanılama; varsayılan config: 32)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.max_epochs is not None:
        config["anomaly"]["ae"]["max_epochs"] = args.max_epochs
    if args.latent is not None:
        config["anomaly"]["ae"]["latent"] = args.latent
    if args.seed is not None:
        config["seed"] = args.seed
    df = pd.read_parquet(resolve_path(PROJECT_ROOT, config["paths"]["processed_file"]))
    best_loss, history = train_autoencoder(
        df, config, args.exp, args.device, args.exclude_contamination
    )
    print(f"{args.exp}: en iyi val yeniden yapılandırma MSE {best_loss:.5f}, {len(history)} epoch")


if __name__ == "__main__":
    main()
