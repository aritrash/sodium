import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "training/datasets/12ax7/12ax7_koren.csv"
OUTPUT = Path(__file__).resolve().parent / "results"

SEED = 42
EPOCHS = 1000
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
PATIENCE = 60

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_dataset():
    data = np.loadtxt(DATASET, delimiter=",", skiprows=1, dtype=np.float32)
    x = data[:, :2]
    y = data[:, 2:3]
    vgk, vpk = x[:, 0], x[:, 1]

    val_mask = (vgk >= -4.0) & (vgk <= -3.2) & (vpk >= 90.0) & (vpk <= 160.0)
    test_mask = (vgk >= -2.8) & (vgk <= -2.0) & (vpk >= 190.0) & (vpk <= 260.0)
    train_mask = ~(val_mask | test_mask)

    return x[train_mask], y[train_mask], x[val_mask], y[val_mask], x[test_mask], y[test_mask]

class TubeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.network(x)

def prepare_tensors(x, y, x_mean, x_std, y_mean, y_std, device):
    x = torch.tensor((x - x_mean) / x_std, dtype=torch.float32, device=device)
    y = torch.tensor((np.log1p(y) - y_mean) / y_std, dtype=torch.float32, device=device)
    return x, y

def evaluate(model, x, y, x_mean, x_std, y_mean, y_std, device):
    model.eval()
    with torch.no_grad():
        tx, _ = prepare_tensors(x, y, x_mean, x_std, y_mean, y_std, device)
        prediction = model(tx).cpu().numpy() * y_std + y_mean
    prediction = np.expm1(prediction)
    error = prediction - y
    active = y[:, 0] > 0.1
    return {
        "mae_mA": float(np.mean(np.abs(error))),
        "rmse_mA": float(np.sqrt(np.mean(error ** 2))),
        "max_absolute_error_mA": float(np.max(np.abs(error))),
        "mean_absolute_percentage_error_percent": float(np.mean(np.abs(error[active] / y[active])) * 100) if np.any(active) else None
    }

def main():
    set_seed(SEED)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    x_train, y_train, x_val, y_val, x_test, y_test = load_dataset()
    x_mean, x_std = x_train.mean(axis=0), x_train.std(axis=0)
    y_log = np.log1p(y_train)
    y_mean, y_std = y_log.mean(axis=0), y_log.std(axis=0)
    x_std = np.maximum(x_std, 1e-8)
    y_std = np.maximum(y_std, 1e-8)

    tx, ty = prepare_tensors(x_train, y_train, x_mean, x_std, y_mean, y_std, device)
    vx, vy = prepare_tensors(x_val, y_val, x_mean, x_std, y_mean, y_std, device)
    loader = DataLoader(TensorDataset(tx, ty), batch_size=BATCH_SIZE, shuffle=True)

    model = TubeModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_epoch = 0
    patience = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * bx.size(0)

        train_loss = running_loss / len(loader.dataset)
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(vx), vy).item()

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            patience = 0
            torch.save({"model_state": model.state_dict(), "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std, "architecture": "2-64-64-64-1", "activation": "SiLU"}, OUTPUT / "best_model.pt")
        else:
            patience += 1

        if epoch == 1 or epoch % 25 == 0:
            print(f"Epoch {epoch:4d} | Train: {train_loss:.7f} | Validation: {val_loss:.7f}")

        if patience >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    checkpoint = torch.load(OUTPUT / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    metrics = {
        "experiment": "SODIUM-0001",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "dataset_samples": int(len(x_train) + len(x_val) + len(x_test)),
        "split_samples": {"train": len(x_train), "validation": len(x_val), "test": len(x_test)},
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "test_metrics": evaluate(model, x_test, y_test, x_mean, x_std, y_mean, y_std, device),
        "training_config": {"seed": SEED, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "max_epochs": EPOCHS, "patience": PATIENCE}
    }

    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=4))
    (OUTPUT / "history.json").write_text(json.dumps(history, indent=4))
    print("\nTest results:")
    print(json.dumps(metrics["test_metrics"], indent=4))
    print(f"\nBest checkpoint: {OUTPUT / 'best_model.pt'}")

if __name__ == "__main__":
    main()
