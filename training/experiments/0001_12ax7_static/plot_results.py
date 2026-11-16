import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parents[2]
DATASET = ROOT / "training/datasets/12ax7/12ax7_koren.csv"
CHECKPOINT = EXPERIMENT_DIR / "results/best_model.pt"
OUTPUT = EXPERIMENT_DIR / "results/figures"

sys.path.insert(0, str(EXPERIMENT_DIR))
from koren_reference import plate_current

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TubeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(torch.nn.Linear(2, 64), torch.nn.SiLU(), torch.nn.Linear(64, 64), torch.nn.SiLU(), torch.nn.Linear(64, 64), torch.nn.SiLU(), torch.nn.Linear(64, 1))

    def forward(self, x):
        return self.network(x)

def load_model():
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model = TubeModel().to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint

def predict(model, checkpoint, vgk, vpk):
    x = np.column_stack((np.asarray(vgk).ravel(), np.asarray(vpk).ravel())).astype(np.float32)
    x_mean = checkpoint["x_mean"]
    x_std = checkpoint["x_std"]
    y_mean = checkpoint["y_mean"]
    y_std = checkpoint["y_std"]

    tx = torch.tensor((x - x_mean) / x_std, dtype=torch.float32, device=DEVICE)

    with torch.inference_mode():
        output = model(tx).cpu().numpy() * y_std + y_mean

    return np.expm1(output).reshape(np.shape(vgk))

def plot_characteristics(model, checkpoint):
    vpk = np.linspace(50.0, 300.0, 300)
    vgk_values = [-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0]

    fig, ax = plt.subplots(figsize=(11, 7))

    for vgk in vgk_values:
        reference = plate_current(np.full_like(vpk, vgk), vpk) * 1000.0
        predicted = predict(model, checkpoint, np.full_like(vpk, vgk), vpk)
        ax.plot(vpk, reference, linewidth=2, label=f"Koren Vgk={vgk:.0f} V")
        ax.plot(vpk, predicted, linestyle="--", linewidth=1.5, label=f"Neural Vgk={vgk:.0f} V")

    ax.set_title("12AX7 Plate Characteristics: Koren vs Neural Model")
    ax.set_xlabel("Plate-to-Cathode Voltage Vpk (V)")
    ax.set_ylabel("Plate Current Ip (mA)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT / "01_plate_characteristics.png", dpi=200)
    plt.close(fig)

def plot_error_heatmap(model, checkpoint):
    vgk = np.linspace(-6.0, 0.0, 150)
    vpk = np.linspace(50.0, 300.0, 150)
    VGK, VPK = np.meshgrid(vgk, vpk, indexing="ij")

    reference = plate_current(VGK, VPK) * 1000.0
    predicted = predict(model, checkpoint, VGK, VPK)
    error = np.abs(predicted - reference)

    fig, ax = plt.subplots(figsize=(11, 7))
    image = ax.imshow(error, origin="lower", aspect="auto", extent=[50, 300, -6, 0], cmap="inferno")
    fig.colorbar(image, ax=ax, label="Absolute Error (mA)")

    ax.add_patch(Rectangle((90, -4), 70, 0.8, fill=False, edgecolor="cyan", linewidth=2, label="Validation region"))
    ax.add_patch(Rectangle((190, -2.8), 70, 0.8, fill=False, edgecolor="lime", linewidth=2, label="Test region"))

    ax.set_title("Absolute Prediction Error Across the Operating Domain")
    ax.set_xlabel("Plate-to-Cathode Voltage Vpk (V)")
    ax.set_ylabel("Grid-to-Cathode Voltage Vgk (V)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "02_error_heatmap.png", dpi=200)
    plt.close(fig)

def plot_parity(model, checkpoint):
    data = np.loadtxt(DATASET, delimiter=",", skiprows=1)
    vgk, vpk, reference = data[:, 0], data[:, 1], data[:, 2]

    test_mask = (vgk >= -2.8) & (vgk <= -2.0) & (vpk >= 190.0) & (vpk <= 260.0)
    actual = reference[test_mask]
    predicted = predict(model, checkpoint, vgk[test_mask], vpk[test_mask])

    low = min(actual.min(), predicted.min())
    high = max(actual.max(), predicted.max())

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(actual, predicted, s=18, alpha=0.65, label="Held-out test samples")
    ax.plot([low, high], [low, high], "r--", linewidth=2, label="Ideal prediction")

    ax.set_title("Neural Model Prediction Parity")
    ax.set_xlabel("Koren Reference Current (mA)")
    ax.set_ylabel("Predicted Current (mA)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(OUTPUT / "03_prediction_parity.png", dpi=200)
    plt.close(fig)

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")

    model, checkpoint = load_model()

    plot_characteristics(model, checkpoint)
    print("Generated plate characteristic curves.")

    plot_error_heatmap(model, checkpoint)
    print("Generated absolute error heatmap.")

    plot_parity(model, checkpoint)
    print("Generated prediction parity plot.")

    print(f"\nFigures saved in: {OUTPUT}")

if __name__ == "__main__":
    main()
