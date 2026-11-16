import argparse
import json
from pathlib import Path
import numpy as np

MU = 100.0
EX = 1.4
KG1 = 1060.0
KP = 600.0
KVB = 300.0

def plate_current(vgk: np.ndarray, vpk: np.ndarray) -> np.ndarray:
    z = KP * (1.0 / MU + vgk / np.sqrt(KVB + np.square(vpk)))
    e1 = (vpk / KP) * np.logaddexp(0.0, z)
    return (2.0 / KG1) * np.power(np.maximum(e1, 0.0), EX)

def generate_dataset(output: Path, grid_size: int = 150):
    vgk = np.linspace(-6.0, 0.0, grid_size)
    vpk = np.linspace(50.0, 300.0, grid_size)
    VGK, VPK = np.meshgrid(vgk, vpk, indexing="ij")
    IP = plate_current(VGK, VPK)
    data = np.column_stack((VGK.ravel(), VPK.ravel(), IP.ravel() * 1000.0))
    output.mkdir(parents=True, exist_ok=True)
    np.savetxt(output / "12ax7_koren.csv", data, delimiter=",", header="vgk_V,vpk_V,ip_mA", comments="")
    metadata = {"model": "Koren Triode Model", "tube": "12AX7", "parameters": {"mu": MU, "ex": EX, "kg1": KG1, "kp": KP, "kvb": KVB}, "grid": {"vgk_V": [-6.0, 0.0], "vpk_V": [50.0, 300.0], "resolution": grid_size}, "samples": len(data), "target": "plate_current_mA", "limitations": ["Static phenomenological model", "Negative grid-to-cathode voltage domain", "Does not model grid current or dynamic parasitics"]}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=4))
    print(f"Generated {len(data):,} samples")
    print(f"Plate current range: {IP.min() * 1000:.6f} - {IP.max() * 1000:.6f} mA")
    print(f"Dataset: {output / '12ax7_koren.csv'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("training/datasets/12ax7"))
    parser.add_argument("--resolution", type=int, default=150)
    args = parser.parse_args()
    generate_dataset(args.output, args.resolution)
