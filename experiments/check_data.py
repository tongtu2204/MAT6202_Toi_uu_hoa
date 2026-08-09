import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

for file_name in ["ridge_raw.npz", "ridge.npz", "poly.npz", "lasso.npz"]:
    path = DATA_DIR / file_name

    if not path.exists():
        print(f"\n{file_name}: NOT FOUND")
        continue

    data = np.load(path)

    print(f"\n===== {file_name} =====")
    print("Keys:", data.files)

    for key in data.files:
        arr = data[key]
        print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")