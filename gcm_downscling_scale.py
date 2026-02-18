import numpy as np
import pickle
import os
from pathlib import Path

base_dir = Path(os.environ.get("SRGAN_BASE_DIR", Path(__file__).resolve().parent))

# --------------------------------------------------
# Load scaler
# --------------------------------------------------
with open(
    base_dir / 'models' / 'dy_v0.4' / 'scaler.pkl',
    'rb'
) as f:
    scaler = pickle.load(f)

# --------------------------------------------------
# Lazy-load scaled predictions
# --------------------------------------------------
scaled_path = base_dir / 'gcm_ds' / '0.1' / 'pr' / 'ACCESS-CM2' / 'y_pred_4_scaled.npy'

out = np.load(scaled_path, mmap_mode='r')  # 🚨 lazy
print(f"[INFO] Loaded scaled file lazily: shape={out.shape}, dtype={out.dtype}")

# Handle (T, H, W, 1) vs (T, H, W)
if out.ndim == 4:
    out = out[..., 0]

T, H, W = out.shape
N = T * H * W

# --------------------------------------------------
# Prepare output memmap
# --------------------------------------------------
out_path = scaled_path.with_name('y_pred_4.npy')

yinv = np.memmap(
    out_path,
    dtype=np.float32,
    mode='w+',
    shape=(T, H, W)
)

# --------------------------------------------------
# Chunked inverse transform
# --------------------------------------------------
chunk_size = 2_000_000   # ~8 MB per chunk, safe

flat_in = out.reshape(-1)          # memmap-backed view
flat_out = yinv.reshape(-1)        # memmap-backed output

for i in range(0, N, chunk_size):
    j = min(i + chunk_size, N)

    block = flat_in[i:j].reshape(-1, 1)
    inv = scaler.inverse_transform(block).astype(np.float32)

    # Physical constraint
    np.maximum(inv, 0, out=inv)

    flat_out[i:j] = inv[:, 0]

    if i % (50 * chunk_size) == 0:
        print(f"[INFO] Inverse-scaled {i:,} / {N:,}")

# Flush to disk
yinv.flush()

print(f"[DONE] Inverse-scaled data written to: {out_path}")
print(f"[DONE] Final shape: {(T, H, W)}")
