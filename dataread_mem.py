import os
import numpy as np
import pickle
from netCDF4 import Dataset
from sklearn.model_selection import train_test_split

def read_saved_data(name, path_output, scaler):
    """
    Load cached numpy data and apply scaler.
    Returns float32 array with shape (N, H, W, C)
    """
    arr = np.load(f"{path_output}/{name}.npy", mmap_mode="r")  # mmap saves RAM
    shp = arr.shape
    print(f"[read_saved_data] {name} original shape: {shp}", flush=True)

    flat = arr.reshape(-1, 1)
    scaled = scaler.transform(flat)

    scaled = scaled.reshape(shp[0], shp[1], shp[2], 1)
    return scaled.astype(np.float32, copy=False)

# -------------------------
# Scalers that mimic sklearn API but are streaming-friendly
# -------------------------
class StreamingStandardScaler:
    def __init__(self, eps=1e-12):
        self.eps = eps
        self.n = 0
        self.s1 = 0.0
        self.s2 = 0.0
        self.mean_ = None
        self.scale_ = None

    def partial_fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.n += x.size
        self.s1 += float(x.sum())
        self.s2 += float((x * x).sum())
        return self

    def finalize(self):
        mean = self.s1 / self.n
        var = max(self.s2 / self.n - mean * mean, 0.0)
        std = float(np.sqrt(var) + self.eps)
        self.mean_ = mean
        self.scale_ = std
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean_) / self.scale_

    def inverse_transform(self, x):
        x = np.asarray(x, dtype=np.float32)
        return x * self.scale_ + self.mean_


class StreamingMinMaxScaler:
    def __init__(self, eps=1e-12):
        self.eps = eps
        self.data_min_ = None
        self.data_max_ = None
        self.scale_ = None  # (max-min)
        self.min_ = None    # same naming style as sklearn not required, but kept

    def partial_fit(self, x):
        x = np.asarray(x, dtype=np.float32)
        xmin = float(np.nanmin(x))
        xmax = float(np.nanmax(x))
        if self.data_min_ is None:
            self.data_min_ = xmin
            self.data_max_ = xmax
        else:
            self.data_min_ = min(self.data_min_, xmin)
            self.data_max_ = max(self.data_max_, xmax)
        return self

    def finalize(self):
        rng = self.data_max_ - self.data_min_
        if rng <= 0:
            rng = 1.0
        self.scale_ = rng
        self.min_ = self.data_min_
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float32)
        return (x - self.data_min_) / self.scale_

    def inverse_transform(self, x):
        x = np.asarray(x, dtype=np.float32)
        return x * self.scale_ + self.data_min_


# -------------------------
# File reader (stream one year at a time)
# -------------------------
def _open_daymet_file(var, year, deg, Daymet_ERA5=True, yearly=True):
    """
    Reuses your naming logic, but opens ONE file at a time.
    """
    if yearly:
        if Daymet_ERA5:
            try:
                return Dataset(f"/lustre/orion/proj-shared/cli138/dr6/NA-Downscaling/data/Daymet_ERA5_{var}_dy_{year}_{deg}deg.nc")
            except Exception:
                if deg == 0.25:
                    return Dataset(f"/lustre/orion/proj-shared/cli138/dr6/NA-Downscaling/data/Daymet_ERA5_{var}_dy_{year}_0p25deg.nc")
                elif deg == 0.0416:
                    return Dataset(f"/lustre/orion/proj-shared/cli138/dr6/NA-Downscaling/data/Daymet_ERA5_{var}_dy_{year}_trim.nc")
                raise
        else:
            return Dataset(f"/lustre/orion/cli138/proj-shared/7hn/data/Daymet/DaymetV4_VIC4_{var}_{year}_{deg}deg_US.nc")
    else:
        raise NotImplementedError("monthly path not wired here; can be added if you need it.")


def _read_var_from_file(nc, var, yearly=True):
    key = f"{var}_dy" if yearly else f"{var}_3h"
    arr = nc.variables[key][:]
    arr = np.array(arr, dtype=np.float32, copy=False)

    # match your cleaning rules
    arr[np.isnan(arr)] = 0.0
    if var in ("pr", "prcp"):
        arr[arr < 0] = 0.0
    return arr


def _iter_year_blocks(var, year_start, year_end, deg, Daymet_ERA5=True, yearly=True):
    """
    Yields (year, array) for year_start..year_end-1 (matching your original range()).
    """
    for year in range(year_start, year_end):
        print(f"read {deg} degree data from year {year}", flush=True)
        nc = _open_daymet_file(var, year, deg, Daymet_ERA5=Daymet_ERA5, yearly=yearly)
        try:
            arr = _read_var_from_file(nc, var, yearly=yearly)
        finally:
            nc.close()
        yield year, arr


# -------------------------
# Elevation: avoid tiling tt times in memory
# -------------------------
def _read_elev_2d(deg):
    if deg == 1:
        deg_str = '1deg'
    elif deg == 0.25:
        deg_str = '0p25deg'
    elif deg == 0.0416:
        deg_str = 'trim'
    nc = Dataset(f"/lustre/orion/proj-shared/cli138/dr6/NA-Downscaling/DEM/final-elev/VICa_DEM_{deg_str}_fill0.nc")
    try:
        elev2d = np.array(nc.variables["DEM"][:], dtype=np.float32)  # (x,y)
    finally:
        nc.close()
    return elev2d


# -------------------------
# Main function: same outputs as your daymetread()
# -------------------------
def daymetread(
    path_output,
    checkpoint_dir,
    elevation=False,
    elevation_hr=False,
    Daymet_ERA5=False,
    high_deg=False,
    scaler="standard",
    var="prcp",
    year_start=1980,
    year_end=1990,
    yearly=True,
    test_size=0.2,
    random_state=42,
):
    """
    Produces the same artifacts as your original daymetread():

      - checkpoint_dir/scaler.pkl  (object with transform / inverse_transform)
      - path_output/x_train.npy    (UNSCALED, original units, shape [Ttrain, nlr1, nlr2])
      - path_output/x_test.npy
      - path_output/y_train.npy    (UNSCALED, original units, shape [Ttrain, nhr1, nhr2])
      - path_output/y_test.npy

    Elevation behavior preserved:
      - if elevation: writes elev_lr_scaled.npy (scaled 0..1)
      - if elevation_hr: writes elev_hr_scaled.npy (scaled 0..1)

    NOTE: The train/test split is identical to train_test_split over the time dimension,
    but done via indices to avoid holding the full arrays.
    """
    os.makedirs(path_output, exist_ok=True)
    # clear everything in path_output
    for fname in os.listdir(path_output):
        if fname.endswith(".npy"):
            os.remove(os.path.join(path_output, fname))
    os.makedirs(checkpoint_dir, exist_ok=True)

    # degrees logic identical to yours
    deg_hr = "0p25"
    deg_lr = 1
    if high_deg == 1:
        deg_hr = 0.0416
        deg_lr = 0.25
    elif high_deg == 2:
        deg_hr = 0.0416
        deg_lr = 1

    # ------------------------------------------------------------
    # Pass 0: determine shapes and total tt without concatenating
    # ------------------------------------------------------------
    # Read first year for LR/HR shapes + timestep count pattern
    _, lr0 = next(_iter_year_blocks(var, year_start, year_start + 1, deg_lr, Daymet_ERA5=Daymet_ERA5, yearly=yearly))
    _, hr0 = next(_iter_year_blocks(var, year_start, year_start + 1, deg_hr, Daymet_ERA5=Daymet_ERA5, yearly=yearly))

    nlr1, nlr2 = lr0.shape[1], lr0.shape[2]
    nhr1, nhr2 = hr0.shape[1], hr0.shape[2]

    # total timesteps across years
    tt = 0
    for _, lr_y in _iter_year_blocks(var, year_start, year_end, deg_lr, Daymet_ERA5=Daymet_ERA5, yearly=yearly):
        tt += lr_y.shape[0]
    print(f"[info] total timesteps tt={tt}", flush=True)

    # ------------------------------------------------------------
    # Pass 1: fit scaler on BOTH HR and LR values (streaming)
    # ------------------------------------------------------------
    if scaler == "standard":
        sc = StreamingStandardScaler()
    elif scaler == "minmax":
        sc = StreamingMinMaxScaler()
    else:
        raise NotImplementedError(
            "To keep this streaming and memory-safe, this implementation supports scaler='standard' or 'minmax'. "
            "If you truly need 'robust' or 'log', paste your Log1pScaler implementation and I will add a streaming-safe version."
        )

    # stream LR
    for _, lr_y in _iter_year_blocks(var, year_start, year_end, deg_lr, Daymet_ERA5=Daymet_ERA5, yearly=yearly):
        sc.partial_fit(lr_y)

    # stream HR
    for _, hr_y in _iter_year_blocks(var, year_start, year_end, deg_hr, Daymet_ERA5=Daymet_ERA5, yearly=yearly):
        sc.partial_fit(hr_y)

    sc.finalize()
    print(f"[scaler] type={scaler}", flush=True)
    if scaler == "standard":
        print(f"[scaler] mean={sc.mean_:.6f}, std={sc.scale_:.6f}", flush=True)
    else:
        print(f"[scaler] min={sc.data_min_:.6f}, max={sc.data_max_:.6f}", flush=True)

    # Save scaler.pkl (same role as your original)
    with open(os.path.join(checkpoint_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(sc, f)

    # ------------------------------------------------------------
    # Train/test split identical to sklearn train_test_split on time dim
    # ------------------------------------------------------------
    all_idx = np.arange(tt)
    train_idx, test_idx = train_test_split(
        all_idx, test_size=test_size, random_state=random_state
    )

    # We will write in the same order as train_test_split returns.
    n_train = len(train_idx)
    n_test = len(test_idx)
    print(f"[split] train={n_train} test={n_test}", flush=True)

    # mapping global timestep -> position in train/test arrays
    train_pos = np.full(tt, -1, dtype=np.int64)
    test_pos = np.full(tt, -1, dtype=np.int64)
    train_pos[train_idx] = np.arange(n_train)
    test_pos[test_idx] = np.arange(n_test)

    # ------------------------------------------------------------
    # Create four .npy outputs as memmaps (UNSCALED values)
    # (This matches your invtrans_write outputs.)
    # ------------------------------------------------------------
    x_train = np.lib.format.open_memmap(
        os.path.join(path_output, "x_train.npy"),
        mode="w+", dtype=np.float32, shape=(n_train, nlr1, nlr2)
    )
    x_test = np.lib.format.open_memmap(
        os.path.join(path_output, "x_test.npy"),
        mode="w+", dtype=np.float32, shape=(n_test, nlr1, nlr2)
    )
    y_train = np.lib.format.open_memmap(
        os.path.join(path_output, "y_train.npy"),
        mode="w+", dtype=np.float32, shape=(n_train, nhr1, nhr2)
    )
    y_test = np.lib.format.open_memmap(
        os.path.join(path_output, "y_test.npy"),
        mode="w+", dtype=np.float32, shape=(n_test, nhr1, nhr2)
    )

    # ------------------------------------------------------------
    # Optional elevation: preserve your current outputs (scaled elevation saved separately)
    # (Your x_train/y_train files do NOT include elevation, because invtrans_write drops channels.)
    # ------------------------------------------------------------
    if elevation:
        elev2d_lr = _read_elev_2d(deg_lr)  # (nlr1,nlr2)
        e_min = float(np.min(elev2d_lr))
        e_max = float(np.max(elev2d_lr))
        denom = (e_max - e_min) if (e_max > e_min) else 1.0
        elev_lr_scaled = ((elev2d_lr - e_min) / denom).astype(np.float32)

        # match your saved shape: (tt, nlr1, nlr2, 1)
        elev_lr_out = np.lib.format.open_memmap(
            os.path.join(path_output, "elev_lr_scaled.npy"),
            mode="w+", dtype=np.float32, shape=(tt, nlr1, nlr2, 1)
        )
        for t in range(tt):
            elev_lr_out[t, :, :, 0] = elev_lr_scaled
        elev_lr_out.flush()

        if elevation_hr:
            elev2d_hr = _read_elev_2d(deg_hr)
            elev_hr_scaled = ((elev2d_hr - e_min) / denom).astype(np.float32)
            elev_hr_out = np.lib.format.open_memmap(
                os.path.join(path_output, "elev_hr_scaled.npy"),
                mode="w+", dtype=np.float32, shape=(tt, nhr1, nhr2, 1)
            )
            for t in range(tt):
                elev_hr_out[t, :, :, 0] = elev_hr_scaled
            elev_hr_out.flush()

    # ------------------------------------------------------------
    # Pass 2: stream years again and write UNSCALED LR/HR into the four files
    # (This is equivalent to: transform -> split -> inverse_transform -> save)
    # but avoids ever materializing the 4D scaled tensors.
    # ------------------------------------------------------------
    global_t = 0

    # LR
    for _, lr_y in _iter_year_blocks(var, year_start, year_end, deg_lr, Daymet_ERA5=Daymet_ERA5, yearly=yearly):
        T = lr_y.shape[0]
        for k in range(T):
            t = global_t + k
            p = train_pos[t]
            if p != -1:
                x_train[p] = lr_y[k]
            else:
                q = test_pos[t]
                if q != -1:
                    x_test[q] = lr_y[k]
        global_t += T
        x_train.flush(); x_test.flush()

    # HR (reset global index and stream again)
    global_t = 0
    for _, hr_y in _iter_year_blocks(var, year_start, year_end, deg_hr, Daymet_ERA5=Daymet_ERA5, yearly=yearly):
        T = hr_y.shape[0]
        for k in range(T):
            t = global_t + k
            p = train_pos[t]
            if p != -1:
                y_train[p] = hr_y[k]
            else:
                q = test_pos[t]
                if q != -1:
                    y_test[q] = hr_y[k]
        global_t += T
        y_train.flush(); y_test.flush()

    # Final clamp for precipitation to match your invtrans_write behavior
    if var in ("pr", "prcp"):
        # In-place clamp
        x_train[x_train < 0] = 0
        x_test[x_test < 0] = 0
        y_train[y_train < 0] = 0
        y_test[y_test < 0] = 0

    x_train.flush(); x_test.flush()
    y_train.flush(); y_test.flush()

    print("[done] wrote x_train/x_test/y_train/y_test + scaler.pkl", flush=True)

    return {
        "x_train": os.path.join(path_output, "x_train.npy"),
        "x_test":  os.path.join(path_output, "x_test.npy"),
        "y_train": os.path.join(path_output, "y_train.npy"),
        "y_test":  os.path.join(path_output, "y_test.npy"),
        "scaler_pkl": os.path.join(checkpoint_dir, "scaler.pkl"),
    }
