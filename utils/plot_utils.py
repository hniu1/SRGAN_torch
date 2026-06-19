"""
Shared plotting utilities for SRGAN downscaling diagnostics.
Mirrors the style used in downscaled_analysis/code/utils_plot.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ---------------------------------------------------------------------------
# Core spatial subplot
# ---------------------------------------------------------------------------

def plot_subplot(ax, data, lat, lon, title="", var="pr", stat="mean",
                 bias=False, vmin=None, vmax=None):
    """
    Plot a 2-D field on *ax* using cartopy PlateCarree.

    Parameters
    ----------
    ax   : GeoAxes created with projection=ccrs.PlateCarree() or similar.
    data : 2-D array (lat x lon).
    lat  : 1-D latitude array (degrees north).
    lon  : 1-D longitude array (degrees east, -180..180).
    title: subplot title string.
    var  : variable name – used to pick colour map and default ranges.
           Recognised values: 'pr', 'prcp', 'tmax', 'tmin'.
    stat : statistic label – 'mean', '95th', '5th' (used for default ranges).
    bias : if True, use a diverging colour map centred on zero.
    vmin / vmax : override the automatic colour range.
    """
    # ---- colour map and default range ----
    if not bias:
        cmap = "Spectral_r" if var in ("tmax", "tmin", "tasmax", "tasmin") else "Spectral"
        if var in ("pr", "prcp"):
            defaults = {"mean": (0, 5), "95th": (0, 25), "5th": (0, 1)}
        elif var in ("tmax", "tasmax"):
            defaults = {"mean": (0, 30), "95th": (0, 40), "5th": (-10, 10)}
        else:  # tmin / tasmin
            defaults = {"mean": (-5, 15), "95th": (0, 50), "5th": (-25, 10)}
    else:
        cmap = "RdBu_r" if var in ("tmax", "tmin", "tasmax", "tasmin") else "RdBu"
        if var in ("pr", "prcp"):
            defaults = {"mean": (-1, 1), "95th": (-5, 5), "5th": (-0.1, 0.1)}
        elif var in ("tmax", "tasmax"):
            defaults = {"mean": (-2, 2), "95th": (-5, 5), "5th": (-1, 1)}
        else:
            defaults = {"mean": (-2, 2), "95th": (-2, 2), "5th": (-5, 5)}

    _vmin, _vmax = defaults.get(stat, (None, None))
    if vmin is not None:
        _vmin = vmin
    if vmax is not None:
        _vmax = vmax

    im = ax.pcolormesh(
        lon, lat, data,
        rasterized=True,
        vmin=_vmin, vmax=_vmax,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=0.6)
    ax.add_feature(cfeature.STATES, edgecolor="black", linewidth=0.3)
    if title:
        ax.set_title(title, fontsize=11)

    return im  # caller can use for colorbar


# ---------------------------------------------------------------------------
# Histogram helper
# ---------------------------------------------------------------------------

def plot_histogram(ax, data, var="pr", label="", bins=80, log_scale=True):
    """
    Plot a histogram of *data* (flattened, NaN-safe) on *ax*.

    Parameters
    ----------
    log_scale : if True set y-axis to log scale (useful for precipitation).
    """
    flat = data[~np.isnan(data)].ravel()
    if var in ("pr", "prcp") and log_scale:
        flat = flat[flat > 0]  # exclude exact zeros for log-scale readability

    ax.hist(flat, bins=bins, edgecolor="none", alpha=0.75, label=label)
    ax.set_xlabel("Value (mm/day)" if var in ("pr", "prcp") else "Value (°C)")
    ax.set_ylabel("Count")
    if log_scale and var in ("pr", "prcp"):
        ax.set_yscale("log")
    if label:
        ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Convenience: print distribution summary to stdout
# ---------------------------------------------------------------------------

def print_stats(data, label="data", var="pr"):
    flat = data[~np.isnan(data)].ravel()
    print(f"\n=== {label} ===")
    print(f"  shape      : {data.shape}")
    print(f"  dtype      : {data.dtype}")
    print(f"  min        : {flat.min():.4f}")
    print(f"  max        : {flat.max():.4f}")
    print(f"  mean       : {flat.mean():.4f}")
    print(f"  std        : {flat.std():.4f}")
    pcts = np.percentile(flat, [1, 5, 25, 50, 75, 95, 99])
    labels = ["p1", "p5", "p25", "p50", "p75", "p95", "p99"]
    for lbl, v in zip(labels, pcts):
        print(f"  {lbl:<6}     : {v:.4f}")
    if var in ("pr", "prcp"):
        wet = np.mean(flat > 1.0) * 100
        dry = np.mean(flat == 0.0) * 100
        print(f"  wet days>1 : {wet:.1f}%")
        print(f"  exact zero : {dry:.1f}%")
