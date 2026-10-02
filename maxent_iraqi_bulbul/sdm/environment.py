"""Get environmental (bioclimatic) layers, crop them to the study area and select variables."""
from __future__ import annotations

import logging
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds

log = logging.getLogger(__name__)

WORLDCLIM_BASE = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_{res}_bio.zip"
WORLDCLIM_FUTURE = "https://geodata.ucdavis.edu/cmip6/{res}/{gcm}/ssp{ssp}/wc2.1_{res}_bioc_{gcm}_ssp{ssp}_{period}.tif"

BIOCLIM_NAMES = {
    "bio_1": "Annual mean temperature", "bio_2": "Mean diurnal range",
    "bio_3": "Isothermality", "bio_4": "Temperature seasonality",
    "bio_5": "Max temperature of warmest month", "bio_6": "Min temperature of coldest month",
    "bio_7": "Temperature annual range", "bio_8": "Mean temperature of wettest quarter",
    "bio_9": "Mean temperature of driest quarter", "bio_10": "Mean temperature of warmest quarter",
    "bio_11": "Mean temperature of coldest quarter", "bio_12": "Annual precipitation",
    "bio_13": "Precipitation of wettest month", "bio_14": "Precipitation of driest month",
    "bio_15": "Precipitation seasonality", "bio_16": "Precipitation of wettest quarter",
    "bio_17": "Precipitation of driest quarter", "bio_18": "Precipitation of warmest quarter",
    "bio_19": "Precipitation of coldest quarter",
}


def describe(var: str) -> str:
    return BIOCLIM_NAMES.get(var, var)


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        log.info("Using cached %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s (this can take a while)...", url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.rename(dest)
    return dest


def _var_name(path: str) -> str:
    """'wc2.1_2.5m_bio_12.tif' -> 'bio_12'; any other file -> its stem."""
    m = re.search(r"bio_?(\d+)", Path(path).stem)
    return f"bio_{int(m.group(1))}" if m else Path(path).stem


def _sort_key(name: str):
    m = re.match(r"bio_(\d+)$", name)
    return (0, int(m.group(1)), "") if m else (1, 0, name)


def _crop(src, bbox, template=None):
    """Read a raster cropped to bbox; if a template profile is given, warp onto its grid."""
    if template is not None:
        with WarpedVRT(src, crs=template["crs"], transform=template["transform"],
                       width=template["width"], height=template["height"]) as vrt:
            data = vrt.read(1, masked=True)
        return data.astype("float32").filled(np.nan), template
    window = from_bounds(*bbox, transform=src.transform).round_offsets().round_lengths()
    data = src.read(1, window=window, masked=True, boundless=True)
    profile = {
        "crs": src.crs, "transform": src.window_transform(window),
        "width": data.shape[1], "height": data.shape[0],
    }
    return data.astype("float32").filled(np.nan), profile


def load_layers(cfg: dict, cache_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Return {variable: 2-D array} cropped to the study area and the shared grid profile."""
    env_cfg, bbox = cfg["environment"], cfg["study_area"]["bbox"]
    if env_cfg["source"] == "local":
        folder = Path(env_cfg["local_dir"])
        paths = sorted(str(p) for p in folder.glob("*.tif"))
        if not paths:
            raise FileNotFoundError(f"No .tif files found in {folder}")
    else:
        res = env_cfg["resolution"]
        zip_path = _download(WORLDCLIM_BASE.format(res=res), cache_dir / f"wc2.1_{res}_bio.zip")
        with zipfile.ZipFile(zip_path) as z:
            paths = [f"/vsizip/{zip_path}/{n}" for n in z.namelist() if n.endswith(".tif")]

    wanted = env_cfg.get("variables", "all")
    layers, template = {}, None
    for path in sorted(paths, key=lambda p: _sort_key(_var_name(p))):
        name = _var_name(path)
        if wanted != "all" and name not in wanted:
            continue
        with rasterio.open(path) as src:
            layers[name], template = _crop(src, bbox, template)
    if not layers:
        raise RuntimeError("No environmental layers were loaded - check `environment.variables`.")
    log.info("Loaded %d layers on a %dx%d grid", len(layers), template["height"], template["width"])
    return layers, template


def future_scenarios(fut_cfg: dict) -> list[dict]:
    """All GCM x SSP x period combinations from the `future` config (scalars or lists accepted)."""
    def as_list(*keys):
        for k in keys:
            if fut_cfg.get(k) is not None:
                v = fut_cfg[k]
                return [str(x) for x in (v if isinstance(v, list) else [v])]
        return []
    return [{"gcm": g, "ssp": s, "period": p}
            for g in as_list("gcms", "gcm") for s in as_list("ssps", "ssp") for p in as_list("periods", "period")]


def load_future_layers(cfg: dict, cache_dir: Path, template: dict, variables: list[str],
                       scenario: dict) -> dict[str, np.ndarray]:
    """Download a WorldClim CMIP6 multi-band bioclim GeoTIFF and warp it onto the current grid."""
    res = cfg["environment"]["resolution"]
    url = WORLDCLIM_FUTURE.format(res=res, **scenario)
    path = _download(url, cache_dir / Path(url).name)
    out = {}
    with rasterio.open(path) as src, WarpedVRT(src, crs=template["crs"], transform=template["transform"],
                                                width=template["width"], height=template["height"]) as vrt:
        for var in variables:
            band = int(var.split("_")[1])
            out[var] = vrt.read(band, masked=True).astype("float32").filled(np.nan)
    return out


def write_raster(path: Path, data: np.ndarray, template: dict, nodata=-9999.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.where(np.isnan(data), nodata, data).astype("float32")
    with rasterio.open(path, "w", driver="GTiff", height=template["height"], width=template["width"],
                       count=1, dtype="float32", crs=template["crs"], transform=template["transform"],
                       nodata=nodata, compress="deflate") as dst:
        dst.write(arr, 1)


def extract(layers: dict[str, np.ndarray], template: dict, lon, lat) -> pd.DataFrame:
    """Sample every layer at the given coordinates (NaN outside the grid)."""
    rows, cols = rasterio.transform.rowcol(template["transform"], np.asarray(lon), np.asarray(lat))
    rows, cols = np.asarray(rows), np.asarray(cols)
    inside = (rows >= 0) & (rows < template["height"]) & (cols >= 0) & (cols < template["width"])
    out = {}
    for name, arr in layers.items():
        vals = np.full(len(rows), np.nan, dtype="float32")
        vals[inside] = arr[rows[inside], cols[inside]]
        out[name] = vals
    return pd.DataFrame(out)


def vif(df: pd.DataFrame) -> pd.Series:
    """Variance inflation factor of each column."""
    x = (df - df.mean()) / df.std(ddof=0)
    corr = np.corrcoef(x.to_numpy(), rowvar=False)
    inv = np.linalg.pinv(corr)
    return pd.Series(np.diag(inv), index=df.columns)


def select_variables(bg_env: pd.DataFrame, vif_threshold: float, corr_threshold: float,
                     protected=()) -> tuple[list[str], list[dict]]:
    """Stepwise VIF removal followed by pairwise-correlation removal. Returns kept vars and a removal log.

    Variables in `protected` are never removed (other variables are dropped around them).
    """
    protected = set(protected or ())
    keep = [c for c in bg_env.columns if bg_env[c].std() > 0]
    removed = [{"variable": c, "reason": "constant in study area"} for c in bg_env.columns if c not in keep]
    while len(keep) > 2:
        v = vif(bg_env[keep]).drop(labels=list(protected), errors="ignore")
        if v.empty:
            break
        worst = v.idxmax()
        if v[worst] < vif_threshold:
            break
        removed.append({"variable": worst, "reason": f"VIF = {v[worst]:.1f}"})
        keep.remove(worst)
    while len(keep) > 2:
        corr = bg_env[keep].corr().abs().to_numpy().copy()
        np.fill_diagonal(corr, 0)
        pair = corr.copy()
        for k, name in enumerate(keep):  # pairs of two protected variables are allowed
            if name in protected:
                pair[k, [m for m, other in enumerate(keep) if other in protected]] = 0
        i, j = np.unravel_index(pair.argmax(), pair.shape)
        if pair[i, j] <= corr_threshold:
            break
        # drop the unprotected member; if both are unprotected, the one more correlated with the rest
        a, b = keep[i], keep[j]
        if a in protected or b in protected:
            drop = b if a in protected else a
        else:
            drop = a if corr[i].mean() >= corr[j].mean() else b
        removed.append({"variable": drop, "reason": f"|r| = {corr[i, j]:.2f} with {b if drop == a else a}"})
        keep.remove(drop)
    return keep, removed
