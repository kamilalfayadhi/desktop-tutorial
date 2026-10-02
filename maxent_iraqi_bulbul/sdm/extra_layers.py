"""Non-climate predictors: distance to rivers and land-cover fractions, on the model grid.

* Distance to rivers: Natural Earth 1:10m river centre-lines (major rivers such as the Tigris,
  Euphrates, Karun and their main tributaries), as great-circle distance in km from each cell centre.
* Land cover: ESA WorldCover 2021 (10 m). Each 3x3 degree tile is read at a reduced resolution
  (~150 m, from the file's built-in overviews) and turned into the fraction (0-1) of each model cell
  covered by the chosen classes.

Both are static: the same layers are used for current and future projections.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from sklearn.neighbors import BallTree

log = logging.getLogger(__name__)

RIVERS_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
              "ne_10m_rivers_lake_centerlines.geojson")
WORLDCOVER_URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
                  "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")
WORLDCOVER_CLASSES = {
    "tree": 10, "shrub": 20, "grass": 30, "cropland": 40, "built": 50,
    "bare": 60, "water": 80, "wetland": 90, "mangrove": 95,
}
DESCRIPTIONS = {
    "dist_river_km": "Distance to nearest major river (km, capped)",
    **{f"lc_{k}": f"Land cover: {k} fraction (ESA WorldCover)" for k in WORLDCOVER_CLASSES},
}


def _cache_name(prefix: str, template: dict) -> str:
    b = rasterio.transform.array_bounds(template["height"], template["width"], template["transform"])
    return f"{prefix}_{template['width']}x{template['height']}_" + "_".join(f"{v:.4f}" for v in b) + ".tif"


def _read_cached(path: Path, count: int):
    if path.exists():
        with rasterio.open(path) as src:
            if src.count == count:
                log.info("Using cached %s", path.name)
                return src.read()
    return None


def _write_cached(path: Path, data: np.ndarray, template: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=template["height"], width=template["width"],
                       count=data.shape[0], dtype="float32", crs=template["crs"],
                       transform=template["transform"], nodata=np.nan, compress="deflate") as dst:
        dst.write(data.astype("float32"))


def river_distance(template: dict, cache_dir: Path) -> np.ndarray:
    """Great-circle distance (km) from every cell centre to the nearest Natural Earth river line."""
    cache = cache_dir / _cache_name("dist_river", template)
    cached = _read_cached(cache, 1)
    if cached is not None:
        return cached[0]

    log.info("Downloading Natural Earth river centre-lines")
    gj = requests.get(RIVERS_URL, timeout=300).json()
    b = rasterio.transform.array_bounds(template["height"], template["width"], template["transform"])
    pad = 5.0  # include rivers just outside the grid so distances near the edge are right
    shapes = []
    for f in gj["features"]:
        geom = f.get("geometry")
        if not geom:
            continue
        coords = geom["coordinates"] if geom["type"] == "MultiLineString" else [geom["coordinates"]]
        xs = [p[0] for line in coords for p in line]
        ys = [p[1] for line in coords for p in line]
        if not xs:
            continue
        if max(xs) < b[0] - pad or min(xs) > b[2] + pad or max(ys) < b[1] - pad or min(ys) > b[3] + pad:
            continue
        shapes.append(geom)
    if not shapes:
        raise RuntimeError("No rivers found in or near the model area")

    # Rasterize the lines on a 4x finer grid padded around the model grid, then measure distances
    # from the model cell centres to the centres of river cells.
    fine = 4
    t = template["transform"]
    res = abs(t.a) / fine
    width = int(math.ceil((b[2] - b[0] + 2 * pad) / res))
    height = int(math.ceil((b[3] - b[1] + 2 * pad) / res))
    ftransform = rasterio.transform.from_origin(b[0] - pad, b[3] + pad, res, res)
    mask = rasterize(((s, 1) for s in shapes), out_shape=(height, width), transform=ftransform,
                     all_touched=True, dtype="uint8")
    rr, cc = np.nonzero(mask)
    river_lon = ftransform.c + (cc + 0.5) * res
    river_lat = ftransform.f - (rr + 0.5) * res
    tree = BallTree(np.radians(np.c_[river_lat, river_lon]), metric="haversine")

    rows, cols = np.mgrid[0:template["height"], 0:template["width"]]
    lon = t.c + (cols.ravel() + 0.5) * t.a
    lat = t.f + (rows.ravel() + 0.5) * t.e
    dist, _ = tree.query(np.radians(np.c_[lat, lon]), k=1)
    out = (dist[:, 0] * 6371.0).reshape(template["height"], template["width"]).astype("float32")
    _write_cached(cache, out[None], template)
    log.info("River distance: %d river line(s) used", len(shapes))
    return out


def _worldcover_tiles(bounds):
    """Names of the 3x3 degree WorldCover tiles covering `bounds` (lower-left corner naming)."""
    x0, y0, x1, y1 = bounds
    for lat in range(int(math.floor(y0 / 3) * 3), int(math.ceil(y1 / 3) * 3), 3):
        for lon in range(int(math.floor(x0 / 3) * 3), int(math.ceil(x1 / 3) * 3), 3):
            ns = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
            ew = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
            yield f"{ns}{ew}"


def landcover_fractions(template: dict, classes: list[str], cache_dir: Path,
                        overview_level: int = 3) -> dict[str, np.ndarray]:
    """Fraction (0-1) of each model cell covered by each WorldCover class in `classes`."""
    unknown = [c for c in classes if c not in WORLDCOVER_CLASSES]
    if unknown:
        raise ValueError(f"Unknown land-cover classes {unknown}; choose from {list(WORLDCOVER_CLASSES)}")
    cache = cache_dir / _cache_name("worldcover_" + "-".join(classes), template)
    cached = _read_cached(cache, len(classes))
    if cached is not None:
        return {f"lc_{c}": cached[i] for i, c in enumerate(classes)}

    shape = (template["height"], template["width"])
    total = {c: np.zeros(shape, "float64") for c in classes}
    weight = np.zeros(shape, "float64")
    bounds = rasterio.transform.array_bounds(template["height"], template["width"], template["transform"])
    tiles = list(_worldcover_tiles(bounds))
    log.info("Reading %d ESA WorldCover tiles (at ~%d m)", len(tiles), 10 * 2 ** (overview_level + 1))
    n_read = 0
    for i, tile in enumerate(tiles, 1):
        url = "/vsicurl/" + WORLDCOVER_URL.format(tile=tile)
        try:
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="4",
                              GDAL_HTTP_RETRY_DELAY="5"), \
                    rasterio.open(url, overview_level=overview_level) as src:
                lc = src.read(1)
                src_transform, src_crs = src.transform, src.crs
        except rasterio.errors.RasterioIOError:
            continue  # no tile here (open sea)
        n_read += 1
        valid = (lc != 0).astype("float32")
        layers = [valid] + [(lc == WORLDCOVER_CLASSES[c]).astype("float32") for c in classes]
        for k, arr in enumerate(layers):
            dst = np.full(shape, np.nan, "float32")
            reproject(arr, dst, src_transform=src_transform, src_crs=src_crs, dst_transform=template["transform"],
                      dst_crs=template["crs"], resampling=Resampling.average, src_nodata=None, dst_nodata=np.nan)
            ok = ~np.isnan(dst)
            if k == 0:
                weight[ok] += dst[ok]
            else:
                total[classes[k - 1]][ok] += dst[ok]
        if i % 10 == 0:
            log.info("    ... %d of %d tiles", i, len(tiles))
    log.info("Read %d WorldCover tiles", n_read)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {c: np.where(weight > 0, total[c] / weight, np.nan).astype("float32") for c in classes}
    _write_cached(cache, np.stack([out[c] for c in classes]), template)
    return {f"lc_{c}": v for c, v in out.items()}


def load(cfg: dict, template: dict, cache_dir: Path) -> dict[str, np.ndarray]:
    """All configured extra layers on the model grid."""
    extra_cfg = cfg["environment"].get("extra_layers") or {}
    out = {}
    if extra_cfg.get("river_distance"):
        dist = river_distance(template, cache_dir)
        cap = extra_cfg.get("river_distance_cap_km")
        # Beyond a few tens of km a river has no local effect; an uncapped distance mostly encodes region
        # (e.g. the river-less Gulf coast), so it is capped.
        out["dist_river_km"] = np.minimum(dist, cap) if cap else dist
    if extra_cfg.get("landcover"):
        out.update(landcover_fractions(template, list(extra_cfg["landcover"]), cache_dir,
                                       extra_cfg.get("landcover_overview_level", 3)))
    return out
