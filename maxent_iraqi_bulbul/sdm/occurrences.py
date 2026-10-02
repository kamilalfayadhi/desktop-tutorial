"""Load, download (GBIF), clean and spatially thin occurrence records."""
from __future__ import annotations

import csv
import io
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

GBIF_API = "https://api.gbif.org/v1"

LON_NAMES = ["decimallongitude", "longitude", "long", "lon", "lng", "x", "xcoord", "east"]
LAT_NAMES = ["decimallatitude", "latitude", "lat", "y", "ycoord", "north"]


def _find_column(columns, candidates):
    lookup = {c.lower().strip().replace(" ", "").replace("_", ""): c for c in columns}
    for name in candidates:
        if name in lookup:
            return lookup[name]
    return None


def _read_resaved_gbif_tsv(path: Path) -> pd.DataFrame:
    """Recover a tab-separated GBIF export that a spreadsheet re-saved as CSV.

    Such files wrap each tab-separated line in quotes and split it wherever the text had a comma
    (e.g. "Pycnonotus leucotis (Gould"," 1836)"), padding rows with empty fields. Re-joining the
    comma-split pieces restores the original tab-separated line.
    """
    with open(path, encoding="utf-8-sig", newline="") as f:
        lines = [",".join(field for field in row if field != "") for row in csv.reader(f)]
    log.info("Recovered %s as a re-saved tab-separated (GBIF) file", path.name)
    return pd.read_csv(io.StringIO("\n".join(lines)), sep="\t", dtype=str)


def read_occurrence_file(path: Path) -> pd.DataFrame:
    """Read a CSV/TSV/XLSX file and return it with standard `lon`/`lat` columns."""
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, sep=None, engine="python", dtype=str)

    lon_col = _find_column(df.columns, LON_NAMES)
    lat_col = _find_column(df.columns, LAT_NAMES)
    if (lon_col is None or lat_col is None) and suffix not in (".xlsx", ".xls"):
        df = _read_resaved_gbif_tsv(path)
        lon_col = _find_column(df.columns, LON_NAMES)
        lat_col = _find_column(df.columns, LAT_NAMES)
    if lon_col is None or lat_col is None:
        raise ValueError(
            f"Could not find longitude/latitude columns in {path}. "
            f"Columns found: {list(df.columns)}. Rename them to 'longitude' and 'latitude'."
        )
    df = df.rename(columns={lon_col: "lon", lat_col: "lat"})
    log.info("Read %d records from %s (lon=%s, lat=%s)", len(df), path, lon_col, lat_col)
    return df


def _gbif_get(url, params, retries=8, pause=0.3):
    """GET a GBIF API endpoint politely: short pause per request, honour 429/Retry-After, back off on errors."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60,
                             headers={"User-Agent": "maxent-iraqi-bulbul (species distribution model)"})
            if r.status_code == 429 or r.status_code >= 500:
                wait = float(r.headers.get("Retry-After") or min(60, 5 * 2 ** attempt))
                raise requests.HTTPError(f"HTTP {r.status_code}, waiting {wait:.0f} s", response=r)
            r.raise_for_status()
            time.sleep(pause)
            return r.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            resp = getattr(e, "response", None)
            wait = (float(resp.headers.get("Retry-After") or min(60, 5 * 2 ** attempt))
                    if resp is not None and (resp.status_code == 429 or resp.status_code >= 500)
                    else min(60, 2 ** attempt))
            log.warning("GBIF request failed (%s); retrying in %.0f s", e, wait)
            time.sleep(wait)


def _gbif_record(r: dict) -> dict:
    return {
        "lon": r.get("decimalLongitude"),
        "lat": r.get("decimalLatitude"),
        "year": r.get("year"),
        "coordinateUncertaintyInMeters": r.get("coordinateUncertaintyInMeters"),
        "basisOfRecord": r.get("basisOfRecord"),
        "country": r.get("countryCode"),
        "gbifID": r.get("key"),
        "datasetKey": r.get("datasetKey"),
    }


def download_gbif(species: str, bbox, tile_deg: float = 1.0, max_per_tile: int = 600,
                  n_threads: int = 2) -> pd.DataFrame:
    """Download georeferenced occurrence records from the GBIF occurrence search API.

    The area is split into `tile_deg` x `tile_deg` tiles and at most `max_per_tile` records are taken from
    each. This keeps every request shallow (GBIF search gets very slow at deep page offsets) and stops
    dense birding hotspots from dominating the sample; records are thinned to ~10 km later anyway.
    """
    match = _gbif_get(f"{GBIF_API}/species/match", {"name": species})
    if "usageKey" not in match:
        raise RuntimeError(f"GBIF could not match the species name '{species}'")
    taxon_key = match.get("acceptedUsageKey", match["usageKey"])
    log.info("GBIF taxon: %s (key %s)", match.get("scientificName"), taxon_key)

    min_lon, min_lat, max_lon, max_lat = bbox
    lons = np.arange(min_lon, max_lon, tile_deg)
    lats = np.arange(min_lat, max_lat, tile_deg)
    tiles = [(x, y, min(x + tile_deg, max_lon), min(y + tile_deg, max_lat)) for x in lons for y in lats]

    def fetch_tile(tile):
        x0, y0, x1, y1 = tile
        params = {
            "taxonKey": taxon_key, "hasCoordinate": "true", "hasGeospatialIssue": "false",
            "occurrenceStatus": "PRESENT", "limit": 300,
            # half-open ranges so records on a tile edge are not fetched twice
            "decimalLongitude": f"{x0},{x1 - 1e-7 if x1 < max_lon else x1}",
            "decimalLatitude": f"{y0},{y1 - 1e-7 if y1 < max_lat else y1}",
        }
        rows, offset = [], 0
        while offset < max_per_tile:
            params["offset"] = offset
            page = _gbif_get(f"{GBIF_API}/occurrence/search", params)
            rows += [_gbif_record(r) for r in page.get("results", [])]
            if page.get("endOfRecords", True):
                break
            offset += params["limit"]
        return rows

    from concurrent.futures import ThreadPoolExecutor

    rows, done = [], 0
    with ThreadPoolExecutor(n_threads) as pool:
        for tile_rows in pool.map(fetch_tile, tiles):
            rows += tile_rows
            done += 1
            if done % 100 == 0:
                log.info("    ... %d of %d tiles, %d records", done, len(tiles), len(rows))
    df = pd.DataFrame(rows, columns=list(_gbif_record({}).keys()))
    df = df.drop_duplicates(subset="gbifID")
    log.info("Downloaded %d GBIF records from %d tiles", len(df), len(tiles))
    return df


def cached_gbif(cache_dir: Path, max_age_days: float, species: str, bbox, tile_deg=1.0, max_per_tile=600):
    """download_gbif, reusing a cached copy of the same query if it is younger than `max_age_days`."""
    key = "_".join([species.replace(" ", "-"), *(f"{b:g}" for b in bbox), f"{tile_deg:g}", str(max_per_tile)])
    path = cache_dir / f"gbif_{key}.csv"
    if max_age_days and path.exists() and (time.time() - path.stat().st_mtime) < max_age_days * 86400:
        log.info("Using cached GBIF download %s", path.name)
        return pd.read_csv(path, dtype={"gbifID": str})
    df = download_gbif(species, bbox, tile_deg, max_per_tile)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def clean(df: pd.DataFrame, bbox, min_year=None, max_uncertainty_m=None) -> tuple[pd.DataFrame, dict]:
    """Basic coordinate cleaning. Returns cleaned data and a step-by-step record count log."""
    steps = {"input": len(df)}
    df = df.copy()
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df = df.dropna(subset=["lon", "lat"])
    df = df[df["lon"].between(-180, 180) & df["lat"].between(-90, 90)]
    df = df[~((df["lon"] == 0) & (df["lat"] == 0))]
    steps["valid_coordinates"] = len(df)

    min_lon, min_lat, max_lon, max_lat = bbox
    df = df[df["lon"].between(min_lon, max_lon) & df["lat"].between(min_lat, max_lat)]
    steps["inside_study_area"] = len(df)

    year_col = _find_column(df.columns, ["year", "eventyear"])
    if min_year and year_col:
        years = pd.to_numeric(df[year_col], errors="coerce")
        df = df[years.isna() | (years >= min_year)]
        steps[f"year_>=_{min_year}"] = len(df)

    unc_col = _find_column(df.columns, ["coordinateuncertaintyinmeters", "uncertainty", "uncertaintym"])
    if max_uncertainty_m and unc_col:
        unc = pd.to_numeric(df[unc_col], errors="coerce")
        df = df[unc.isna() | (unc <= max_uncertainty_m)]
        steps[f"uncertainty_<=_{max_uncertainty_m}m"] = len(df)

    basis_col = _find_column(df.columns, ["basisofrecord"])
    if basis_col:
        df = df[~df[basis_col].astype(str).str.upper().isin(["FOSSIL_SPECIMEN", "LIVING_SPECIMEN"])]
        steps["drop_fossil_and_captive"] = len(df)

    df = df.drop_duplicates(subset=["lon", "lat"])
    steps["unique_coordinates"] = len(df)
    return df.reset_index(drop=True), steps


def thin(df: pd.DataFrame, distance_km: float, seed: int = 42) -> pd.DataFrame:
    """Greedy random spatial thinning: no two kept points closer than `distance_km`."""
    if not distance_km or len(df) < 2:
        return df
    from sklearn.neighbors import BallTree

    rng = np.random.default_rng(seed)
    coords = np.radians(df[["lat", "lon"]].to_numpy())
    neighbours = BallTree(coords, metric="haversine").query_radius(coords, r=distance_km / 6371.0)
    removed = np.zeros(len(df), dtype=bool)
    kept = []
    for i in rng.permutation(len(df)):
        if removed[i]:
            continue
        kept.append(i)
        removed[neighbours[i]] = True
    return df.iloc[sorted(kept)].reset_index(drop=True)
