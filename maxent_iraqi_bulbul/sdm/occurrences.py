"""Load, download (GBIF), clean and spatially thin occurrence records."""
from __future__ import annotations

import logging
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


def read_occurrence_file(path: Path) -> pd.DataFrame:
    """Read a CSV/TSV/XLSX file and return it with standard `lon`/`lat` columns."""
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, sep=None, engine="python")

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


def download_gbif(species: str, bbox, max_records: int) -> pd.DataFrame:
    """Download georeferenced occurrence records from the GBIF occurrence search API."""
    match = requests.get(f"{GBIF_API}/species/match", params={"name": species}, timeout=60).json()
    if "usageKey" not in match:
        raise RuntimeError(f"GBIF could not match the species name '{species}'")
    taxon_key = match.get("acceptedUsageKey", match["usageKey"])
    log.info("GBIF taxon: %s (key %s)", match.get("scientificName"), taxon_key)

    min_lon, min_lat, max_lon, max_lat = bbox
    params = {
        "taxonKey": taxon_key,
        "hasCoordinate": "true",
        "hasGeospatialIssue": "false",
        "occurrenceStatus": "PRESENT",
        "decimalLongitude": f"{min_lon},{max_lon}",
        "decimalLatitude": f"{min_lat},{max_lat}",
        "limit": 300,
    }
    rows, offset = [], 0
    while offset < max_records:
        params["offset"] = offset
        page = requests.get(f"{GBIF_API}/occurrence/search", params=params, timeout=120).json()
        for r in page.get("results", []):
            rows.append({
                "lon": r.get("decimalLongitude"),
                "lat": r.get("decimalLatitude"),
                "year": r.get("year"),
                "coordinateUncertaintyInMeters": r.get("coordinateUncertaintyInMeters"),
                "basisOfRecord": r.get("basisOfRecord"),
                "country": r.get("countryCode"),
                "gbifID": r.get("key"),
                "datasetKey": r.get("datasetKey"),
            })
        if page.get("endOfRecords", True):
            break
        offset += params["limit"]
    log.info("Downloaded %d GBIF records", len(rows))
    return pd.DataFrame(rows)


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


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def thin(df: pd.DataFrame, distance_km: float, seed: int = 42) -> pd.DataFrame:
    """Greedy random spatial thinning: no two kept points closer than `distance_km`."""
    if not distance_km or len(df) < 2:
        return df
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    lon, lat = df["lon"].to_numpy(), df["lat"].to_numpy()
    kept: list[int] = []
    for i in order:
        if kept:
            d = haversine_km(lon[i], lat[i], lon[kept], lat[kept])
            if d.min() < distance_km:
                continue
        kept.append(i)
    return df.iloc[sorted(kept)].reset_index(drop=True)
