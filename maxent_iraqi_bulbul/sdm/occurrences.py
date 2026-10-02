"""Load, download (GBIF), clean and spatially thin occurrence records."""
from __future__ import annotations

import csv
import io
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
        if offset % 6000 == 0:
            log.info("    ... %d of %s GBIF records", len(rows), page.get("count", "?"))
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
