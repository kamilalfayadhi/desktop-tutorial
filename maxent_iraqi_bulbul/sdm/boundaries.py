"""Country and province outlines (Natural Earth 1:10m) for maps, and a mask to clip results to a country."""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import requests
from rasterio.features import rasterize
from shapely.geometry import box

log = logging.getLogger(__name__)

NE_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/{name}.geojson"


def _natural_earth(name: str, cache_dir: Path) -> gpd.GeoDataFrame:
    path = cache_dir / f"{name}.geojson"
    if not path.exists():
        log.info("Downloading Natural Earth %s", name)
        r = requests.get(NE_URL.format(name=name), timeout=300)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    return gpd.read_file(path)


def load(cfg: dict, cache_dir: Path, view_bbox) -> dict[str, gpd.GeoDataFrame]:
    """Outlines for the maps: neighbouring 'countries', the target 'country' and its 'provinces'.

    Layers are cut to `view_bbox` (plus a margin) so plotting stays light. Missing pieces are omitted.
    """
    sa = cfg["study_area"]
    out: dict[str, gpd.GeoDataFrame] = {}
    x0, y0, x1, y1 = view_bbox
    view = box(x0 - 2, y0 - 2, x1 + 2, y1 + 2)
    if sa.get("country") or sa.get("show_country_borders", True):
        countries = _natural_earth("ne_10m_admin_0_countries", cache_dir).to_crs("EPSG:4326")
        out["countries"] = countries[countries.intersects(view)].clip(view)
        if sa.get("country"):
            iso = sa["country"].upper()
            match = countries[(countries["ADM0_A3"] == iso) | (countries["ISO_A3"] == iso)]
            if match.empty:
                raise ValueError(f"Country code {iso!r} not found in Natural Earth (use ISO3, e.g. IRQ)")
            out["country"] = match
            if sa.get("show_provinces", True):
                prov = _natural_earth("ne_10m_admin_1_states_provinces", cache_dir).to_crs("EPSG:4326")
                out["provinces"] = prov[prov["adm0_a3"] == match["ADM0_A3"].iloc[0]]
    if sa.get("boundaries_file"):
        out["custom"] = gpd.read_file(sa["boundaries_file"]).to_crs("EPSG:4326")
    return out


def country_mask(template: dict, country: gpd.GeoDataFrame) -> np.ndarray:
    """True for grid cells whose centre lies inside the country."""
    return rasterize(((g, 1) for g in country.geometry), out_shape=(template["height"], template["width"]),
                     transform=template["transform"], dtype="uint8").astype(bool)


def provinces(cache_dir: Path, country_iso: str, names: list[str]) -> gpd.GeoDataFrame:
    """Natural Earth admin-1 polygons for the named provinces of a country (names as in Natural Earth)."""
    prov = _natural_earth("ne_10m_admin_1_states_provinces", cache_dir).to_crs("EPSG:4326")
    prov = prov[prov["adm0_a3"] == country_iso.upper()]
    missing = sorted(set(names) - set(prov["name"]))
    if missing:
        raise ValueError(f"Unknown province name(s) {missing}; available: {sorted(prov['name'])}")
    return prov[prov["name"].isin(names)]
