"""Build a SYNTHETIC test dataset (fake climate rasters + virtual-species occurrences).

Used only to check that the pipeline runs end-to-end and recovers a known niche when real
data cannot be downloaded. The results are NOT about the real Iraqi bulbul.

    python tests/make_synthetic_data.py tests/synthetic
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

BBOX = (37.5, 28.5, 49.5, 38.0)
RES = 2.5 / 60  # 2.5 arc-minutes


def main(out: Path, seed: int = 1):
    rng = np.random.default_rng(seed)
    w = round((BBOX[2] - BBOX[0]) / RES)
    h = round((BBOX[3] - BBOX[1]) / RES)
    lon = BBOX[0] + (np.arange(w) + 0.5) * RES
    lat = BBOX[3] - (np.arange(h) + 0.5) * RES
    LON, LAT = np.meshgrid(lon, lat)

    # smooth "terrain": mountains in the north-east (Zagros/Kurdistan)
    elev = 2500 * np.exp(-(((LON - 45.5) / 2.2) ** 2 + ((LAT - 36.8) / 1.3) ** 2)) + 300 * (LAT - 28.5) / 9.5
    noise = lambda s: s * rng.standard_normal((h, w)).cumsum(0).cumsum(1) / (h * w) ** 0.5  # noqa: E731
    layers = {
        "bio_1": 30 - 0.9 * (LAT - 28.5) - elev / 160 + noise(0.3),          # annual mean temp (°C)
        "bio_4": 700 + 40 * (LON - 37.5) + 20 * (LAT - 28.5) + noise(10),     # temp seasonality
        "bio_5": 47 - 0.8 * (LAT - 28.5) - elev / 150 + noise(0.3),           # max temp warmest month
        "bio_6": 6 - 1.0 * (LAT - 28.5) - elev / 170 + noise(0.3),            # min temp coldest month
        "bio_12": 60 + 45 * (LAT - 28.5) + elev * 0.25 + noise(8),            # annual precipitation (mm)
        "bio_15": 110 - 3 * (LAT - 28.5) + 2 * (LON - 37.5) + noise(3),       # precip seasonality
    }
    # sea / no-data in the far south-east corner (Persian Gulf stand-in)
    sea = (LON > 48.3) & (LAT < 29.8)
    out.joinpath("env").mkdir(parents=True, exist_ok=True)
    transform = from_origin(BBOX[0], BBOX[3], RES, RES)
    for name, arr in layers.items():
        arr = np.where(sea, -9999, arr).astype("float32")
        with rasterio.open(out / "env" / f"wc2.1_2.5m_{name}.tif", "w", driver="GTiff", height=h, width=w,
                           count=1, dtype="float32", crs="EPSG:4326", transform=transform, nodata=-9999) as dst:
            dst.write(arr, 1)

    # virtual species: warm lowlands with moderate rainfall (river plains), avoids mountains
    t, p = layers["bio_1"], layers["bio_12"]
    suit = np.exp(-((t - 22.5) / 3.0) ** 2) * np.exp(-((p - 250) / 160) ** 2)
    suit[sea] = 0
    probs = suit.ravel() / suit.sum()
    idx = rng.choice(suit.size, size=400, p=probs)
    r, c = np.unravel_index(idx, suit.shape)
    occ = pd.DataFrame({
        "species": "virtual species (synthetic)",
        "longitude": lon[c] + rng.uniform(-RES / 2, RES / 2, len(c)),
        "latitude": lat[r] + rng.uniform(-RES / 2, RES / 2, len(r)),
        "year": rng.integers(1995, 2024, len(c)),
    })
    occ.to_csv(out / "occurrences.csv", index=False)
    with rasterio.open(out / "true_suitability.tif", "w", driver="GTiff", height=h, width=w, count=1,
                       dtype="float32", crs="EPSG:4326", transform=transform) as dst:
        dst.write(suit.astype("float32"), 1)
    print(f"Synthetic data written to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "tests/synthetic"))
