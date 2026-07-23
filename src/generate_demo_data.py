"""
Generates small synthetic demo datasets standing in for real remote-sensing products.

NOTE: This sandbox has no network access to real geospatial data providers
(USGS, Copernicus, Google Earth Engine, PANGAEA, etc.), so this script builds
plausible synthetic rasters with the *same structure* as real products
(GeoTIFF, CRS, resolution). Swapping in real data later means only changing
the file paths in semantic_layer.yaml -- the rest of the pipeline (tools,
agent, API) works unchanged against any GeoTIFF that matches the schema.

Region: a fictional ~50km x 50km Arctic tile, loosely modelled on Herschel
Island / Beaufort Sea coastal terrain (elevation 0-120m, coastal lowland
with a ridge), to keep the demo grounded in the kind of terrain Adam's
research actually covers.
"""
import numpy as np
import rasterio
from rasterio.transform import from_origin
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)

WIDTH, HEIGHT = 200, 200          # 200x200 px
RES_M = 250                        # 250m pixel size -> ~50km x 50km tile
WEST, NORTH = -139.10, 69.60       # near Herschel Island, Yukon coast (approx)
CRS = "EPSG:4326"

transform = from_origin(WEST, NORTH, RES_M / 111320, RES_M / 111320)  # deg approx


def write_raster(path, array, dtype="float32", nodata=-9999.0):
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=array.shape[0], width=array.shape[1],
        count=1, dtype=dtype,
        crs=CRS, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(array.astype(dtype), 1)


def make_dem():
    """Coastal lowland rising to an inland ridge, plus noise."""
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    ridge = 90 * np.exp(-((x - 150) ** 2) / (2 * 40 ** 2))
    coastal_gradient = (x / WIDTH) * 60
    noise = np.random.default_rng(42).normal(0, 3, size=(HEIGHT, WIDTH))
    dem = ridge + coastal_gradient + noise
    dem = np.clip(dem, 0, None)
    return dem


def make_permafrost_probability(dem):
    """Higher permafrost probability at low elevation coastal ice-rich terrain
    and moderate elevation, tapering at the exposed ridge crest (proxy pattern)."""
    base = 0.9 - 0.003 * dem
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 0.05, size=dem.shape)
    prob = np.clip(base + noise, 0, 1)
    return prob


def make_ground_ice_content(dem, permafrost_prob):
    """Massive ground ice content proxy (%), correlated with permafrost probability
    in low-lying coastal terrain (loosely evoking Herschel Island massive ice bodies)."""
    rng = np.random.default_rng(13)
    base = permafrost_prob * 70 * np.exp(-dem / 80)
    noise = rng.normal(0, 4, size=dem.shape)
    ice = np.clip(base + noise, 0, 90)
    return ice


if __name__ == "__main__":
    dem = make_dem()
    permafrost = make_permafrost_probability(dem)
    ice = make_ground_ice_content(dem, permafrost)

    write_raster(os.path.join(OUT_DIR, "dem.tif"), dem)
    write_raster(os.path.join(OUT_DIR, "permafrost_probability.tif"), permafrost)
    write_raster(os.path.join(OUT_DIR, "ground_ice_content.tif"), ice)

    print("Demo rasters written to", OUT_DIR)
    print(f"DEM range: {dem.min():.1f} - {dem.max():.1f} m")
    print(f"Permafrost probability range: {permafrost.min():.2f} - {permafrost.max():.2f}")
    print(f"Ground ice content range: {ice.min():.1f} - {ice.max():.1f} %")
