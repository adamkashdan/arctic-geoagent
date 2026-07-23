"""
Geospatial tool functions. These are the functions the LLM agent can call
via Claude's tool-use (function-calling). Each function reads a raster
listed in semantic_layer.yaml and returns a small, LLM-friendly result
(numbers/strings/base64 image) -- never raw pixel arrays.
"""
from __future__ import annotations
from typing import Optional
import os
import io
import base64
import yaml
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import shape, box
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")

with open(os.path.join(BASE_DIR, "semantic_layer.yaml")) as f:
    SEMANTIC_LAYER = yaml.safe_load(f)


def _dataset_path(dataset: str) -> str:
    if dataset not in SEMANTIC_LAYER["datasets"]:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Available: {list(SEMANTIC_LAYER['datasets'])}"
        )
    return os.path.join(BASE_DIR, SEMANTIC_LAYER["datasets"][dataset]["path"])


def list_datasets() -> dict:
    """Returns the semantic layer's dataset catalogue -- what the agent
    consults to know what data exists before calling any other tool."""
    return {
        name: {
            "label": d["label"],
            "unit": d["unit"],
            "description": d["description"].strip(),
        }
        for name, d in SEMANTIC_LAYER["datasets"].items()
    }


def query_raster_at_point(dataset: str, lat: float, lon: float) -> dict:
    """Returns the raster value at a single lat/lon point."""
    path = _dataset_path(dataset)
    with rasterio.open(path) as src:
        row, col = src.index(lon, lat)
        if row < 0 or col < 0 or row >= src.height or col >= src.width:
            return {"error": "Point is outside the dataset extent."}
        value = src.read(1)[row, col]
        nodata = src.nodata
        if nodata is not None and value == nodata:
            return {"error": "No data at this location."}
    unit = SEMANTIC_LAYER["datasets"][dataset]["unit"]
    return {"dataset": dataset, "lat": lat, "lon": lon, "value": float(value), "unit": unit}


def compute_zonal_statistics(dataset: str, bbox: list) -> dict:
    """Computes min/max/mean/std within a bounding box [west, south, east, north]."""
    path = _dataset_path(dataset)
    west, south, east, north = bbox
    geom = box(west, south, east, north)
    with rasterio.open(path) as src:
        try:
            out_image, _ = rio_mask(src, [geom], crop=True)
        except ValueError:
            return {"error": "Bounding box does not overlap the dataset."}
        arr = out_image[0]
        nodata = src.nodata
        if nodata is not None:
            arr = arr[arr != nodata]
        if arr.size == 0:
            return {"error": "No valid data in this bounding box."}
    unit = SEMANTIC_LAYER["datasets"][dataset]["unit"]
    return {
        "dataset": dataset,
        "bbox": bbox,
        "unit": unit,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "pixel_count": int(arr.size),
    }


def generate_map_image(dataset: str, bbox: Optional[list] = None) -> dict:
    """Renders the dataset (optionally cropped to bbox) as a PNG map,
    returned as a base64 string the API layer can serve back to a client."""
    path = _dataset_path(dataset)
    with rasterio.open(path) as src:
        if bbox:
            west, south, east, north = bbox
            geom = box(west, south, east, north)
            try:
                out_image, _ = rio_mask(src, [geom], crop=True)
            except ValueError:
                return {"error": "Bounding box does not overlap the dataset."}
            arr = out_image[0]
        else:
            arr = src.read(1)
        nodata = src.nodata
        arr_masked = np.ma.masked_equal(arr, nodata) if nodata is not None else arr

    label = SEMANTIC_LAYER["datasets"][dataset]["label"]
    unit = SEMANTIC_LAYER["datasets"][dataset]["unit"]

    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(arr_masked, cmap="viridis")
    ax.set_title(f"{label} ({unit})", fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.75)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return {"dataset": dataset, "image_base64": b64, "format": "png"}


def compare_datasets(dataset_a: str, dataset_b: str, bbox: list) -> dict:
    """Compares mean values of two datasets over the same bounding box --
    e.g. checking whether ground ice content tracks permafrost probability."""
    stats_a = compute_zonal_statistics(dataset_a, bbox)
    stats_b = compute_zonal_statistics(dataset_b, bbox)
    if "error" in stats_a or "error" in stats_b:
        return {"error": "One or both datasets have no data in this bounding box."}
    return {
        "bbox": bbox,
        dataset_a: {"mean": stats_a["mean"], "unit": stats_a["unit"]},
        dataset_b: {"mean": stats_b["mean"], "unit": stats_b["unit"]},
    }


# --- Tool schema for Claude's tool-use API -------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "list_datasets",
        "description": "List all available geospatial datasets with their meaning, unit, and description. Call this first if unsure what data exists.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "query_raster_at_point",
        "description": "Get the value of a dataset at a specific latitude/longitude point.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset": {"type": "string", "description": "Dataset key, e.g. 'dem', 'permafrost_probability', 'ground_ice_content'"},
                "lat": {"type": "number"},
                "lon": {"type": "number"},
            },
            "required": ["dataset", "lat", "lon"],
        },
    },
    {
        "name": "compute_zonal_statistics",
        "description": "Compute min/max/mean/std of a dataset within a bounding box [west, south, east, north].",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset": {"type": "string"},
                "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
            },
            "required": ["dataset", "bbox"],
        },
    },
    {
        "name": "generate_map_image",
        "description": "Render a dataset as a PNG map image, optionally cropped to a bounding box. Returns base64-encoded PNG.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset": {"type": "string"},
                "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
            },
            "required": ["dataset"],
        },
    },
    {
        "name": "compare_datasets",
        "description": "Compare the mean values of two datasets over the same bounding box.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_a": {"type": "string"},
                "dataset_b": {"type": "string"},
                "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
            },
            "required": ["dataset_a", "dataset_b", "bbox"],
        },
    },
]

TOOL_FUNCTIONS = {
    "list_datasets": lambda **kwargs: list_datasets(),
    "query_raster_at_point": query_raster_at_point,
    "compute_zonal_statistics": compute_zonal_statistics,
    "generate_map_image": generate_map_image,
    "compare_datasets": compare_datasets,
}
