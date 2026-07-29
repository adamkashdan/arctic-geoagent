# Arctic GeoAgent

An agentic tool built on the Claude API that answers natural-language questions
about Arctic permafrost/remote-sensing raster data — using Claude's tool-use
(function-calling) over a small semantic layer, with a deployed FastAPI
interface rather than a notebook.

Built as a portfolio project connecting 20+ years of geospatial/remote-sensing
research experience (GDAL, Rasterio, QGIS, Google Earth Engine) to modern
agentic AI tooling.

## Why this exists

Most of my geospatial work has lived in research pipelines and notebooks.
This project is a deliberately small, complete example of the *other* half
of that skillset: taking a geospatial data pipeline and exposing it as a
tool-using agent with a real deployed interface — the pattern increasingly
expected in industry data science roles that combine domain data with LLM
agents.

## Architecture

```
User question (natural language)
        |
        v
  FastAPI /ask endpoint  (src/main.py)
        |
        v
  Agent loop (src/agent.py)  <-- Claude API, tool-use
        |
        v
  Tool functions (src/tools.py)  <-- Rasterio over GeoTIFFs
        |
        v
  Semantic layer (semantic_layer.yaml)  <-- describes datasets to the LLM
        |
        v
  Raster data (data/*.tif)
```

**Semantic layer** (`semantic_layer.yaml`): a human-readable catalogue of
what each dataset is, its unit, and what it's useful for. This is what lets
the agent reason about *which* tool to call and *how to interpret* the
result, without ever seeing raw pixel values directly.

**Tools** (`src/tools.py`): five functions exposed to Claude —
`list_datasets`, `query_raster_at_point`, `compute_zonal_statistics`,
`generate_map_image`, `compare_datasets`. Each takes simple, LLM-friendly
arguments and returns simple, LLM-friendly results (numbers, short dicts, or
a base64 PNG) — never a raw NumPy array.

**Agent** (`src/agent.py`): the tool-use loop against the Claude API
(`claude-sonnet-4-6`). Sends the question, executes whatever tools Claude
requests, feeds results back, repeats until Claude has a final answer.

**API** (`src/main.py`): a thin FastAPI layer (`/ask`, `/datasets`,
`/health`) — the "deployed interface" piece, not just a script.

## About the data

This sandbox had no network access to real geospatial data providers, so
`src/generate_demo_data.py` generates small **synthetic** GeoTIFFs (DEM,
permafrost probability, ground ice content) with realistic structure (real
CRS, resolution, and value ranges) for a fictional tile near Herschel
Island, Yukon — a real study area from my published research.

**To use real data**: replace the files in `data/` with real GeoTIFFs
(e.g., exported from Google Earth Engine, built from Sentinel-2/Landsat via
GDAL, or a real permafrost probability map) and update the `path` fields in
`semantic_layer.yaml`. Nothing else in the pipeline needs to change — this
is the point of the semantic-layer pattern.

## Setup

```bash
pip install -r requirements.txt
python src/generate_demo_data.py     # builds the demo rasters in data/
export ANTHROPIC_API_KEY=sk-...      # your key
```

## Run it

**CLI demo:**
```bash
cd src
python agent.py "What's the average ground ice content in the study area, \
and how does it relate to permafrost probability there?"
```

**As a service:**
```bash
cd src
uvicorn main:app --reload --port 8000
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "What is the elevation at 69.3N, -138.9W?"}'
```

## What's verified vs. what needs your API key

Everything **except the live Claude API call** was tested in the build
environment (no API key was available there):
- ✅ Demo raster generation
- ✅ All five tool functions, run directly (point query, zonal stats,
  map image generation, dataset comparison) — see test output in the
  build log
- ✅ Map image rendering (confirmed visually)
- ⬜ The Claude tool-use loop itself (`agent.py`) — logically follows the
  standard Anthropic tool-use pattern, but wasn't run end-to-end since no
  API key was present in the sandbox. Run it once with your key to confirm
  before an interview demo.

## Possible next steps

- Swap in real data (Google Earth Engine export, or your own published
  Herschel Island dataset)
- Add a minimal chat frontend (Streamlit or a small HTML/JS page)
- Add a `compute_slope` or `hazard_susceptibility_index` tool drawing on
  your avalanche/landslide hazard-assessment publications
- Deploy the FastAPI app somewhere reachable (Render/Railway) for a live
  demo link on resume
