# SIH26067 — OceanViz 3D Grand Finale Prototype

A web-based interactive 3D ocean visualization platform for the Smart India Hackathon 2026 problem statement **SIH26067**:

> Develop a web-based interactive 3D visualization platform that integrates numerical ocean model outputs and in-situ observations.

## What it does

- **3D Cesium.js globe** showing the Indian EEZ region (72°E–86°E, 8°N–20°N).
- **Colored ocean data layers** for temperature, salinity, chlorophyll and current speed.
- **Depth slider** animates a horizontal slice through 0, 50, 100, 200, 500 m.
- **Time animation** plays through 6 synthetic time steps to show seasonal/monsoon variations.
- **Click-to-probe** on the data layer shows the exact variable value at any lat/lon.
- **Current vectors** rendered as colored 3D arrows on the selected depth slice.
- **Instrument markers** for Argo floats, gliders and CTD with clickable depth profiles.
- **2026-grade glassmorphism dashboard** with colorbar legend, variable/depth/time controls and profile chart.

## Tech stack

- **Backend:** FastAPI + Python + NumPy + Pillow
- **Frontend:** HTML/CSS + vanilla JavaScript + Cesium.js + Chart.js
- **Data:** synthetic but physically-inspired 4D ocean data (lat/lon/depth/time) and instrument profiles
- **Packaging:** one-command `python main.py` local server

## Files

- `main.py` — FastAPI server and REST routes
- `data/adapter.py` — pluggable `DataSource` interface and `DataAdapter`
- `data/synthetic.py` — realistic synthetic 4D ocean dataset and image rendering
- `static/index.html` — dashboard UI
- `static/style.css` — dark ocean-science theme
- `static/app.js` — Cesium.js globe, data layers, current vectors, chart
- `docs/SIH26067_Presentation.pptx` — 7-slide pitch deck
- `docs/demo_script.md` — 3-minute demo script
- `docs/architecture_diagram.md` — system architecture

## How to run

Requires Python 3.10+ and the packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
python main.py
```

Or use the one-click scripts:

- Windows PowerShell: `.\run.ps1`
- Linux/macOS: `bash run.sh`
- Docker: `docker-compose up --build`

Run backend health checks with `python test_endpoints.py`.

Open http://localhost:8765 in your browser.

## API endpoints

- `GET /api/variables` — variables, depths, times and map bounds
- `GET /api/slice?variable={id}&depth={m}&time={t}` — 2D data slice with min/max
- `GET /api/image?variable={id}&depth={m}&time={t}` — PNG georeferenced overlay for Cesium
- `GET /api/colorbar?variable={id}` — colorbar PNG
- `GET /api/instruments` — Argo/Glider/CTD positions
- `GET /api/profile/{id}?variable={id}&time={t}` — depth profile for an instrument
- `GET /api/vectors?depth={m}&time={t}&n={n}` — current vector grid

## National-level improvements in this version

1. Replaced Three.js with **Cesium.js** for a real geospatial globe and OSM basemap.
2. **Realistic synthetic data** inspired by Indian Ocean physics: monsoon currents, coastal chlorophyll blooms, thermal gradients and salinity patterns.
3. **PNG image tiles** for fast Cesium overlay rendering with proper color maps.
4. **Current vectors** with U/V components and speed color mapping.
5. **Time animation** with play/pause.
6. **Profile charts** for any selected instrument and variable.
7. **Click-to-probe** data layer for exact lat/lon/variable values.
8. **2026-grade UI** — glassmorphism, spatial depth, bento cards, HUD accents, microinteractions and dark-mode excellence.
9. **Pluggable backend**: `DataSource` interface + `SyntheticSource` makes real-data integration a one-file change.
10. **Presentation materials**: PPT, demo script and architecture diagram.

## Notes

- Data is synthetic and deterministic. The `data/adapter.py` `DataSource` interface makes swapping to real NetCDF/xarray/OPeNDAP sources a one-file change.
- Cesium.js and Chart.js are vendored under `static/lib/` so the demo runs fully offline once the backend is started.
