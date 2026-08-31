# SIH26067 — Architecture

## System Overview

The platform is split into three layers: a browser frontend, a FastAPI backend, and a pluggable data-source layer. The backend talks to the frontend through REST endpoints, while a `DataAdapter` keeps the rest of the code unchanged when we move from synthetic demo data to real NetCDF or OPeNDAP streams.

## ASCII Architecture Diagram

```
                 ┌─────────────────────────────────────────────────┐
                 │               User Browser                      │
                 │   Cesium.js 3D globe + vanilla JS UI            │
                 │   Chart.js profiles + time/depth/variable       │
                 │   controls, color scale and instrument panel    │
                 └───────────────────────┬─────────────────────────┘
                                         │  HTTPS / REST
                                         ▼
                 ┌─────────────────────────────────────────────────┐
                 │              FastAPI Backend                    │
                 │  ┌───────────┐ ┌───────────┐ ┌───────────────┐  │
                 │  │  /image   │ │  /slice   │ │  /variables   │  │
                 │  │   .png    │ │  (grid)   │ │  (metadata)   │  │
                 │  └─────┬─────┘ └─────┬─────┘ └───────┬───────┘  │
                 │  ┌─────┴─────────────┴───────────────┴───────┐  │
                 │  │              DataAdapter                   │  │
                 │  │  abstract: get_tile(), get_slice()         │  │
                 │  │            get_instruments(),              │  │
                 │  │            get_profile(), get_variables()  │  │
                 │  └────────────────────┬───────────────────────┘  │
                 └───────────────────────┼──────────────────────────┘
                                         │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                             ▼
        ┌────────────────────────────┐           ┌────────────────────────────┐
        │      SyntheticSource       │           │      NetCDFSource          │
        │  in-memory 4D ocean grid   │           │  xarray + local .nc file   │
        │  temperature, salinity,    │           │  or OPeNDAP / THREDDS      │
        │  chlorophyll, current U/V  │           │  remote endpoint           │
        │  + instrument trajectories │           │  (future plug-in)          │
        └────────────────────────────┘           └────────────────────────────┘
```

## Component Explanation

### 1. Frontend

- **Cesium.js** renders the 3D globe with a free OpenStreetMap basemap. We avoid paid map tokens, which is important for offline nodal-centre demos.
- **Vanilla JavaScript** drives the UI: variable selector, depth slider, time slider, play/pause, current-vector toggle and instrument panel.
- **Chart.js** draws depth profiles and comparison charts when an instrument is clicked.
- The frontend consumes PNG tiles for the globe, JSON grids for optional overlays, and profile data for instrument popups.

### 2. Backend

- **FastAPI** exposes the following REST endpoints:
  - `GET /api/variables` — lists variables, depths, times and geographic extents.
  - `GET /api/image?variable={id}&depth={m}&time={t}` — georeferenced PNG overlay for the Cesium imagery layer.
  - `GET /api/slice?variable=...&depth=...&time=...` — 2D grid values with min/max.
  - `GET /api/vectors?depth=...&time=...` — U/V current grid and speed.
  - `GET /api/instruments` — Argo, glider, CTD and BGC-Argo positions and metadata.
  - `GET /api/profile/{id}?variable=...` — depth profile for a clicked instrument.
- CORS is enabled and the backend can serve static files, so the app runs from a single `python main.py` command.

### 3. DataAdapter

- The `DataAdapter` is a thin abstraction that calls a `DataSource` interface.
- The interface has five methods: `get_tile()`, `get_slice()`, `get_instruments()`, `get_profile()` and `get_variables()`.
- Because every route talks to the adapter, the rest of the backend stays unchanged when we swap data sources.

### 4. Data Sources

- **SyntheticSource** (current implementation): a deterministic, physics-inspired 4D ocean dataset generated in Python. It produces lat/lon/depth/time grids for temperature, salinity, chlorophyll and current U/V, plus instrument trajectories. It is used today for offline demos and validation.
- **NetCDFSource** (stub / future): an implementation using `xarray` to read local NetCDF files or OPeNDAP/THREDDS endpoints. Because it exposes the same five methods, replacing the synthetic source is a one-file configuration change, proving the architecture is ready for real INCOIS data.

This layered design keeps the demo self-contained and offline-capable today, while making the jump to operational data straightforward tomorrow.
