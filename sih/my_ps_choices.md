# My SIH 2026 Problem Statement Choices

This is a running list of PSs I am considering. I will score each one on fit, difficulty and shortlisting odds.

---

## 1. SIH26067 — 3D Ocean Data Visualization Platform

| Field | Value |
|-------|-------|
| PS Number | SIH26067 |
| Title | Develop a web-based interactive 3D visualization platform that integrates numerical ocean model outputs and in-situ observations. |
| Organization | Ministry of Earth Sciences (MoES) |
| Department | Indian National Centre for Ocean Information Services (INCOIS) Ocean Valley |
| Category | Software |
| Theme | Smart Automation |
| Submitted ideas | 0/500 |

### What this PS is asking

INCOIS has a lot of ocean model data (NetCDF files with temperature, salinity, currents, chlorophyll, etc.) and real-world data from Argo floats and underwater gliders. Right now there is no single web tool to see all of this in 3D. They want a browser-based 3D visualization platform where you can:

- Render 3D ocean fields (temp, salinity, currents) with depth slices and isosurfaces.
- Overlay Argo/Glider/CTD instrument data on the same view.
- Read NetCDF and text files.
- Let users pick variables, animate through time, control colorbars and opacity.
- Build it with WebGL / Three.js or Cesium.js and a REST/OPeNDAP backend.

### Why it could be a good choice

- **Clear technical brief** — it tells you exactly what tech stack and features are expected.
- **Low submissions right now** (0/500), so shortlisting competition may be lower.
- **From a technical org** (MoES/INCOIS), so the problem is well-scoped and judges will know what they want.
- **Good for a CSE/IT team** if you have people who know web dev, 3D graphics or Python data tools.

### Why it could be risky

- **Heavy 3D graphics work** — WebGL/Three.js is not easy if no one on the team has done it before.
- **NetCDF/ocean data** — you need someone who can parse and serve large scientific datasets.
- **36-hour MVP risk** — a full 3D volumetric renderer with time animation is a lot. For the first round, a 2D/3D prototype with one variable and one dataset may be enough, but the final demo needs to look convincing.

### My initial score

| Criteria | Rating (1-5) | Notes |
|----------|--------------|-------|
| Domain interest | ? | Ocean science — do I/we care about this? |
| Technical fit | ? | Need someone with 3D JS or data viz experience |
| Feasibility in 36h | ? | Prototype is doable, full 3D is hard |
| Competition | 4/5 | 0/500 submissions at the time of download |
| Clarity of PS | 5/5 | Very detailed and specific |

### Verdict so far

Worth keeping on the shortlist because of the clear requirements and low submission count, but only if the team has or can quickly learn 3D web viz and NetCDF handling. If not, this will be very hard to finish.

---

## Notes

- I will add more PSs here as I go through the CSV.
- Before locking this, I need to check team skills in: Three.js / Cesium / WebGL, Python xarray/NetCDF, frontend (React/Vue), and backend (FastAPI/Flask).
