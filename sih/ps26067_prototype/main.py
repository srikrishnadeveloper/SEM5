"""
SIH26067 — OceanViz 3D Grand Finale Backend

A FastAPI server that serves a Cesium.js 3D globe frontend along with
georeferenced ocean data images, current vectors, instrument positions
and depth profiles. Data is provided through a pluggable DataSource;
the default source is a realistic synthetic ocean dataset.

Run:
    python main.py

Open http://localhost:8765 in the browser.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from data.adapter import DataAdapter
from data.synthetic import SyntheticSource


BASE_DIR = Path(__file__).resolve().parent


def create_adapter() -> DataAdapter:
    """Create the active data adapter. Swap the source here for real data."""
    source = SyntheticSource()
    return DataAdapter(source)


DATA: DataAdapter = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global DATA
    DATA = create_adapter()
    yield


app = FastAPI(title="OceanViz 3D — SIH26067", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/variables")
async def variables():
    return DATA.get_variables()


@app.get("/api/slice")
async def slice_data(
    variable: str = Query(...),
    depth: float = Query(...),
    time: int = Query(...),
):
    try:
        return DATA.get_slice(variable, depth, time)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/image")
async def image(
    variable: str = Query(...),
    depth: float = Query(...),
    time: int = Query(...),
    width: int = Query(1024),
    height: int = Query(None),
):
    try:
        body = DATA.get_image(variable, depth, time, width=width, height=height)
        return Response(content=body, media_type="image/png")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/colorbar")
async def colorbar(
    variable: str = Query(...),
    width: int = Query(256),
    height: int = Query(32),
):
    try:
        body = DATA.get_colorbar(variable, width=width, height=height)
        return Response(content=body, media_type="image/png")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/instruments")
async def instruments():
    return DATA.get_instruments()


@app.get("/api/profile/{instrument_id}")
async def profile(
    instrument_id: str,
    variable: str = Query(...),
    time: int = Query(0),
):
    try:
        return DATA.get_profile(instrument_id, variable, time)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/vectors")
async def vectors(
    depth: float = Query(...),
    time: int = Query(...),
    n: int = Query(10),
    scale: float = Query(0.8),
):
    try:
        return DATA.get_vectors(depth, time, n=n, scale=scale)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8765))
    uvicorn.run("main:app", host=host, port=port, reload=False)
