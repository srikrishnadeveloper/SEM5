"""
SyntheticSource: realistic deterministic ocean data for SIH26067 demos.

Replace this with NetCDFSource / OPeNDAPSource to plug in real data.
"""

import io
import math
from typing import Dict, List, Any

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

from .adapter import DataSource


class SyntheticSource(DataSource):
    def __init__(
        self,
        lon_min: float = 72.0,
        lon_max: float = 86.0,
        lat_min: float = 8.0,
        lat_max: float = 20.0,
        nx: int = 120,
        ny: int = 120,
        depths: List[int] = None,
        times: List[int] = None,
    ):
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.nx = nx
        self.ny = ny
        self.lons = np.linspace(lon_min, lon_max, nx)
        self.lats = np.linspace(lat_min, lat_max, ny)
        self.depths = depths or [0, 50, 100, 200, 500]
        self.times = times or [0, 1, 2, 3, 4, 5]

        self.variables = [
            {"id": "temperature", "name": "Sea Water Temperature", "unit": "°C", "colormap": "thermal"},
            {"id": "salinity", "name": "Salinity", "unit": "psu", "colormap": "haline"},
            {"id": "chlorophyll", "name": "Chlorophyll", "unit": "mg/m³", "colormap": "algae"},
            {"id": "current_speed", "name": "Current Speed", "unit": "m/s", "colormap": "speed"},
        ]

        self.colormaps = {
            "thermal": LinearSegmentedColormap.from_list("thermal", [
                "#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8",
                "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026"
            ]),
            "haline": LinearSegmentedColormap.from_list("haline", [
                "#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6",
                "#4292c6", "#2171b5", "#08519c", "#08306b"
            ]),
            "algae": LinearSegmentedColormap.from_list("algae", [
                "#ffffcc", "#c7e9b4", "#7fcdbb", "#41b6c4", "#1d91c0",
                "#225ea8", "#0c2c84"
            ]),
            "speed": LinearSegmentedColormap.from_list("speed", [
                "#0d0887", "#41049d", "#6a00a8", "#8d0da5", "#b12a90",
                "#cc4778", "#e16462", "#f2844b", "#f9c631", "#f0f921"
            ]),
        }

        self.data_4d: Dict[str, np.ndarray] = {}
        self.current_u: np.ndarray = None
        self.current_v: np.ndarray = None
        self.profiles: Dict[str, Any] = {}

        self._generate_data()
        self._build_profiles()

    def _generate_data(self):
        """Generate realistic synthetic ocean data."""
        rng = np.random.default_rng(2026)
        lon_grid, lat_grid = np.meshgrid(self.lons, self.lats)

        for var in self.variables:
            vid = var["id"]
            if vid == "current_speed":
                continue

            arr = np.zeros((len(self.times), len(self.depths), self.ny, self.nx), dtype=np.float32)

            for t_idx, t in enumerate(self.times):
                monsoon_factor = 1.0 if t < 3 else -1.0

                for d_idx, depth in enumerate(self.depths):
                    if vid == "temperature":
                        base = 28.5 - 0.06 * (lat_grid - 8) - 0.005 * depth
                        warm_pool = 1.2 * np.exp(-((lon_grid - 84) ** 2 + (lat_grid - 10) ** 2) / 20)
                        seasonal = 0.5 * np.sin(np.radians(t * 60))
                        arr[t_idx, d_idx] = base + warm_pool + seasonal + rng.normal(0, 0.3, (self.ny, self.nx))

                    elif vid == "salinity":
                        base = 35.2 + 0.01 * (lat_grid - 8) + 0.0004 * depth
                        bay_dilution = -0.8 * np.exp(-((lon_grid - 87) ** 2) / 8)
                        coastal_low = -0.4 * (np.exp(-((lon_grid - 72) ** 2) / 6) + np.exp(-((lon_grid - 86) ** 2) / 6))
                        arr[t_idx, d_idx] = base + bay_dilution + coastal_low + rng.normal(0, 0.1, (self.ny, self.nx))

                    elif vid == "chlorophyll":
                        coastal = (np.exp(-((lon_grid - 72.5) ** 2) / 5) + np.exp(-((lon_grid - 85.5) ** 2) / 5))
                        base = 2.5 * coastal * np.exp(-depth / 60)
                        bloom = 0.5 * np.exp(-((lon_grid - 76) ** 2 + (lat_grid - 15) ** 2) / 15) * np.exp(-depth / 40)
                        arr[t_idx, d_idx] = base + bloom + rng.normal(0, 0.05, (self.ny, self.nx))
                        arr[t_idx, d_idx] = np.maximum(0, arr[t_idx, d_idx])

            self.data_4d[vid] = arr

        # Currents (U/V)
        self.current_u = np.zeros((len(self.times), len(self.depths), self.ny, self.nx), dtype=np.float32)
        self.current_v = np.zeros((len(self.times), len(self.depths), self.ny, self.nx), dtype=np.float32)

        for t_idx, t in enumerate(self.times):
            monsoon_factor = 1.0 if t < 3 else -1.0
            for d_idx, depth in enumerate(self.depths):
                u = monsoon_factor * (0.3 + 0.4 * np.cos(np.radians(lat_grid * 2)))
                v = monsoon_factor * (0.2 + 0.3 * np.sin(np.radians(lon_grid * 1.5)))
                u += 0.2 * np.sin(np.radians((lon_grid - 79) * 2))
                v += 0.2 * np.cos(np.radians((lat_grid - 14) * 2))
                self.current_u[t_idx, d_idx] = u + rng.normal(0, 0.05, (self.ny, self.nx))
                self.current_v[t_idx, d_idx] = v + rng.normal(0, 0.05, (self.ny, self.nx))

        # Current speed
        self.data_4d["current_speed"] = np.sqrt(self.current_u ** 2 + self.current_v ** 2)

        # Instrument definitions
        self.instruments = [
            {"id": "argo_1", "name": "Argo-01", "type": "Argo Float", "lat": 12.5, "lon": 76.2},
            {"id": "argo_2", "name": "Argo-02", "type": "Argo Float", "lat": 15.3, "lon": 80.1},
            {"id": "glider_1", "name": "Glider-01", "type": "Underwater Glider", "lat": 10.8, "lon": 74.5},
            {"id": "glider_2", "name": "Glider-02", "type": "Underwater Glider", "lat": 17.2, "lon": 83.4},
            {"id": "ctd_1", "name": "CTD-01", "type": "CTD", "lat": 13.9, "lon": 78.7},
            {"id": "argo_3", "name": "Argo-03", "type": "BGC-Argo", "lat": 16.5, "lon": 81.8},
            {"id": "glider_3", "name": "Glider-03", "type": "Underwater Glider", "lat": 9.8, "lon": 75.2},
            {"id": "ctd_2", "name": "CTD-02", "type": "CTD", "lat": 14.7, "lon": 79.9},
        ]

    def _nearest_indices(self, lat: float, lon: float):
        j = int(np.argmin(np.abs(self.lats - lat)))
        i = int(np.argmin(np.abs(self.lons - lon)))
        return j, i

    def _build_profiles(self):
        rng = np.random.default_rng(99)

        for inst in self.instruments:
            self.profiles[inst["id"]] = {}
            j, i = self._nearest_indices(inst["lat"], inst["lon"])

            for var in [v["id"] for v in self.variables] + ["current_u", "current_v"]:
                self.profiles[inst["id"]][var] = {}

                for t_idx, t in enumerate(self.times):
                    profile = []

                    for d_idx, depth in enumerate(self.depths):
                        if var == "current_u":
                            val = float(self.current_u[t_idx, d_idx, j, i])
                        elif var == "current_v":
                            val = float(self.current_v[t_idx, d_idx, j, i])
                        else:
                            val = float(self.data_4d[var][t_idx, d_idx, j, i])

                        val += float(rng.normal(0, 0.05))

                        if var == "temperature":
                            val -= 0.006 * depth
                        elif var == "salinity":
                            val += 0.0006 * depth
                        elif var == "chlorophyll":
                            val = max(0.0, val * np.exp(-depth / 90))

                        profile.append({"depth": int(depth), "value": round(val, 3)})

                    self.profiles[inst["id"]][var][t] = profile

    def _meta_for(self, var_id: str):
        for v in self.variables:
            if v["id"] == var_id:
                return v
        return None

    def get_variables(self) -> Dict[str, Any]:
        return {
            "variables": self.variables,
            "depths": self.depths,
            "times": self.times,
            "bounds": {
                "lon_min": self.lon_min, "lon_max": self.lon_max,
                "lat_min": self.lat_min, "lat_max": self.lat_max,
            },
        }

    def get_slice(self, variable: str, depth: float, time: int) -> Dict[str, Any]:
        if variable not in self.data_4d:
            raise ValueError("Variable not found")
        if depth not in self.depths:
            raise ValueError("Depth not found")
        if time not in self.times:
            raise ValueError("Time not found")

        d_idx = self.depths.index(depth)
        t_idx = self.times.index(time)
        grid = self.data_4d[variable][t_idx, d_idx]

        return {
            "variable": variable,
            "depth": depth,
            "time": time,
            "lons": self.lons.tolist(),
            "lats": self.lats.tolist(),
            "values": grid.tolist(),
            "min": round(float(grid.min()), 3),
            "max": round(float(grid.max()), 3),
        }

    def _render_png(self, grid: np.ndarray, variable: str, width: int = 1024, height: int = None) -> bytes:
        if height is None:
            lon_range = self.lon_max - self.lon_min
            lat_range = self.lat_max - self.lat_min
            height = int(width * (lat_range / lon_range))

        meta = self._meta_for(variable)
        cmap = self.colormaps.get(meta["colormap"], self.colormaps["thermal"]) if meta else self.colormaps["thermal"]

        vmin = float(grid.min())
        vmax = float(grid.max())
        if vmax - vmin < 1e-9:
            vmax = vmin + 1
        norm = (grid - vmin) / (vmax - vmin)
        norm = np.clip(norm, 0, 1)

        rgba = cmap(norm)
        rgba_uint8 = (rgba * 255).astype(np.uint8)

        img = Image.fromarray(rgba_uint8)
        img = img.resize((width, height), Image.BILINEAR)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def get_image(self, variable: str, depth: float, time: int, width: int = 1024, height: int = None) -> bytes:
        if variable not in self.data_4d:
            raise ValueError("Variable not found")
        if depth not in self.depths:
            raise ValueError("Depth not found")
        if time not in self.times:
            raise ValueError("Time not found")

        d_idx = self.depths.index(depth)
        t_idx = self.times.index(time)
        grid = self.data_4d[variable][t_idx, d_idx]
        return self._render_png(grid, variable, width=width, height=height)

    def get_colorbar(self, variable: str, width: int = 256, height: int = 32) -> bytes:
        meta = self._meta_for(variable)
        cmap = self.colormaps.get(meta["colormap"], self.colormaps["thermal"]) if meta else self.colormaps["thermal"]

        gradient = np.linspace(0, 1, width).reshape(1, width)
        rgba = cmap(gradient)
        rgba_uint8 = (rgba * 255).astype(np.uint8)
        rgba_uint8 = np.repeat(rgba_uint8, height, axis=0)

        img = Image.fromarray(rgba_uint8)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def get_instruments(self) -> Dict[str, Any]:
        return {"instruments": self.instruments}

    def get_profile(self, instrument_id: str, variable: str, time: int) -> Dict[str, Any]:
        if instrument_id not in self.profiles:
            raise ValueError("Instrument not found")
        if variable not in self.profiles[instrument_id]:
            raise ValueError("Variable not found")
        if time not in self.times:
            raise ValueError("Time not found")

        return {
            "instrument_id": instrument_id,
            "variable": variable,
            "time": time,
            "profile": self.profiles[instrument_id][variable][time],
        }

    def get_vectors(self, depth: float, time: int, n: int = 10, scale: float = 0.8) -> Dict[str, Any]:
        if depth not in self.depths:
            raise ValueError("Depth not found")
        if time not in self.times:
            raise ValueError("Time not found")

        d_idx = self.depths.index(depth)
        t_idx = self.times.index(time)

        step_x = max(1, self.nx // n)
        step_y = max(1, self.ny // n)

        vectors = []
        for j in range(0, self.ny, step_y):
            for i in range(0, self.nx, step_x):
                u = float(self.current_u[t_idx, d_idx, j, i])
                v = float(self.current_v[t_idx, d_idx, j, i])
                speed = float(self.data_4d["current_speed"][t_idx, d_idx, j, i])

                if speed < 0.05:
                    continue

                lon = float(self.lons[i])
                lat = float(self.lats[j])

                deg_per_m = 1.0 / (111000.0 * math.cos(math.radians(lat)))
                lat_per_m = 1.0 / 111000.0

                end_lon = lon + u * scale * 1000 * deg_per_m
                end_lat = lat + v * scale * 1000 * lat_per_m

                vectors.append({
                    "lat": lat,
                    "lon": lon,
                    "end_lat": end_lat,
                    "end_lon": end_lon,
                    "u": round(u, 3),
                    "v": round(v, 3),
                    "speed": round(speed, 3),
                })

        return {
            "depth": depth,
            "time": time,
            "count": len(vectors),
            "vectors": vectors,
        }
