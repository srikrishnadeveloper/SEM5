"""
NetCDFSource stub — ready for real INCOIS / Copernicus / HYCOM data.

To activate:
    from data.netcdf_source import NetCDFSource
    from data.adapter import DataAdapter
    DATA = DataAdapter(NetCDFSource("path/to/file.nc"))

Currently the class raises NotImplementedError for methods until a real
NetCDF is wired in. The interface matches SyntheticSource exactly, so no
FastAPI route needs changing.
"""

from typing import Any, Dict

from .adapter import DataSource


class NetCDFSource(DataSource):
    """Placeholder for NetCDF / xarray / OPeNDAP ocean data."""

    def __init__(self, path: str, variable_map: Dict[str, str] = None):
        self.path = path
        self.variable_map = variable_map or {
            "temperature": "TEMP",
            "salinity": "SALT",
            "chlorophyll": "CHL",
            "current_u": "U",
            "current_v": "V",
        }

    def _not_ready(self) -> None:
        raise NotImplementedError(
            "NetCDFSource is a stub. Implement it with xarray/netcdf4 "
            "and a real .nc file before use."
        )

    def get_variables(self) -> Dict[str, Any]:
        return {"message": "NetCDFSource stub — implement with xarray"}

    def get_slice(self, variable: str, depth: float, time: int) -> Dict[str, Any]:
        self._not_ready()

    def get_image(self, variable: str, depth: float, time: int, width: int = None, height: int = None) -> bytes:
        self._not_ready()

    def get_colorbar(self, variable: str, width: int = 256, height: int = 32) -> bytes:
        self._not_ready()

    def get_instruments(self) -> Dict[str, Any]:
        self._not_ready()

    def get_profile(self, instrument_id: str, variable: str, time: int) -> Dict[str, Any]:
        self._not_ready()

    def get_vectors(self, depth: float, time: int, n: int = 10, scale: float = 0.8) -> Dict[str, Any]:
        self._not_ready()
