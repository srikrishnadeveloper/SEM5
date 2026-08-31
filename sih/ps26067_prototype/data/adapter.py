"""
Data adapter interface for SIH26067.

The DataSource abstraction lets the backend switch between synthetic demo data
and real NetCDF/OPeNDAP sources without touching the FastAPI routes.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any


class DataSource(ABC):
    """Abstract interface for an ocean data source."""

    @abstractmethod
    def get_variables(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def get_slice(self, variable: str, depth: float, time: int) -> Dict[str, Any]:
        ...

    @abstractmethod
    def get_image(self, variable: str, depth: float, time: int, width: int = None, height: int = None) -> bytes:
        ...

    @abstractmethod
    def get_colorbar(self, variable: str, width: int = 256, height: int = 32) -> bytes:
        ...

    @abstractmethod
    def get_instruments(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def get_profile(self, instrument_id: str, variable: str, time: int) -> Dict[str, Any]:
        ...

    @abstractmethod
    def get_vectors(self, depth: float, time: int, n: int = 10, scale: float = 0.8) -> Dict[str, Any]:
        ...


class DataAdapter:
    """Thin adapter that delegates all calls to the active DataSource."""

    def __init__(self, source: DataSource):
        self.source = source

    def __getattr__(self, name: str):
        """Pass through any call to the underlying source."""
        return getattr(self.source, name)
