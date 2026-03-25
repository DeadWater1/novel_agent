from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import BackendHealth


class BaseBackend(ABC):
    name: str

    @abstractmethod
    def healthcheck(self) -> BackendHealth:
        raise NotImplementedError
