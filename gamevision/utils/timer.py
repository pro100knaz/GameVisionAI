"""Performance timing utility."""

from __future__ import annotations

import time
from typing import Any


class Timer:
    """Simple context-manager timer for measuring execution time.

    Example:
        >>> with Timer("inference") as t:
        ...     result = model.predict(frame)
        >>> print(f"Took {t.ms:.1f}ms")
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.start: float = 0.0
        self.end: float = 0.0

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        return self.end - self.start

    @property
    def ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed * 1000

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.end = time.perf_counter()

    def __repr__(self) -> str:
        if self.end > 0:
            return f"Timer({self.name!r}, {self.ms:.1f}ms)"
        return f"Timer({self.name!r}, running)"
