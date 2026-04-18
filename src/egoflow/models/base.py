from __future__ import annotations

from typing import Optional, Protocol

import numpy as np


class Model(Protocol):
    def __init__(self, device: str = "cpu", weights_path: Optional[str] = None): ...

    def load(self) -> None: ...

    def predict(self, frame: np.ndarray) -> dict: ...

    def unload(self) -> None: ...
