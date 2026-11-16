"""Type stubs for the sweepeq convolution engine (Rust, via PyO3)."""

import numpy as np
from numpy.typing import NDArray

__version__: str

class Convolver:
    """A streaming convolver: call ``process`` with consecutive blocks of any length."""

    def __init__(
        self, taps: NDArray[np.float32], method: str = "partitioned", block: int = 256
    ) -> None: ...
    def process(self, x: NDArray[np.float32]) -> NDArray[np.float32]: ...
    def reset(self) -> None: ...
    @property
    def latency(self) -> int: ...
    @property
    def taps(self) -> int: ...
