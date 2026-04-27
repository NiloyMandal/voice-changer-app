from typing import Protocol

import numpy as np


class FrameProcessor(Protocol):
    def process(self, frame: np.ndarray) -> np.ndarray:
        ...


class PassthroughProcessor:
    def process(self, frame: np.ndarray) -> np.ndarray:
        return frame
