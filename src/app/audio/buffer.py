from queue import Full, Queue

import numpy as np


class FrameBuffer:
    """Bounded frame queue to prevent unbounded latency growth."""

    def __init__(self, max_frames: int = 32) -> None:
        self._q: Queue[np.ndarray] = Queue(maxsize=max_frames)

    def push(self, frame: np.ndarray) -> bool:
        try:
            self._q.put_nowait(frame)
            return True
        except Full:
            return False

    def pop(self) -> np.ndarray | None:
        if self._q.empty():
            return None
        return self._q.get_nowait()

    def size(self) -> int:
        return self._q.qsize()
