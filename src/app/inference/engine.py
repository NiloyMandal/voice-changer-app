import numpy as np


class InferenceEngine:
    """Minimal backend switch for real-time frame inference."""

    def __init__(self, backend: str = "torch") -> None:
        if backend not in {"torch", "onnx"}:
            raise ValueError("backend must be 'torch' or 'onnx'")
        self.backend = backend

    def infer(self, frame: np.ndarray) -> np.ndarray:
        # Placeholder: return input unchanged until a model is wired.
        return frame
