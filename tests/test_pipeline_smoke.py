import numpy as np

from app.inference.engine import InferenceEngine
from app.pipeline.processor import PassthroughProcessor


def test_passthrough_and_engine_roundtrip() -> None:
    frame = np.zeros((256, 1), dtype=np.float32)

    processor = PassthroughProcessor()
    engine = InferenceEngine("torch")

    out = engine.infer(processor.process(frame))

    assert out.shape == frame.shape
    assert out.dtype == frame.dtype
