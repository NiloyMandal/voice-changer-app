import numpy as np

from app.pipeline.processor import PassthroughProcessor


def test_passthrough_processor_returns_same_frame() -> None:
    frame = np.random.randn(64, 1).astype(np.float32)
    processor = PassthroughProcessor()

    out = processor.process(frame)

    assert out is frame
