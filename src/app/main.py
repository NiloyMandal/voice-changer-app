import time

import numpy as np

from app.audio.io import AudioLoopbackStream, pitch_shift_numpy
from app.config.settings import settings
from app.inference.engine import InferenceEngine
from app.pipeline.processor import PassthroughProcessor


def run() -> None:
    processor = PassthroughProcessor()
    engine = InferenceEngine(
        backend=settings.inference_backend,
        torch_model_path=settings.torch_model_path,
        onnx_model_path=settings.onnx_model_path,
        device=settings.inference_device,
        model_input_layout=settings.model_input_layout,
        normalize_mode=settings.inference_normalize,
        target_dbfs=settings.target_dbfs,
        restore_level=settings.restore_level,
    )
    semitones_down = -4.0

    pitch_ratio = 2.0 ** (semitones_down / 12.0)
    source_positions = np.arange(settings.block_size, dtype=np.float32) / pitch_ratio

    def process_frame(frame):
        frame = processor.process(frame)
        frame = pitch_shift_numpy(
            audio=frame,
            semitones=semitones_down,
            source_positions=source_positions,
        )
        return engine.infer(frame)

    io = AudioLoopbackStream(
        sample_rate=settings.sample_rate,
        channels=settings.channels,
        block_size=settings.block_size,
        input_device=settings.input_device,
        output_device=settings.output_device,
        queue_size=4,
        processor=process_frame,
    )

    print("Starting low-latency pitched stream. Press Ctrl+C to stop.")
    try:
        io.start()
    except RuntimeError as exc:
        print(f"Audio startup failed: {exc}")
        return

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        io.stop()
        print("Stopped.", io.get_stats())


if __name__ == "__main__":
    run()
