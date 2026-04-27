from collections.abc import Callable
from queue import Empty, Full, Queue
from typing import Any

import numpy as np
import sounddevice as sd

from app.audio.buffer import FrameBuffer


def pitch_shift_numpy(
    audio: np.ndarray,
    semitones: float,
    source_positions: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a lightweight pitch shift using linear interpolation.

    The function preserves frame length to keep callback timing stable.
    """
    if audio.ndim not in (1, 2):
        raise ValueError("audio must be 1D or 2D (frames[, channels])")

    frames = int(audio.shape[0])
    if frames < 2:
        return audio.copy()

    pitch_ratio = 2.0 ** (semitones / 12.0)
    if source_positions is None or source_positions.shape[0] != frames:
        source_positions = np.arange(frames, dtype=np.float32) / pitch_ratio

    source_positions_arr = np.asarray(source_positions, dtype=np.float32)
    source_positions_arr = np.clip(source_positions_arr, 0.0, float(frames - 1))
    x = np.arange(frames, dtype=np.float32)

    if audio.ndim == 1:
        shifted = np.interp(source_positions_arr, x, audio)
        return shifted.astype(audio.dtype, copy=False)

    out = np.empty_like(audio)
    for ch in range(audio.shape[1]):
        out[:, ch] = np.interp(source_positions_arr, x, audio[:, ch]).astype(
            audio.dtype,
            copy=False,
        )
    return out


class RealTimeAudioIO:
    """Duplex stream wrapper using callback mode for low-latency operation."""

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        block_size: int,
        input_device: int | None = None,
        output_device: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        self.input_device = input_device
        self.output_device = output_device
        self.input_frames = FrameBuffer()
        self.output_frames = FrameBuffer()
        self._stream: sd.Stream | None = None

    def start(self, on_input_frame: Callable[[np.ndarray], np.ndarray]) -> None:
        def callback(indata, outdata, frames, time, status):  # noqa: ANN001
            if status:
                # Keep callback light; heavy logging here can add glitches.
                pass

            in_frame = np.copy(indata)
            processed = on_input_frame(in_frame)
            outdata[:] = processed

        self._stream = sd.Stream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=self.channels,
            dtype="float32",
            device=(self.input_device, self.output_device),
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None


class AudioLoopbackStream:
    """Low-latency input/output loopback using separate sounddevice streams."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        block_size: int = 1024,
        input_device: int | None = None,
        output_device: int | None = None,
        queue_size: int = 4,
        dtype: str = "float32",
        processor: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        self.input_device = input_device
        self.output_device = output_device
        self.dtype = dtype
        self.processor = processor

        self._queue: Queue[np.ndarray] = Queue(maxsize=queue_size)
        self._input_stream: sd.InputStream | None = None
        self._output_stream: sd.OutputStream | None = None
        self._running = False

        self.input_overflows = 0
        self.output_underflows = 0
        self.callback_errors = 0

    @property
    def is_running(self) -> bool:
        return self._running

    def _first_device_with_channels(self, direction: str) -> int | None:
        """Return first usable device index for input or output."""
        devices = sd.query_devices()
        key = "max_input_channels" if direction == "input" else "max_output_channels"
        for idx, info in enumerate(devices):
            if int(info[key]) >= self.channels:
                return idx
        return None

    def _resolve_device_index(self, direction: str, requested: int | None) -> int:
        """Resolve a valid device index, including fallback when default is invalid."""
        if requested is not None and requested >= 0:
            return requested

        default_in, default_out = sd.default.device
        default_idx = default_in if direction == "input" else default_out

        if isinstance(default_idx, int) and default_idx >= 0:
            return default_idx

        fallback = self._first_device_with_channels(direction)
        if fallback is not None:
            return fallback

        raise RuntimeError(
            f"No usable {direction} audio device found. "
            "Connect an audio device and/or set VOICE_INPUT_DEVICE and "
            "VOICE_OUTPUT_DEVICE to valid indices."
        )

    def _input_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        del frames, time_info
        if status.input_overflow:
            self.input_overflows += 1

        try:
            frame = indata.copy()
            if self.processor is not None:
                frame = self.processor(frame)
            self._queue.put_nowait(frame)
        except Full:
            # Keep latency bounded by dropping newest data when queue is full.
            self.input_overflows += 1
        except Exception:
            self.callback_errors += 1

    def _output_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        del frames, time_info
        if status.output_underflow:
            self.output_underflows += 1

        try:
            frame = self._queue.get_nowait()
            if frame.shape != outdata.shape:
                outdata.fill(0)
                self.callback_errors += 1
                return
            outdata[:] = frame
        except Empty:
            outdata.fill(0)
            self.output_underflows += 1
        except Exception:
            outdata.fill(0)
            self.callback_errors += 1

    def start(self) -> None:
        if self._running:
            return

        try:
            input_idx = self._resolve_device_index("input", self.input_device)
            output_idx = self._resolve_device_index("output", self.output_device)

            self._input_stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=self.channels,
                dtype=self.dtype,
                device=input_idx,
                callback=self._input_callback,
            )
            self._output_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=self.channels,
                dtype=self.dtype,
                device=output_idx,
                callback=self._output_callback,
            )

            self._input_stream.start()
            self._output_stream.start()
            self._running = True
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        if self._input_stream is not None:
            try:
                self._input_stream.stop()
            finally:
                self._input_stream.close()
                self._input_stream = None

        if self._output_stream is not None:
            try:
                self._output_stream.stop()
            finally:
                self._output_stream.close()
                self._output_stream = None

        self._running = False

    def get_stats(self) -> dict[str, int]:
        return {
            "input_overflows": self.input_overflows,
            "output_underflows": self.output_underflows,
            "callback_errors": self.callback_errors,
            "queued_frames": self._queue.qsize(),
        }

    def reset_stats(self) -> None:
        self.input_overflows = 0
        self.output_underflows = 0
        self.callback_errors = 0
