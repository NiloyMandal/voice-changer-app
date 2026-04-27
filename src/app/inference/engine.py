from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - dependency/runtime dependent
    torch = None  # type: ignore[assignment]

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - dependency/runtime dependent
    ort = None  # type: ignore[assignment]


class InferenceEngine:
    """Real-time inference bridge for NumPy audio frames.

    Expected frame shape is (samples, channels) or (samples,).
    """

    def __init__(
        self,
        backend: str = "torch",
        torch_model_path: str | None = None,
        onnx_model_path: str | None = None,
        device: str | None = None,
        model_input_layout: str | None = None,
        normalize_mode: str | None = None,
        target_dbfs: float | None = None,
        restore_level: bool | None = None,
    ) -> None:
        if backend not in {"torch", "onnx"}:
            raise ValueError("backend must be 'torch' or 'onnx'")

        self.backend = backend
        self.torch_model_path = torch_model_path or os.getenv("VOICE_TORCH_MODEL_PATH")
        self.onnx_model_path = onnx_model_path or os.getenv("VOICE_ONNX_MODEL_PATH")
        self.model_input_layout = (model_input_layout or os.getenv("VOICE_MODEL_INPUT_LAYOUT", "bct")).lower()
        self.normalize_mode = (normalize_mode or os.getenv("VOICE_INFERENCE_NORMALIZE", "unit")).lower()
        self.target_dbfs = float(target_dbfs if target_dbfs is not None else os.getenv("VOICE_TARGET_DBFS", "-20.0"))
        restore = restore_level if restore_level is not None else os.getenv("VOICE_RESTORE_LEVEL", "true")
        self.restore_level = str(restore).lower() in {"1", "true", "yes", "on"}

        requested_device = device or os.getenv("VOICE_INFERENCE_DEVICE", "cuda")
        if requested_device == "cuda" and torch is not None and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = requested_device

        self._torch_model: Any = None
        self._onnx_session: Any = None

        if self.backend == "torch" and self.torch_model_path:
            self._load_torch_model(self.torch_model_path)
        elif self.backend == "onnx" and self.onnx_model_path:
            self._load_onnx_model(self.onnx_model_path)

    def _load_torch_model(self, model_path: str) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is not available but torch backend was requested")

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Torch model not found: {model_path}")

        try:
            self._torch_model = torch.jit.load(str(model_file), map_location=self.device)
        except Exception:
            # Fallback for standard nn.Module checkpoints.
            self._torch_model = torch.load(str(model_file), map_location=self.device)

        if hasattr(self._torch_model, "eval"):
            self._torch_model.eval()

    def _load_onnx_model(self, model_path: str) -> None:
        if ort is None:
            raise RuntimeError("onnxruntime is not available but onnx backend was requested")

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        providers = ["CPUExecutionProvider"]
        if self.device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._onnx_session = ort.InferenceSession(str(model_file), providers=providers)

    @staticmethod
    def _ensure_float32_frame(frame: np.ndarray) -> np.ndarray:
        if frame.dtype != np.float32:
            return frame.astype(np.float32)
        return frame

    @staticmethod
    def _rms(signal: np.ndarray) -> float:
        if signal.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(signal, dtype=np.float32), dtype=np.float32)))

    def _normalize_frame(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        if self.normalize_mode == "none":
            return frame, 1.0

        if self.normalize_mode == "unit":
            return np.clip(frame, -1.0, 1.0), 1.0

        if self.normalize_mode == "dbfs":
            rms = self._rms(frame)
            if rms <= 1e-8:
                return frame, 1.0
            target_amp = 10.0 ** (self.target_dbfs / 20.0)
            gain = target_amp / rms
            return np.clip(frame * gain, -1.0, 1.0), gain

        raise ValueError(
            "normalize_mode must be one of: 'none', 'unit', 'dbfs'"
        )

    def _denormalize_frame(self, frame: np.ndarray, gain: float) -> np.ndarray:
        if not self.restore_level:
            return frame
        if gain <= 1e-8 or gain == 1.0:
            return frame
        return np.clip(frame / gain, -1.0, 1.0)

    def _reshape_for_model(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
        """Map frame to configured model input layout and keep original shape."""
        original_shape = frame.shape

        if frame.ndim == 1:
            time_major = frame[:, None]  # [T, 1]
        else:
            time_major = frame  # [T, C]

        if self.model_input_layout == "bct":
            # [T, C] -> [1, C, T]
            model_input = np.transpose(time_major, (1, 0))[None, :, :]
        elif self.model_input_layout == "btc":
            # [T, C] -> [1, T, C]
            model_input = time_major[None, :, :]
        elif self.model_input_layout == "bt":
            # [T, C] -> [1, T] (first channel only)
            model_input = time_major[:, 0][None, :]
        else:
            raise ValueError(
                "model_input_layout must be one of: 'bct', 'btc', 'bt'"
            )

        return model_input.astype(np.float32, copy=False), original_shape

    def _restore_frame_shape(self, model_output: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
        """Map model output from configured layout back to original frame shape."""
        out = np.asarray(model_output)

        while out.ndim > 3:
            out = out[0]

        if self.model_input_layout in {"bct", "btc"} and out.ndim == 3:
            out = out[0]

        if self.model_input_layout == "bct":
            # [C, T] -> [T, C]
            if out.ndim == 1:
                out = out[None, :]
            restored = np.transpose(out, (1, 0))
        elif self.model_input_layout == "btc":
            # [T, C] already time-major
            if out.ndim == 1:
                restored = out[:, None]
            else:
                restored = out
        elif self.model_input_layout == "bt":
            # [T] -> [T, 1]
            if out.ndim == 2:
                out = out[0]
            restored = out[:, None]
        else:
            restored = out

        if len(original_shape) == 1:
            return restored[:, 0].astype(np.float32, copy=False)

        if restored.shape != original_shape:
            # Guard against slight shape mismatches.
            samples, channels = original_shape
            corrected = np.zeros((samples, channels), dtype=np.float32)
            s = min(samples, restored.shape[0])
            c = min(channels, restored.shape[1])
            corrected[:s, :c] = restored[:s, :c]
            return corrected

        return restored.astype(np.float32, copy=False)

    def _infer_torch(self, frame: np.ndarray) -> np.ndarray:
        if torch is None or self._torch_model is None:
            return frame

        model_input, original_shape = self._reshape_for_model(frame)

        with torch.inference_mode():
            tensor = torch.from_numpy(model_input)
            tensor = tensor.to(self.device)
            output = self._torch_model(tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]
            output_np = output.detach().to("cpu").float().numpy()

        return self._restore_frame_shape(output_np, original_shape)

    def _infer_onnx(self, frame: np.ndarray) -> np.ndarray:
        if ort is None or self._onnx_session is None:
            return frame

        model_input, original_shape = self._reshape_for_model(frame)

        input_name = self._onnx_session.get_inputs()[0].name
        outputs = self._onnx_session.run(None, {input_name: model_input})
        output_np = outputs[0]
        return self._restore_frame_shape(output_np, original_shape)

    def infer(self, frame: np.ndarray) -> np.ndarray:
        frame_f32 = self._ensure_float32_frame(frame)
        frame_norm, gain = self._normalize_frame(frame_f32)

        if self.backend == "torch":
            out = self._infer_torch(frame_norm)
            return self._denormalize_frame(out, gain)

        out = self._infer_onnx(frame_norm)
        return self._denormalize_frame(out, gain)

    def is_ready(self) -> bool:
        if self.backend == "torch":
            return self._torch_model is not None
        return self._onnx_session is not None

    def status(self) -> dict[str, Any]:
        providers: list[str] = []
        if self._onnx_session is not None:
            providers = list(self._onnx_session.get_providers())

        return {
            "backend": self.backend,
            "device": self.device,
            "ready": self.is_ready(),
            "torch_model_path": self.torch_model_path,
            "onnx_model_path": self.onnx_model_path,
            "model_input_layout": self.model_input_layout,
            "normalize_mode": self.normalize_mode,
            "onnx_providers": providers,
        }
