import numpy as np

from app.inference.engine import InferenceEngine


class DummyModel:
    def __call__(self, _tensor):
        raise RuntimeError("Not expected to execute in this unit test")


class DummyOnnxSession:
    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]


def test_profile_model_selection_and_readiness() -> None:
    engine = InferenceEngine(backend="torch", torch_model_path=None)

    # Simulate loaded models without requiring real weight files.
    engine._torch_models["default"] = DummyModel()  # type: ignore[attr-defined]
    engine._torch_models["deep"] = DummyModel()  # type: ignore[attr-defined]

    assert engine.is_ready()
    assert engine.is_ready("deep")
    assert engine.is_ready("chipmunk")


def test_status_exposes_loaded_profile_keys() -> None:
    engine = InferenceEngine(backend="onnx", onnx_model_path=None)

    engine._onnx_sessions["default"] = DummyOnnxSession()  # type: ignore[attr-defined]
    engine._onnx_sessions["robot"] = DummyOnnxSession()  # type: ignore[attr-defined]

    status = engine.status()
    assert "loaded_onnx_profiles" in status
    assert "default" in status["loaded_onnx_profiles"]
    assert "robot" in status["loaded_onnx_profiles"]


def test_infer_passthrough_when_no_model_loaded() -> None:
    frame = np.zeros((128, 1), dtype=np.float32)
    engine = InferenceEngine(backend="torch", torch_model_path=None)

    out = engine.infer(frame, profile="robot")

    assert out.shape == frame.shape
    assert out.dtype == frame.dtype
