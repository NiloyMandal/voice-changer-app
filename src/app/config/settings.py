from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    sample_rate: int = 16000
    channels: int = 1
    block_size: int = 256
    input_device: int | None = None
    output_device: int | None = None
    inference_backend: str = "torch"  # torch | onnx
    inference_device: str = "cuda"  # cuda | cpu
    torch_model_path: str | None = None
    onnx_model_path: str | None = None
    model_input_layout: str = "bct"  # bct | btc | bt
    inference_normalize: str = "unit"  # none | unit | dbfs
    target_dbfs: float = -20.0
    restore_level: bool = True

    model_config = SettingsConfigDict(env_prefix="VOICE_", extra="ignore")


settings = Settings()
