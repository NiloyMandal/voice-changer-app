from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    sample_rate: int = 16000
    channels: int = 1
    block_size: int = 256
    input_device: int | None = None
    output_device: int | None = None
    inference_backend: str = "torch"  # torch | onnx

    model_config = SettingsConfigDict(env_prefix="VOICE_", extra="ignore")


settings = Settings()
