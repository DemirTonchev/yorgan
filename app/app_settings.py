from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_ignore_empty=True,  # use the default values instead of emtpy env var in case it is not configured in .env
    )
    version: str = "0.1.0"


settings = Settings()
