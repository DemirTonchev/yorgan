from enum import StrEnum
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()

# ParseServiceOptions = StrEnum("ParseServiceOptions", "gemini gpt landingai")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_ignore_empty=True,  # use the default values instead of emtpy env var in case it is not configured in .env
    )
    parse_llm: str = "gemini-2.5-flash"
    # parse_llms: dict[str, str] = "gemini-2.5-flash"
    # gpt_parse_llm: Optional[str] = "gpt-4.1"
    # extract_llm: str = "gemini-2.5-flash"

    # parse_service_backend: str = "landingai"


settings = Settings()
