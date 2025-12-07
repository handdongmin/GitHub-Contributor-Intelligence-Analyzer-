from functools import lru_cache
from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings


load_dotenv()

class Settings(BaseSettings):
    github_token: Optional[str] = None
    summarizer_model: str = "kainois/ke-t5-small-ko-summarization"
    log_level: str = "INFO"

    class Config:
        # No prefix so .env의 GITHUB_TOKEN을 바로 읽어옵니다.
        env_prefix = ""
        case_sensitive = False

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
