import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr

BASE_DIR = Path(__file__).resolve().parents[1]
YAML_PATH = BASE_DIR / "config" / "config.yaml"
ENV_PATH = BASE_DIR / "config" / ".env"


class EnvSettings(BaseModel):
    """Sensitive settings loaded from config/.env."""

    azure_openai_endpoint: str = Field(min_length=1)
    azure_openai_api_key: SecretStr = Field(min_length=1)
    azure_openai_deployment: str = Field(min_length=1)


class RetrievalSettings(BaseModel):
    """Local document retrieval settings."""

    top_k: int = Field(gt=0)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    hybrid_top_k: int = Field(gt=0)
    rrf_k: int = Field(default=60, gt=0)


class RerankerSettings(BaseModel):
    """Optional local reranking settings."""

    top_k: int = Field(gt=0)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class Settings(BaseModel):
    """Complete application settings."""

    env: EnvSettings
    retrieval: RetrievalSettings
    reranker: RerankerSettings


@lru_cache
def get_settings() -> Settings:
    """Load and validate YAML and .env configuration."""
    load_dotenv(ENV_PATH, override=True)

    with YAML_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    return Settings(
        env=EnvSettings.model_validate(
            {
                "azure_openai_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
                "azure_openai_api_key": os.getenv("AZURE_OPENAI_API_KEY"),
                "azure_openai_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            }
        ),
        retrieval=RetrievalSettings.model_validate(config["retrieval"]),
        reranker=RerankerSettings.model_validate(config["reranker"]),
    )
