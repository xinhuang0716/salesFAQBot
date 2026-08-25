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


class EmbeddingSettings(BaseModel):
    """Local embedding model settings."""

    type: str = Field(pattern="^(aoai|sentence_transformer)$")
    repo: str | None = None


class RetrievalSettings(BaseModel):
    """Local document retrieval settings."""

    top_k: int = Field(gt=0)
    score_threshold: float = Field(ge=0.0, le=1.0)


class RerankerSettings(BaseModel):
    """Optional local reranking settings."""

    apply: bool = False
    type: str = Field(default="sentence_transformer")
    repo: str | None = None
    top_k: int = Field(gt=0)
    score_threshold: float = Field(ge=0.0, le=1.0)


class Settings(BaseModel):
    """Complete application settings."""

    env: EnvSettings
    embedding: EmbeddingSettings
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
        embedding=EmbeddingSettings.model_validate(config["embedding"]),
        retrieval=RetrievalSettings.model_validate(config["retrieval"]),
        reranker=RerankerSettings.model_validate(config["reranker"]),
    )
