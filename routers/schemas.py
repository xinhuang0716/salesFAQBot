from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """User query request body."""

    message: str = Field(min_length=1, max_length=256)


class ApiResponse[T](BaseModel):
    """Unified API response envelope."""

    status: str = "success"
    data: T


class RagResponseData(BaseModel):
    """RAG response payload."""

    response: str
    references: list[dict[str, Any]]
