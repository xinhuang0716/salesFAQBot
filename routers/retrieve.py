from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

router = APIRouter(prefix="/retrieve", tags=["Retrieve"])


class QueryRequest(BaseModel):
    """User query request body."""

    message: str = Field(min_length=1, max_length=256, description="The user query to search for.")


@router.post("/")
def retrieve(request: Request, body: QueryRequest) -> list[dict]:
    """Return dense-search results for a user query."""
    try:
        settings = request.app.state.settings

        return request.app.state.dense_searcher.search(
            query=body.message,
            top_k=settings.retrieval.top_k,
            score_threshold=settings.retrieval.score_threshold,
        )

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documents.",
        ) from error
