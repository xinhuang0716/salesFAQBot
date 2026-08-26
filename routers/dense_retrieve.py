from fastapi import APIRouter, HTTPException, Request, status

from routers.schemas import ApiResponse, QueryRequest

router = APIRouter(prefix="/dense-retrieve", tags=["Dense Retrieve"])


@router.post("/")
def dense_retrieve(request: Request, body: QueryRequest) -> ApiResponse[list[dict]]:
    """Return dense-search results for a user query."""
    try:
        settings = request.app.state.settings

        results = request.app.state.dense_searcher.search(
            query=body.message,
            top_k=settings.retrieval.top_k,
            score_threshold=settings.retrieval.score_threshold,
        )
        return ApiResponse(data=results)

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documents.",
        ) from error
