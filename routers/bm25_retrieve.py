from fastapi import APIRouter, HTTPException, Request, status

from routers.schemas import ApiResponse, QueryRequest

router = APIRouter(prefix="/bm25-retrieve", tags=["BM25 Retrieve"])


@router.post("/")
def bm25_retrieve(request: Request, body: QueryRequest) -> ApiResponse[list[dict]]:
    """Return BM25 retrieval results for a user query."""
    try:
        top_k = request.app.state.settings.retrieval.top_k
        results = request.app.state.bm25.search(body.message, top_k)
        return ApiResponse(data=results)

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve BM25 documents.",
        ) from error
