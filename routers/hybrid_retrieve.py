import asyncio

from fastapi import APIRouter, HTTPException, Request, status

from routers.schemas import ApiResponse, QueryRequest

router = APIRouter(prefix="/hybrid-retrieve", tags=["Hybrid Retrieve"])


@router.post("/")
async def hybrid_retrieve(request: Request, body: QueryRequest) -> ApiResponse[list[dict]]:
    """Return Dense and BM25 results fused with RRF."""
    try:
        settings = request.app.state.settings

        results = await asyncio.to_thread(
            request.app.state.hybrid_searcher.search,
            body.message,
            settings.retrieval.top_k,
            settings.retrieval.score_threshold,
            settings.retrieval.hybrid_top_k,
        )
        return ApiResponse(data=results)

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve hybrid documents.",
        ) from error
