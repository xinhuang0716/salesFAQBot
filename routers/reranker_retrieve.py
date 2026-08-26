import asyncio

from fastapi import APIRouter, HTTPException, Request, status

from routers.schemas import ApiResponse, QueryRequest

router = APIRouter(prefix="/reranker", tags=["Reranker"])


@router.post("/")
async def rerank(request: Request, body: QueryRequest) -> ApiResponse[list[dict]]:
    """Retrieve dense candidates and return their reranked order."""
    try:
        settings = request.app.state.settings

        dense_documents = await asyncio.to_thread(
            request.app.state.dense_searcher.search,
            body.message,
            settings.retrieval.top_k,
            settings.retrieval.score_threshold,
        )

        results = await asyncio.to_thread(
            request.app.state.reranker.rerank,
            body.message,
            dense_documents,
            settings.reranker.top_k,
            settings.reranker.score_threshold,
        )
        return ApiResponse(data=results)

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to rerank documents.",
        ) from error
