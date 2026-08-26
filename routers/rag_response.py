import asyncio

from fastapi import APIRouter, HTTPException, Request, status

from core.context import format_document
from routers.schemas import ApiResponse, QueryRequest, RagResponseData

router = APIRouter(prefix="/rag-response", tags=["RAG Response"])


@router.post("/", response_model=ApiResponse[RagResponseData])
async def rag_response(request: Request, body: QueryRequest) -> ApiResponse[RagResponseData]:
    """Generate a RAG response and references from hybrid-retrieved, reranked documents."""
    try:
        settings = request.app.state.settings

        hybrid_documents = await asyncio.to_thread(
            request.app.state.hybrid_searcher.search,
            body.message,
            settings.retrieval.top_k,
            settings.retrieval.score_threshold,
            settings.retrieval.hybrid_top_k,
        )

        reranked_documents = await asyncio.to_thread(
            request.app.state.reranker.rerank,
            body.message,
            hybrid_documents,
            settings.reranker.top_k,
            settings.reranker.score_threshold,
        )

        response = await request.app.state.aoai_client.rag_response(
            query=body.message,
            top_k_docs=[format_document(doc) for doc in reranked_documents],
        )

        return ApiResponse(
            data=RagResponseData(
                response=response,
                references=reranked_documents,
            )
        )

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate a response.",
        ) from error
