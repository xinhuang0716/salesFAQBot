import asyncio

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

router = APIRouter(prefix="/response", tags=["RAG Response"])


class QueryRequest(BaseModel):
    """User query request body."""

    message: str = Field(min_length=1, max_length=256)


class QueryResponse(BaseModel):
    """RAG answer with its retrieved document references."""

    response: str
    reference: list[dict]
    status: str = "success"


@router.post("", response_model=QueryResponse)
async def rag_response(request: Request, body: QueryRequest) -> QueryResponse:
    """Retrieve dense-search documents and generate an AOAI response."""
    try:
        settings = request.app.state.settings

        documents = await asyncio.to_thread(
            request.app.state.dense_searcher.search,
            body.message,
            settings.retrieval.top_k,
            settings.retrieval.score_threshold,
        )

        response = await request.app.state.aoai_client.rag_response(
            query=body.message,
            top_k_docs=[
                f"[主題]{document['topic']}\n[子題]{document['subtype']}\n[內容]{document['relevance']}"
                for document in documents
            ]
        )

        reference = [
            {
                "id": document.get("id"),
                "source": document.get("source"),
                "topic": document.get("topic"),
                "subtype": document.get("subtype"),
                "score": document.get("score"),
            }
            for document in documents
        ]

        return QueryResponse(response=response, reference=reference)

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate a response.",
        ) from error
