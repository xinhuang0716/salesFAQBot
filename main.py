"""FastAPI application entry point."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from core.aoai import AOAIClient
from core.dense_search import DenseSearcher
from core.embedder import Embedder
from infra.database import initialize_database
from infra.indexer import build_index_data
from infra.settings import get_settings
from routers.health import router as health_router
from routers.pages import router as pages_router
from routers.rag_response import router as rag_response_router
from routers.retrieve import router as retrieve_router

BASE_DIR = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle."""
    # Startup
    logger.info("Application startup initiated.")

    settings = get_settings()
    app.state.settings = settings

    # Shared async HTTP client.
    app.state.http_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))

    # Initialize local embedding and retrieval resources.
    app.state.embedder = Embedder()
    app.state.client = initialize_database(
        collection_name="FAQ",
        build_index_data=lambda: build_index_data(app.state.embedder),
    )

    # Initialize the DenseSearcher for local dense retrieval.
    app.state.dense_searcher = DenseSearcher(
        client=app.state.client,
        embedder=app.state.embedder,
    )

    # Initialize the AOAI client.
    app.state.aoai_client = AOAIClient(
        endpoint=settings.env.azure_openai_endpoint,
        api_key=settings.env.azure_openai_api_key.get_secret_value(),
        model=settings.env.azure_openai_deployment,
        http_client=app.state.http_client,
    )

    logger.info("Application startup completed.")

    yield

    # Shutdown
    logger.info("Application shutdown initiated.")

    app.state.client.close()
    await app.state.http_client.aclose()

    logger.info("Application shutdown completed.")


# Create the FastAPI application instance.
app = FastAPI(
    title="Sales FAQ Bot",
    version="2.0.0",
    description="A FastAPI application for answering sales-related FAQs.",
    lifespan=lifespan,
)

# Mount static files.
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Include routers.
app.include_router(health_router)
app.include_router(pages_router)
app.include_router(retrieve_router)
app.include_router(rag_response_router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info", access_log=True, reload=True)
