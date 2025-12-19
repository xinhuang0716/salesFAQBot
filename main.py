import uvicorn
import os, sys, logging, yaml
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from core.init.builder import initialize
from core.retrieve.dense_search import DenseSearcher
from core.response.geminiAPI import RAGresponse
from utils.corpus import build_text


# Configuration
with open("./config/config.yaml", "r", encoding="utf-8") as f: config: dict = yaml.safe_load(f)
SERVER_CONFIG: dict = config["server"]
DB_CONFIG: dict = config["db"]
EMBEDDER_CONFIG: dict = config["embedder"]
RETRIEVE_CONFIG : dict = config["retriever"]


# Logging setup
os.makedirs("./logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join("./logs", "app.log"), mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# Lifespan context manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):

    # Startup
    global client, embedder, reranker
    client, embedder = initialize(collection_name=DB_CONFIG["collection_name"], repo=EMBEDDER_CONFIG["repo"])
    logger.info("Starting Sales FAQ Bot v1.0.0")
    logger.info("Server running on 0.0.0.0:8010")
    yield

    # Shutdown
    client.close()
    logger.info("Shutting down Sales FAQ Bot...")


# Initialize FastAPI app
app = FastAPI(
    title="Sales FAQ Bot",
    version="1.0.0",
    description="A FastAPI application for answering sales-related FAQs.",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory="./static"), name="static")


# CORS middleware with configurable origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=SERVER_CONFIG["cors_origins"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    max_age=SERVER_CONFIG["cors_max_age"],
)


# Request/Response models with validation
class QueryRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=SERVER_CONFIG["max_message_length"],
        description="User query message"
    )


class QueryResponse(BaseModel):
    response: str = Field(..., description="Bot response to the query")
    status: str = Field(default="success")


# Unified exception handler
@app.exception_handler(Exception)
async def global_exception_handler(exc: Exception):
    if isinstance(exc, HTTPException):
        logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal error occurred. Please try again later."},
    )


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response


@app.get("/", tags=["Root"])
async def root():
    return FileResponse("./template/index.html", media_type="text/html")


@app.post("/retrieveDocs", response_model=QueryResponse, tags=["Query"])
async def retrieve_documents(request: QueryRequest):
    """
    Retrieve top-k relevant documents for the user query.

    Args:
        request (QueryRequest): QueryRequest with user message

    Raises:
        HTTPException: If processing fails

    Returns:
        QueryResponse with retrieved documents
    """

    try:
        logger.info(f"Processing query: {request.message[:50]}...")
        
        searcher: DenseSearcher = DenseSearcher(client=client, embedder=embedder)
        result: list[dict] = searcher.search(query=request.message, k=RETRIEVE_CONFIG["top_k"], score=RETRIEVE_CONFIG["score_threshold"])
        top_k_docs: list[str] = [build_text(doc) for doc in result]

        logger.info("Query processed successfully")
        return QueryResponse(response=str(top_k_docs), status="success")

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process query. Please try again.",
        )


@app.post("/response", response_model=QueryResponse, tags=["Query"])
async def get_response(request: QueryRequest):
    """
    Process user query and return FAQ bot response using RAG.

    Args:
        request: QueryRequest with user message

    Raises:
        HTTPException: If processing fails

    Returns:
        QueryResponse with bot answer
    """

    try:
        logger.info(f"Processing query: {request.message[:50]}...")

        searcher: DenseSearcher = DenseSearcher(client=client, embedder=embedder)
        result: list[dict] = searcher.search(query=request.message, k=RETRIEVE_CONFIG["top_k"], score=RETRIEVE_CONFIG["score_threshold"])
        top_k_docs: list[str] = [build_text(doc) for doc in result]

        # Call RAG response function
        response_text: str = RAGresponse(
            query=request.message,
            top_k_docs=top_k_docs
        )

        logger.info("Query processed successfully")
        return QueryResponse(response=response_text, status="success")

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process query. Please try again.",
        )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8010,
        log_level="info",
        access_log=True,
    )
