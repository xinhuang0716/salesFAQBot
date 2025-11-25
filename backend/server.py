from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn
import logging
import sys
from typing import Optional
from datetime import datetime, timezone
import os
from contextlib import asynccontextmanager

# Configuration
class Config:
    APP_NAME = "Sales FAQ Bot API"
    VERSION = "1.0.0"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    MAX_MESSAGE_LENGTH = 1000
    LOG_DIR = "logs"

config = Config()

# Ensure log directory exists BEFORE configuring logging
os.makedirs(config.LOG_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(config.LOG_DIR, 'app.log'), mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Lifespan context manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"Starting {config.APP_NAME} v{config.VERSION}")
    logger.info(f"Server running on {config.HOST}:{config.PORT}")
    yield
    # Shutdown
    logger.info(f"Shutting down {config.APP_NAME}")

# Initialize FastAPI app
app = FastAPI(
    title=config.APP_NAME,
    version=config.VERSION,
    description="MVP FAQ Bot API with RAG capabilities",
    lifespan=lifespan
)

# CORS middleware with configurable origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    max_age=3600,
)

# Request/Response models with validation
class QueryRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=config.MAX_MESSAGE_LENGTH, description="User query message")
    
    @validator('message')
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError('Message cannot be empty or whitespace only')
        return v.strip()

class QueryResponse(BaseModel):
    response: str = Field(..., description="Bot response to the query")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = Field(default="success")

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str

class ErrorResponse(BaseModel):
    detail: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

# Custom exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            detail="An internal error occurred. Please try again later."
        ).dict()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(detail=exc.detail).dict()
    )

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for monitoring and load balancers"""
    return HealthResponse(
        status="healthy",
        version=config.VERSION,
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": config.APP_NAME,
        "version": config.VERSION,
        "status": "running",
        "endpoints": {
            "health": "/health",
            "response": "/response",
            "docs": "/docs"
        }
    }

@app.post("/response", response_model=QueryResponse, tags=["Query"])
async def get_response(request: QueryRequest):
    """
    Process user query and return FAQ bot response.
    
    Args:
        request: QueryRequest with user message
        
    Returns:
        QueryResponse with bot answer
        
    Raises:
        HTTPException: If processing fails
    """
    try:
        logger.info(f"Processing query: {request.message[:50]}...")
        
        response_text = f"You asked: {request.message}"
        
        logger.info("Query processed successfully")
        return QueryResponse(
            response=response_text,
            status="success"
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process query. Please try again."
        )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level="info",
        access_log=True
    )