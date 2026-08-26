from fastapi import APIRouter

from routers.schemas import ApiResponse

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/")
async def health() -> ApiResponse[dict[str, str]]:
    """Return the health status of the service."""
    return ApiResponse(data={"service": "ok"})
