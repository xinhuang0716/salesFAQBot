from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/")
async def health() -> dict[str, str]:
    """Return the health status of the service."""
    return {"status": "ok"}
