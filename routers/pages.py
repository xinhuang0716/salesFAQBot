from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(tags=["Pages"])

BASE_DIR = Path(__file__).resolve().parents[1]


@router.get("/")
async def landingpage() -> FileResponse:
    """Serve the landing page."""
    return FileResponse(BASE_DIR / "template" / "index.html")
