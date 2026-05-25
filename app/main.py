from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.api.routes import router as api_router
from app.config import get_settings

settings = get_settings()
app = FastAPI(title="PDF QA Demo")
app.include_router(api_router, prefix="/api")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    html_path = settings.project_root / "app" / "web" / "index.html"
    return html_path.read_text(encoding="utf-8")
