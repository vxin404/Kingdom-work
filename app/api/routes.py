from pathlib import Path
import shutil
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import get_settings
from app.services.pipeline import ask_question, process_pdf

router = APIRouter()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    topk: int = Field(default=5, ge=1, le=10)


@router.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="please upload a PDF file")

    settings = get_settings()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        return process_pdf(tmp_path, settings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/ask")
def ask(req: AskRequest):
    settings = get_settings()
    try:
        return ask_question(req.question, settings, topk=req.topk)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
