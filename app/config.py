from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    project_root: Path
    artifacts_root: Path
    current_doc_id: str
    current_doc_dir: Path
    db_path: Path
    ocr_region: str
    hunyuan_region: str
    table_mode: str
    table_policy: str
    render_dpi: int
    render_jpg_quality: int
    text_chunk_chars: int
    text_overlap_chars: int
    snippet_chars: int
    embed_max_chars: int
    embedding_model: str
    chat_model: str



def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[1]
    artifacts_root = project_root / "artifacts"
    current_doc_id = "current"
    current_doc_dir = artifacts_root / current_doc_id
    return Settings(
        project_root=project_root,
        artifacts_root=artifacts_root,
        current_doc_id=current_doc_id,
        current_doc_dir=current_doc_dir,
        db_path=artifacts_root / "app_rag.sqlite",
        ocr_region=os.getenv("TENCENT_OCR_REGION", "ap-guangzhou"),
        hunyuan_region=os.getenv("TENCENT_HUNYUAN_REGION", ""),
        table_mode=os.getenv("TENCENT_TABLE_MODE", "table"),
        table_policy=os.getenv("TENCENT_TABLE_POLICY", "auto"),
        render_dpi=int(os.getenv("PDF_RENDER_DPI", "200")),
        render_jpg_quality=int(os.getenv("PDF_RENDER_JPG_QUALITY", "85")),
        text_chunk_chars=int(os.getenv("TEXT_CHUNK_CHARS", "700")),
        text_overlap_chars=int(os.getenv("TEXT_OVERLAP_CHARS", "120")),
        snippet_chars=int(os.getenv("SNIPPET_CHARS", "240")),
        embed_max_chars=int(os.getenv("EMBED_MAX_CHARS", "1500")),
        embedding_model=os.getenv("EMBEDDING_MODEL", "hunyuan-embedding"),
        chat_model=os.getenv("CHAT_MODEL", "hunyuan-turbos-latest"),
    )
