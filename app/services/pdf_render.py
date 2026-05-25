from pathlib import Path
import fitz



def render_pdf_to_jpgs(pdf_path: Path, out_dir: Path, *, dpi: int, jpg_quality: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    zoom = max(1, dpi / 72)
    mat = fitz.Matrix(zoom, zoom)
    doc = fitz.open(pdf_path)
    try:
        image_paths: list[Path] = []
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            jpg_bytes = pix.tobytes("jpg", jpg_quality=jpg_quality)
            out_path = out_dir / f"page_{page_index + 1:03d}.jpg"
            out_path.write_bytes(jpg_bytes)
            image_paths.append(out_path)
        return image_paths
    finally:
        doc.close()
