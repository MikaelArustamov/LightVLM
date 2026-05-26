"""Parse PDFs, images, text files."""

import base64
from typing import Dict, Any, Literal
from pathlib import Path


class DocumentParser:
    def __init__(self, llm):
        self.llm = llm

    def parse(self, content: bytes, filename: str, mode: Literal["auto", "text", "ocr"] = "auto"):
        ext = Path(filename or 'unknown.pdf').suffix.lower()
        if ext == '.pdf':
            return self._parse_pdf(content, mode)
        elif ext in ('.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp'):
            return self._parse_image(content)
        elif ext in ('.txt', '.md', '.py', '.js', '.html', '.css', '.json'):
            return self._parse_text(content, filename)
        else:
            return {"type": "error", "error": f"Unsupported: {ext}"}

    def _parse_pdf(self, content: bytes, mode: str):
        try:
            import fitz
        except ImportError:
            raise ImportError("pip install pymupdf")

        doc = fitz.open(stream=content, filetype="pdf")
        page_count = len(doc)  # ← СОХРАНИТЬ ДО ЗАКРЫТИЯ
        text_parts = []
        scanned_pages = 0

        for page_num in range(page_count):
            page = doc[page_num]
            page_text = page.get_text()

            if page_text and page_text.strip():
                text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
            else:
                scanned_pages += 1
                if mode in ("auto", "ocr"):
                    text_parts.append(self._ocr_pdf_page(doc, page_num))

        doc.close()  # ← ЗАКРЫТЬ ПОСЛЕ ЦИКЛА

        full_text = "\n\n".join(text_parts)
        max_chars = 12000
        truncated = len(full_text) > max_chars

        return {
            "type": "pdf",
            "text": full_text[:max_chars] + ("...[truncated]" if truncated else ""),
            "pages": page_count,  # ← ИСПОЛЬЗОВАТЬ СОХРАНЁННОЕ ЗНАЧЕНИЕ
            "scanned_pages": scanned_pages,
            "truncated": truncated,
            "total_chars": len(full_text)
        }
    def _ocr_pdf_page(self, doc, page_num: int):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        b64 = base64.b64encode(img_bytes).decode()

        result = self.llm.vision_chat(
            b64,
            "Extract all text from this document page. Preserve paragraphs. Output only text."
        )

        return f"--- Page {page_num + 1} (OCR) ---\n{result}"

    def _parse_image(self, content: bytes):
        b64 = base64.b64encode(content).decode()
        result = self.llm.vision_chat(
            b64,
            "Extract all text from this image. If document, preserve layout. If tables, use markdown. Output only text."
        )
        return {"type": "image_ocr", "text": result, "image_size": len(content)}

    def _parse_text(self, content: bytes, filename: str):
        text = content.decode('utf-8', errors='replace')
        max_chars = 12000
        return {
            "type": "text",
            "text": text[:max_chars] + ("...[truncated]" if len(text) > max_chars else ""),
            "filename": filename,
            "total_chars": len(text),
            "truncated": len(text) > max_chars
        }

    def to_chat_messages(self, parse_result: Dict[str, Any]) -> str:
        if parse_result.get("type") == "error":
            return f"Error: {parse_result.get('error')}"

        text = parse_result.get("text", "")
        meta = []

        if "pages" in parse_result:
            meta.append(f"PDF, {parse_result['pages']} pages")
        if parse_result.get("scanned_pages"):
            meta.append(f"{parse_result['scanned_pages']} pages OCR'd")
        if parse_result.get("truncated"):
            meta.append("truncated")

        header = f"[Document: {', '.join(meta)}]\n\n" if meta else ""
        return header + text