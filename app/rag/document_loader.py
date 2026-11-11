from pathlib import Path
import fitz
from docx import Document


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".docx"
}


def load_pdf(file_path: Path) -> str:
    text_parts = []

    document = fitz.open(file_path)

    for page in document:
        page_text = page.get_text()

        if page_text.strip():
            text_parts.append(page_text)

    document.close()

    return "\n".join(text_parts)


def load_txt(file_path: Path) -> str:
    return file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )


def load_docx(file_path: Path) -> str:
    document = Document(file_path)

    paragraphs = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            paragraphs.append(paragraph.text)

    return "\n".join(paragraphs)


def load_document(file_path: Path) -> str:
    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return load_pdf(file_path)

    if extension == ".txt":
        return load_txt(file_path)

    if extension == ".docx":
        return load_docx(file_path)

    raise ValueError(
        f"Unsupported file type: {extension}. "
        f"Supported types: PDF, TXT, DOCX."
    )