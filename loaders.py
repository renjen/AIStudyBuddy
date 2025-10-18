import io
from typing import List

from langchain_core.documents import Document
from pypdf import PdfReader


def _pdf_to_docs(file_bytes: bytes, source_name: str) -> List[Document]:
    """Extract text from each page of a PDF and return as Documents."""
    docs = []
    pdf = PdfReader(io.BytesIO(file_bytes))
    for i, page in enumerate(pdf.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(
                Document(page_content=text, metadata={"source": source_name, "page": i})
            )
    return docs


def _text_to_docs(text: str, source_name: str) -> List[Document]:
    """Wrap plain text as a single Document."""
    return [Document(page_content=text, metadata={"source": source_name, "page": 1})]


def load_files_to_documents(uploaded_files) -> List[Document]:
    """Load multiple uploaded files (PDF or TXT/MD) into a list of Documents."""
    all_docs: List[Document] = []
    for uf in uploaded_files:
        name = getattr(uf, "name", "uploaded")
        data = uf.read()
        if name.lower().endswith(".pdf"):
            all_docs.extend(_pdf_to_docs(data, name))
        else:
            text = data.decode("utf-8", errors="ignore")
            all_docs.extend(_text_to_docs(text, name))
    return all_docs
