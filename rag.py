import os
from typing import List, Dict, Tuple

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from loaders import load_files_to_documents

CHROMA_DIR = os.getenv("CHROMA_DIR", ".chroma")


def ensure_chroma_persist_dir():
    os.makedirs(CHROMA_DIR, exist_ok=True)


class RAGPipeline:
    """Manages embedding, retrieval, and answering."""

    def __init__(
        self,
        openai_api_key: str,
        collection_name: str,
        embed_model: str = "text-embedding-3-small",
        gen_model: str = "gpt-4o-mini",
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        temperature: float = 0.2,
    ):
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(model=embed_model, api_key=openai_api_key)
        self.llm = ChatOpenAI(model=gen_model, temperature=temperature, api_key=openai_api_key)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", ", ", " "],
        )
        ensure_chroma_persist_dir()
        self.vs = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DIR,
        )

    @staticmethod
    def delete_collection(collection_name: str):
        """Deletes a Chroma collection by name."""
        vs = Chroma(
            collection_name=collection_name,
            embedding_function=OpenAIEmbeddings(),
            persist_directory=CHROMA_DIR,
        )
        vs.delete_collection()

    def index_streamlit(self, uploaded_files) -> Dict[str, int]:
        docs = load_files_to_documents(uploaded_files)
        return self._index_documents(docs)

    def _index_documents(self, docs: List[Document]) -> Dict[str, int]:
        chunks = self.splitter.split_documents(docs)
        self.vs.add_documents(chunks)
        self.vs.persist()
        return {"docs": len(docs), "chunks": len(chunks)}

    def answer(self, question: str, k: int = 4) -> Tuple[str, List[Dict]]:
        retriever = self.vs.as_retriever(search_kwargs={"k": k})
        docs: List[Document] = retriever.get_relevant_documents(question)

        context_blocks = []
        sources = []
        for d in docs:
            meta = d.metadata or {}
            src = meta.get("source", "unknown")
            page = meta.get("page", meta.get("page_number"))
            sources.append({"source": src, "page": page, "score": meta.get("relevance_score", 0)})
            context_blocks.append(f"From {src} p.{page}:\n{d.page_content}")

        system = "You are a helpful study assistant. Answer concisely and cite sources when possible."
        context = "\n\n---\n\n".join(context_blocks)
        user = f"Question: {question}\n\nUse the context above if relevant. If unsure, say so."

        messages = [("system", system), ("user", context), ("user", user)]
        resp = self.llm.invoke(messages)
        return resp.content, sources
