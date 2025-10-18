import os
import uuid
import streamlit as st
from rag import RAGPipeline, ensure_chroma_persist_dir

st.set_page_config(page_title="AI Study Buddy", page_icon="📚", layout="wide")

st.title("📚 AI Study Buddy — Chat with your notes")

with st.sidebar:
    st.header("Settings")
    openai_api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    embed_model = st.selectbox("Embedding model", ["text-embedding-3-small", "text-embedding-3-large"])
    gen_model = st.selectbox("Chat model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"])
    chunk_size = st.slider("Chunk size", 300, 2000, 1000, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, 10)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    st.markdown("---")
    if st.button("🧹 Clear my collection"):
        if "collection_id" in st.session_state:
            RAGPipeline.delete_collection(st.session_state["collection_id"])
            st.success("Cleared!")

ensure_chroma_persist_dir()

if "collection_id" not in st.session_state:
    st.session_state["collection_id"] = f"user-{uuid.uuid4().hex[:8]}"

uploaded_files = st.file_uploader("Upload PDFs or text files", type=["pdf", "txt", "md"], accept_multiple_files=True)

rag = None

if st.button("📥 Index files"):
    if not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    elif not uploaded_files:
        st.warning("Please upload at least one file.")
    else:
        with st.spinner("Indexing docs..."):
            rag = RAGPipeline(
                openai_api_key=openai_api_key,
                collection_name=st.session_state["collection_id"],
                embed_model=embed_model,
                gen_model=gen_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                temperature=temperature,
            )
            stats = rag.index_streamlit(uploaded_files)
            st.success(f"Indexed {stats['docs']} docs ({stats['chunks']} chunks)")

query = st.text_input("Ask a question about your files...", placeholder="e.g., Summarize Chapter 3 with citations")

if st.button("🔎 Ask"):
    if not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    elif not query.strip():
        st.warning("Type a question first.")
    else:
        rag = rag or RAGPipeline(
            openai_api_key=openai_api_key,
            collection_name=st.session_state["collection_id"],
            embed_model=embed_model,
            gen_model=gen_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            temperature=temperature,
        )
        with st.spinner("Thinking..."):
            answer, sources = rag.answer(query)
        st.markdown("### Answer")
        st.write(answer)
        if sources:
            st.markdown("### Sources")
            for i, s in enumerate(sources, 1):
                st.markdown(f"**{i}.** `{s['source']}` — p.{s.get('page', '?')}  (score: {s['score']:.3f})") 

st.caption("💡 Tip: Each session has a random collection ID. Reuse it to keep chatting with the same notes.")
