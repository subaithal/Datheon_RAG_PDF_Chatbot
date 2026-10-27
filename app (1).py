import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG PDF Chatbot")
st.write("Upload a PDF and ask questions about its content.")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_pdf = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"]
)

if uploaded_pdf:
    reader = PdfReader(uploaded_pdf)
    chunks = []
    chunk_sources = []

    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()

        # Split each page into overlapping chunks
        size = 700
        overlap = 100

        for start in range(0, len(text), size - overlap):
            part = text[start:start + size].strip()
            if part:
                chunks.append(part)
                chunk_sources.append(page_no)

    if not chunks:
        st.error("No readable text found in this PDF.")
        st.stop()

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    st.success(f"PDF ready! Pages: {len(reader.pages)}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about your PDF...")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        q_vector = model.encode(
            [question],
            normalize_embeddings=True
        ).astype("float32")

        k = min(3, len(chunks))
        scores, indices = index.search(q_vector, k)

        answer_parts = []
        source_pages = []

        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score > 0.15:
                answer_parts.append(chunks[idx])
                source_pages.append(chunk_sources[idx])

        with st.chat_message("assistant"):
            if answer_parts:
                st.markdown(
                    "**Relevant information from your PDF:**\n\n"
                    + "\n\n---\n\n".join(answer_parts)
                )
                pages = sorted(set(source_pages))
                st.caption(
                    "Source page(s): "
                    + ", ".join(map(str, pages))
                )
                response = (
                    "**Relevant information from your PDF:**\n\n"
                    + "\n\n---\n\n".join(answer_parts)
                    + "\n\nSource page(s): "
                    + ", ".join(map(str, pages))
                )
            else:
                response = (
                    "I couldn't find relevant information "
                    "in the uploaded PDF."
                )
                st.write(response)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
else:
    st.info("Upload a PDF to begin chatting with your document.")
