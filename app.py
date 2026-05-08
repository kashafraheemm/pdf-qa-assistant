import streamlit as st
import os
import tempfile
from rag_pipeline import load_and_split, create_or_load_vectorstore, ask_question, reset_vectorstore

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="PDF Q&A Assistant",
    page_icon="📄",
    layout="centered"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
    .main { max-width: 780px; margin: auto; }
    .stTextInput > div > div > input { border-radius: 8px; }
    .answer-box {
        background: #f0f4ff;
        border-left: 4px solid #4A6CF7;
        border-radius: 6px;
        padding: 16px 20px;
        margin-top: 12px;
        font-size: 15px;
        line-height: 1.7;
    }
    .source-box {
        background: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 13px;
        color: #555;
        margin-top: 6px;
    }
    .tag { 
        background: #4A6CF7; 
        color: white; 
        font-size: 11px; 
        padding: 2px 8px; 
        border-radius: 12px; 
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title("📄 PDF Q&A Assistant")
st.caption("Upload a PDF and ask questions — powered by Claude + local embeddings")

# =========================
# API KEY INPUT
# =========================
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Get your free key at console.anthropic.com"
    )
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key
        st.success("API key set ✓")

    st.divider()
    st.markdown("**How it works**")
    st.markdown("""
1. Upload your PDF  
2. It gets split into chunks  
3. Chunks are embedded locally  
4. Your question finds the most relevant chunks  
5. Claude answers using only those chunks
    """)
    st.divider()
    st.markdown("Built with `LangChain` · `FAISS` · `Claude Haiku`")

# =========================
# SESSION STATE
# =========================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "history" not in st.session_state:
    st.session_state.history = []
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    if uploaded_file.name != st.session_state.current_file:
        # New file uploaded — reset everything
        reset_vectorstore()
        st.session_state.vectorstore = None
        st.session_state.history = []
        st.session_state.current_file = uploaded_file.name

        with st.spinner("Processing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            chunks = load_and_split(tmp_path)
            os.unlink(tmp_path)  # clean up temp file

            st.session_state.vectorstore = create_or_load_vectorstore(chunks)

        st.success(f"✅ Ready — {len(chunks)} chunks indexed")

# =========================
# Q&A INTERFACE
# =========================
if st.session_state.vectorstore:
    st.divider()

    with st.form("qa_form", clear_on_submit=True):
        query = st.text_input("Ask a question about the document:", placeholder="e.g. What is the main conclusion?")
        submitted = st.form_submit_button("Ask →", use_container_width=True)

    if submitted and query.strip():
        if not os.environ.get("ANTHROPIC_API_KEY"):
            st.error("Please enter your Anthropic API key in the sidebar first.")
        else:
            with st.spinner("Thinking..."):
                try:
                    answer, source_docs = ask_question(st.session_state.vectorstore, query)
                    st.session_state.history.append({
                        "question": query,
                        "answer": answer,
                        "sources": source_docs
                    })
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Show history (newest first)
    for item in reversed(st.session_state.history):
        st.markdown(f"**Q: {item['question']}**")
        st.markdown(f"<div class='answer-box'>{item['answer']}</div>", unsafe_allow_html=True)

        with st.expander("View source chunks"):
            for i, doc in enumerate(item["sources"], 1):
                page = doc.metadata.get("page", "?")
                st.markdown(
                    f"<div class='source-box'><span class='tag'>Chunk {i}</span>"
                    f"<span class='tag'>Page {page + 1}</span><br><br>{doc.page_content[:400]}...</div>",
                    unsafe_allow_html=True
                )
        st.divider()

elif not uploaded_file:
    st.info("👆 Upload a PDF to get started")
