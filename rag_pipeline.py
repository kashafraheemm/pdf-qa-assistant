import os
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import anthropic

# =========================
# EMBEDDINGS (singleton)
# =========================
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings


# =========================
# LOAD + SPLIT PDF
# =========================
def load_and_split(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,      # smaller = safer context budget
        chunk_overlap=150
    )
    chunks = splitter.split_documents(pages)
    return chunks


# =========================
# VECTOR STORE (with persistence)
# =========================
def create_or_load_vectorstore(chunks, index_path="faiss_index"):
    if os.path.exists(index_path):
        return FAISS.load_local(
            index_path,
            get_embeddings(),
            allow_dangerous_deserialization=True
        )
    vs = FAISS.from_documents(chunks, get_embeddings())
    vs.save_local(index_path)
    return vs


def reset_vectorstore(index_path="faiss_index"):
    """Call this when a new PDF is uploaded."""
    import shutil
    if os.path.exists(index_path):
        shutil.rmtree(index_path)


# =========================
# QA ENGINE (Claude API)
# =========================
def ask_question(vectorstore, query):
    # Retrieve + score-filter
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=5)

    # FAISS returns L2 distance — lower = more similar. Filter weak matches.
    SCORE_THRESHOLD = 1.2
    filtered = [(doc, score) for doc, score in docs_with_scores if score < SCORE_THRESHOLD]

    # Fallback: if everything is filtered out, keep top 3
    if not filtered:
        filtered = docs_with_scores[:3]

    # Build context with a hard character cap so the prompt stays clean
    MAX_CONTEXT_CHARS = 3000
    context_parts = []
    total = 0
    for doc, _ in filtered:
        chunk = doc.page_content.strip()
        if total + len(chunk) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(chunk)
        total += len(chunk)

    context = "\n\n---\n\n".join(context_parts)
    source_docs = [doc for doc, _ in filtered[:len(context_parts)]]

    # Claude API call
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",   # fastest + cheapest Claude model
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": f"""You are a precise document assistant. Answer ONLY from the context below.
If the answer is not in the context, say: "This information is not found in the document."
Do not make up or infer beyond what is explicitly stated.

Context:
{context}

Question: {query}

Answer concisely and clearly:"""
            }
        ]
    )

    answer = message.content[0].text
    return answer, source_docs
