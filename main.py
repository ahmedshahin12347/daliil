"""Minimal RAG service: upload documents, ask questions, get answers with sources."""

import io
import os
import re
import unicodedata
import uuid
from functools import lru_cache

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))
TOP_K = int(os.getenv("TOP_K", "4"))

LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY", "none")
LLM_MODEL = os.getenv("LLM_MODEL")

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages provided. "
    "Do not add rules, numbers, deadlines or conditions that are not written in the passages. "
    "If the passages do not contain the answer, say so clearly and say what is missing. "
    "Reply in the same language as the question. "
    "Cite only the passages that support your answer, like [1], [2]; if none do, cite nothing. "
    "Keep the answer short and end with one line reminding the reader this is not professional advice."
)

app = FastAPI(
    title="Daliil — Document Intelligence API",
    description=(
        "A multilingual document question-answering API "
        "with semantic search and source-grounded answers."
    ),
    version="0.1.0",
)

# ---------- Components (created lazily so the app starts fast) ----------

@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name="documents", metadata={"hnsw:space": "cosine"}
    )


def llm_enabled() -> bool:
    return bool(LLM_BASE_URL and LLM_MODEL)


# ---------- Document processing ----------

ARABIC_TO_WESTERN = {
    **{0x0660 + i: str(i) for i in range(10)},  # Arabic-Indic digits
    **{0x06F0 + i: str(i) for i in range(10)},  # Persian forms also appear in PDFs
}


def clean_arabic(text: str, reverse_digits: bool = False) -> str:
    """Repair common Arabic PDF extraction problems."""
    # Presentation forms (ﺘﺠﺎﻭﺯ) -> normal letters (تجاوز)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u0640", "")  # tatweel
    if reverse_digits:
        # Some PDFs store number runs backwards (150 comes out as 051).
        text = re.sub(r"[\u0660-\u0669\u06F0-\u06F9]+", lambda m: m.group()[::-1], text)
    text = text.translate(ARABIC_TO_WESTERN)
    text = re.sub(r"\)\s*(\d+)\s*\(", r"(\1)", text)  # ")150(" -> "(150)"
    text = re.sub(r"مادة\s*\(", "مادة (", text)
    # Re-attach letters split from their word ("قاض ى" -> "قاضى"). These
    # letters are never standalone Arabic words, so this is safe.
    text = re.sub(r"\s+([اىً]+)(?=\s|$|[.،,:؛)])", r"\1", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def extract_pages(filename: str, data: bytes, reverse_digits: bool = False) -> list[tuple[int, str]]:
    """Return [(page_number, text)] for a PDF or plain-text file."""
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return [
            (i + 1, clean_arabic(page.extract_text() or "", reverse_digits))
            for i, page in enumerate(reader.pages)
        ]
    if name.endswith((".txt", ".md")):
        return [(1, clean_arabic(data.decode("utf-8", errors="replace")))]
    raise HTTPException(status_code=415, detail="Only .pdf, .txt and .md files are supported.")


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping word windows."""
    words = text.split()
    step = max(CHUNK_WORDS - CHUNK_OVERLAP, 1)
    chunks = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + CHUNK_WORDS])
        if chunk:
            chunks.append(chunk)
        if start + CHUNK_WORDS >= len(words):
            break
    return chunks


def embed(texts: list[str], prefix: str) -> list[list[float]]:
    # e5 models expect "passage: " for documents and "query: " for questions.
    vectors = get_embedder().encode(
        [prefix + t for t in texts], normalize_embeddings=True, batch_size=32
    )
    return vectors.tolist()


# ---------- Exact article lookup ----------

ARTICLE_RE = re.compile(
    r"(?:المادة|مادة)\s*(?:رقم\s*)?\(?\s*([0-9\u0660-\u0669\u06F0-\u06F9]+)\s*\)?"
)


def _as_source(doc: str, meta: dict) -> dict:
    return {"source": meta["source"], "page": meta["page"], "score": 1.0, "text": doc}


def find_article_chunks(question: str, max_articles: int = 2) -> list[dict]:
    """Exact lookup for questions like 'ما نص المادة 150؟' (embeddings are weak at numbers).

    Returns the chunk holding the article heading plus the chunk right after it,
    because an article is often split across two chunks.
    """
    match = ARTICLE_RE.search(question)
    if not match:
        return []
    number = match.group(1).translate(ARABIC_TO_WESTERN)
    collection = get_collection()
    found = collection.get(
        where_document={"$contains": f"مادة ({number})"},
        limit=20,
        include=["documents", "metadatas"],
    )
    pairs = list(zip(found["documents"], found["metadatas"]))
    # Prefer the real heading ("مادة (5) :") over cross-references
    # ("... المادة (5) من هذا القانون").
    heading = re.compile(rf"مادة \({number}\)\s*:")
    pairs = [p for p in pairs if heading.search(p[0])] or pairs

    results = []
    for doc, meta in pairs[:max_articles]:
        results.append(_as_source(doc, meta))
        if "chunk" in meta:  # missing on chunks indexed by older versions
            nxt = collection.get(
                where={"$and": [{"source": meta["source"]}, {"chunk": meta["chunk"] + 1}]},
                include=["documents", "metadatas"],
            )
            results += [_as_source(d, m) for d, m in zip(nxt["documents"], nxt["metadatas"])]
    return results


# ---------- Generation ----------

def generate_answer(question: str, sources: list[dict]) -> str:
    from openai import OpenAI

    context = "\n\n".join(
        f"[{i}] ({s['source']}, page {s['page']})\n{s['text']}"
        for i, s in enumerate(sources, start=1)
    )
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content


# ---------- API ----------

class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=TOP_K, ge=1, le=10)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks_indexed": get_collection().count(),
        "mode": "rag" if llm_enabled() else "retrieval_only",
        "embedding_model": EMBED_MODEL,
    }


@app.get("/documents")
def list_documents():
    metadatas = get_collection().get(include=["metadatas"])["metadatas"]
    return {"documents": sorted({m["source"] for m in metadatas})}


@app.post("/ingest")
def ingest(file: UploadFile = File(...), reverse_digits: bool = False):
    pages = extract_pages(file.filename, file.file.read(), reverse_digits)

    texts, metas = [], []
    for page_no, page_text in pages:
        for chunk in chunk_text(page_text):
            texts.append(chunk)
            metas.append({"source": file.filename, "page": page_no, "chunk": len(texts) - 1})

    if not texts:
        raise HTTPException(
            status_code=422,
            detail="No extractable text found (scanned PDF? it would need OCR).",
        )

    collection = get_collection()
    collection.delete(where={"source": file.filename})  # re-upload replaces old version
    collection.add(
        ids=[str(uuid.uuid4()) for _ in texts],
        documents=texts,
        metadatas=metas,
        embeddings=embed(texts, "passage: "),
    )
    return {"source": file.filename, "pages": len(pages), "chunks_added": len(texts)}


@app.post("/ask")
def ask(req: AskRequest):
    collection = get_collection()
    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed yet. Use /ingest first.")

    result = collection.query(
        query_embeddings=embed([req.question], "query: "),
        n_results=min(req.top_k, collection.count()),
    )
    sources = [
        {
            "source": meta["source"],
            "page": meta["page"],
            "score": round(1 - dist, 4),  # cosine similarity
            "text": doc,
        }
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]

    # Put exact "المادة N" matches first, then the semantic results (deduplicated).
    exact = [] if os.getenv("DISABLE_ARTICLE_LOOKUP") == "1" else find_article_chunks(req.question)
    seen = {s["text"] for s in exact}
    sources = exact + [s for s in sources if s["text"] not in seen]

    if not llm_enabled():
        return {"mode": "retrieval_only", "answer": None, "sources": sources}

    try:
        answer = generate_answer(req.question, sources)
    except Exception as exc:  # network, auth, model name, rate limit...
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}")

    return {"mode": "rag", "answer": answer, "sources": sources}

@app.delete("/documents/{source}")
def delete_document(source: str):
    get_collection().delete(where={"source": source})
    return {"deleted": source}


@app.delete("/documents")
def reset_index():
    chromadb.PersistentClient(path=CHROMA_DIR).delete_collection("documents")
    get_collection.cache_clear()  # next call recreates an empty collection
    return {"status": "index cleared"}