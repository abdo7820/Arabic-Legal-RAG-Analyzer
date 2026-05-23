# ===========================================================================
# rag_core.py — الـ RAG Pipeline الكاملة
# يُستخدَم من: app.py و RAG_Enhanced.ipynb
# ===========================================================================

import os
from dotenv import load_dotenv


load_dotenv()
import re
import numpy as np
import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb

# ===========================================================================
# MODELS (Lazy Loading)
# ===========================================================================
_embedder: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None
_chroma_client: chromadb.ClientAPI | None = None

def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print("[INFO] Loading Embedder model...")
        _embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    return _embedder

def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        print("[INFO] Loading Reranker model...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

def get_chroma_client() -> chromadb.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.EphemeralClient()
    return _chroma_client

# ===========================================================================
# 1. LOADER & CHUNKER
# يقبل: file_path (str) للـ notebook أو file_bytes+filename للـ app
# ===========================================================================
def load_and_chunk_path(file_path: str):
    """للـ notebook — يقرأ من مسار ملف"""
    pdf = fitz.open(file_path)
    pages_text = [(i + 1, page.get_text()) for i, page in enumerate(pdf)]
    pdf.close()
    return _chunk_pages(pages_text, file_path)

def load_and_chunk_bytes(file_bytes: bytes, filename: str):
    """للـ app.py — يقرأ من bytes مرفوعة"""
    pdf = fitz.open(stream=file_bytes, filetype="pdf")
    pages_text = [(i + 1, page.get_text()) for i, page in enumerate(pdf)]
    pdf.close()
    return _chunk_pages(pages_text, filename)

def _chunk_pages(pages_text: list, source: str):
    """مشترك بين الاتنين"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", "،", ".", "؟", "!", " ", ""],
    )
    chunks = []
    chunk_id = 0
    for page_num, text in pages_text:
        if not text.strip():
            continue
        raw_doc = Document(
            page_content=text,
            metadata={"source": source, "page": page_num}
        )
        page_chunks = splitter.split_documents([raw_doc])
        for chunk in page_chunks:
            chunk.metadata["chunk_id"] = chunk_id
            chunk.metadata["page"] = page_num
            chunk_id += 1
        chunks.extend(page_chunks)
    print(f"[INFO] Split into {len(chunks)} chunks across {len(pages_text)} pages.")
    return chunks, len(pages_text)

# ===========================================================================
# 2. EMBEDDER
# ===========================================================================
def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_embedder()
    return np.array(model.encode(
        texts, batch_size=100, normalize_embeddings=True, show_progress_bar=False
    ))

def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]

# ===========================================================================
# 3. VECTOR STORE (ChromaDB)
# ===========================================================================
def build_vector_index(chunks: list[Document]):
    client = get_chroma_client()
    try:
        client.delete_collection("rag_demo")
    except Exception:
        pass
    collection = client.create_collection("rag_demo", metadata={"hnsw:space": "cosine"})
    texts = [c.page_content for c in chunks]
    print("[INFO] Computing embeddings...")
    embeddings = embed_texts(texts)
    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        ids=[str(c.metadata["chunk_id"]) for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )
    print(f"[INFO] Vector index built with {len(chunks)} chunks.")
    return collection

def vector_search(collection, query_emb: np.ndarray, top_k: int = 20) -> list[dict]:
    results = collection.query(
        query_embeddings=[query_emb.tolist()],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"text": doc, "meta": meta, "score": 1.0 - dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]

# ===========================================================================
# 4. BM25
# ===========================================================================
_TOKEN_PATTERN = re.compile(r"[\w\u0600-\u06FF]+")

def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())

def build_bm25(chunks: list[Document]):
    corpus = [tokenize(c.page_content) for c in chunks]
    index = BM25Okapi(corpus)
    print(f"[INFO] BM25 index built with {len(chunks)} chunks.")
    return index

def bm25_search(bm25_index, chunks: list[Document], query: str, top_k: int = 20) -> list[dict]:
    scores = bm25_index.get_scores(tokenize(query))
    ranked = sorted(
        [(score, idx) for idx, score in enumerate(scores) if score > 0],
        reverse=True,
    )
    return [
        {"text": chunks[idx].page_content, "meta": chunks[idx].metadata, "score": float(score)}
        for score, idx in ranked[:top_k]
    ]

# ===========================================================================
# 5. RRF FUSION
# ===========================================================================
def rrf(dense: list[dict], sparse: list[dict], k: int = 60) -> list[dict]:
    scores, meta_map, text_map = {}, {}, {}
    for result_list in (dense, sparse):
        for rank, hit in enumerate(result_list):
            key = hit["text"][:100]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in meta_map:
                meta_map[key] = hit["meta"]
                text_map[key] = hit["text"]
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"text": text_map[k], "meta": meta_map[k], "rrf_score": s} for k, s in fused]

# ===========================================================================
# 6. RERANKER
# ===========================================================================
def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    reranker = get_reranker()
    pairs = [(query, c["text"]) for c in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    result = []
    for score, chunk in ranked[:top_k]:
        chunk["rerank_score"] = float(score)
        result.append(chunk)
    print(f"[INFO] Reranked to top {len(result)} chunks.")
    return result

# ===========================================================================
# 7. QUERY EXPANSION
# ===========================================================================
def expand_queries(query: str) -> list[str]:
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a query expansion assistant. "
            "Generate exactly 2 alternative versions of the user's search query. "
            "Rules:\n"
            "1. Match the language of the original query exactly.\n"
            "2. Return ONLY the 2 alternatives, one per line.\n"
            "3. Do NOT number the lines.\n"
            "4. Do NOT add any explanation.\n"
            "5. Each alternative must rephrase the intent without adding new facts."
        ),
        HumanMessagePromptTemplate.from_template("{query}"),
    ])
    chain = prompt | llm | StrOutputParser()
    try:
        response = chain.invoke({"query": query})
        variations = [line.strip() for line in response.strip().splitlines() if line.strip()][:2]
        return [query] + variations
    except Exception as e:
        print(f"[WARNING] Query expansion failed: {e}")
        return [query]

# ===========================================================================
# 8. HYBRID RETRIEVE
# ===========================================================================
def hybrid_retrieve(query: str, collection, bm25_index, chunks: list[Document], top_k: int = 20) -> list[dict]:
    query_emb = embed_query(query)
    dense = vector_search(collection, query_emb, top_k=5)
    sparse = bm25_search(bm25_index, chunks, query, top_k=top_k)
    return rrf(dense, sparse)

# ===========================================================================
# 9. GENERATOR
# ===========================================================================
def generate_answer(query: str, context_chunks: list[dict], history: list[dict] = []) -> dict:
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

    chunks = context_chunks[:5]
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        page = chunk["meta"].get("page", "?")
        context_parts.append(f"[Chunk {i} | صفحة {page}]\n{chunk['text']}")
    context_text = "\n---\n".join(context_parts)

    history_text = ""
    if history:
        history_text = "سياق المحادثة السابقة:\n"
        for turn in history[-3:]:
            history_text += f"سؤال سابق: {turn['question']}\n"
            history_text += f"إجابة سابقة: {turn['answer'][:200]}...\n\n"

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "أنت مساعد ذكي خبير.\n\n"
            "قواعد:\n"
            "- اعتمد فقط على السياق المقدم\n"
            "- لا تخترع معلومات\n"
            "- لو الإجابة غير موجودة قل: \"غير مذكور في النص\"\n"
            "- اشرح ببساطة\n"
            "- نظّم الإجابة (نقاط أو خطوات إذا لزم)\n"
            "- راعي سياق المحادثة السابقة إذا كان متعلقاً بالسؤال"
        ),
        HumanMessagePromptTemplate.from_template(
            "{history}السياق:\n{context}\n\nالسؤال:\n{question}\n\nالإجابة:"
        ),
    ])
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"history": history_text, "context": context_text, "question": query}).strip()

    sources = [
        {
            "page": c["meta"].get("page", "?"),
            "chunk_id": c["meta"].get("chunk_id", "?"),
            "text": c["text"][:150] + "..."
        }
        for c in chunks
    ]
    return {"answer": answer, "sources": sources}

# ===========================================================================
# 10. FULL ASK PIPELINE (للـ notebook)
# ===========================================================================
def ask(query: str, collection, bm25_index, chunks: list[Document], chat_history: list[dict] = []) -> dict:
    print(f"\n{'='*60}")
    print(f"❓ السؤال: {query}")
    print('='*60)


    print("[INFO] Expanding query...")
    expanded = expand_queries(query)
    print("Expanded queries:")
    for i, q in enumerate(expanded, 1):
        print(f"  {i}. {q}")

    print("[INFO] Running Hybrid Retrieval (Dense + BM25)...")
    all_hits, seen = [], set()
    for q in expanded:
        hits = hybrid_retrieve(q, collection, bm25_index, chunks, top_k=20)
        for hit in hits:
            key = hit["text"][:100]
            if key not in seen:
                seen.add(key)
                all_hits.append(hit)
    print(f"[INFO] Retrieved {len(all_hits)} unique chunks.")

    print("[INFO] Reranking...")
    reranked = rerank(query, all_hits, top_k=5)

    print("[INFO] Generating answer...")
    result = generate_answer(query, reranked, chat_history)

    print(f"\n{'='*60}")
    print("💡 Answer:")
    print('='*60)
    print(result["answer"])
    print("\n📚 Sources:")
    print('-'*60)
    for i, src in enumerate(result["sources"], 1):
        print(f"  [{i}] Chunk ID: {src['chunk_id']} | 📄 صفحة: {src['page']}")
        print(f"      {src['text']}\n")

    chat_history.append({"question": query, "answer": result["answer"]})
    if len(chat_history) > 5:
        chat_history.pop(0)

    return result
