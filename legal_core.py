import os
from dotenv import load_dotenv


load_dotenv()
import json
import re
import fitz
import numpy as np

# ===========================================================================
# API KEY SETUP
# GROQ_API_KEY
# ===========================================================================
_key = os.environ.get("rag_system") or os.environ.get("GROQ_API_KEY", "")
if _key:
    os.environ["GROQ_API_KEY"] = _key
    print("[INFO] GROQ_API_KEY loaded ✅")
else:
    print("[WARNING] GROQ_API_KEY not found — set rag_system env variable")
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
# EMBEDDER
# ===========================================================================
_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
_model: SentenceTransformer | None = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print("[INFO] Loading SentenceTransformer model...")
        _model = SentenceTransformer(_MODEL_NAME)
    return _model

model = _get_model()

def embed_texts(texts: list[str]) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=100,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.array(embeddings)

def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]


# ===========================================================================
# 1. LOADER & LEGAL CHUNKER
# يقطع على المواد والبنود القانونية
# ===========================================================================
def load_and_chunk(file_path: str) -> list[Document]:
    print("[INFO] Extracting text from PDF...")
    pdf = fitz.open(file_path)
    pages_text = []
    for page_num, page in enumerate(pdf):
        text = page.get_text()
        pages_text.append((page_num + 1, text))
    pdf.close()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=[
            "\nالمادة ", "\nالبند ", "\nأولاً", "\nثانياً", "\nثالثاً",
            "\nArticle ", "\nClause ", "\nSection ",
            "\n\n", "\n", "،", ".", "؟", "!", " ", ""
        ],
    )

    print("[INFO] Splitting contract into chunks...")
    chunks = []
    chunk_id = 0
    for page_num, text in pages_text:
        if not text.strip():
            continue
        raw_doc = Document(
            page_content=text,
            metadata={"source": file_path, "page": page_num}
        )
        page_chunks = splitter.split_documents([raw_doc])
        for chunk in page_chunks:
            chunk.metadata["chunk_id"] = chunk_id
            chunk.metadata["page"] = page_num
            chunk_id += 1
        chunks.extend(page_chunks)

    print(f"[INFO] Split into {len(chunks)} chunks across {len(pages_text)} pages.")
    return chunks


# ===========================================================================
# 2. VECTOR STORE (ChromaDB)
# ===========================================================================
_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None
_COLLECTION_NAME = "legal_rag"

def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.EphemeralClient()
    return _client

def build_index(chunks: list[Document]) -> None:
    global _collection
    client = _get_client()

    try:
        client.delete_collection(_COLLECTION_NAME)
    except Exception:
        pass

    _collection = client.create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [chunk.page_content for chunk in chunks]
    print("[INFO] Computing embeddings for vector store...")
    embeddings: np.ndarray = embed_texts(texts)

    _collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        ids=[str(chunk.metadata["chunk_id"]) for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
    )
    print(f"[INFO] Vector index built with {len(chunks)} chunks.")

def vector_search(query_embedding: np.ndarray, top_k: int = 20) -> list[dict]:
    if _collection is None:
        raise RuntimeError("Call build_index() before vector_search().")

    results = _collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, _collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits: list[dict] = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text": doc,
            "meta": meta,
            "score": 1.0 - dist,
        })
    return hits


# ===========================================================================
# 3. BM25 STORE
# ===========================================================================
_bm25_index: BM25Okapi | None = None
_chunks: list[Document] = []
_TOKEN_PATTERN = re.compile(r"[\w\u0600-\u06FF]+")

def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())

def build_bm25_index(chunks: list[Document]) -> None:
    global _bm25_index, _chunks
    _chunks = chunks
    tokenized_corpus = [tokenize(chunk.page_content) for chunk in chunks]
    _bm25_index = BM25Okapi(tokenized_corpus)
    print(f"[INFO] BM25 index built with {len(chunks)} chunks.")

def bm25_search(query: str, top_k: int = 20) -> list[dict]:
    if _bm25_index is None:
        raise RuntimeError("Call build_bm25_index() before bm25_search().")

    query_tokens = tokenize(query)
    scores: np.ndarray = _bm25_index.get_scores(query_tokens)

    ranked = sorted(
        [(score, idx) for idx, score in enumerate(scores) if score > 0],
        reverse=True,
    )

    hits: list[dict] = []
    for score, idx in ranked[:top_k]:
        chunk = _chunks[idx]
        hits.append({
            "text": chunk.page_content,
            "meta": chunk.metadata,
            "score": float(score),
        })
    return hits


# ===========================================================================
# 4. QUERY EXPANSION
# ===========================================================================
_transform_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
)

_transform_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a legal query expansion assistant. "
        "Your task is to generate exactly 2 alternative versions of the user's search query. "
        "Rules:\n"
        "1. Match the language of the original query exactly "
        "(if the query is in Arabic, respond in Arabic; if English, respond in English).\n"
        "2. Return ONLY the 2 alternatives, one per line.\n"
        "3. Do NOT number the lines.\n"
        "4. Do NOT add any explanation, prefix, or commentary.\n"
        "5. Each alternative must rephrase the intent without adding new facts."
    ),
    HumanMessagePromptTemplate.from_template("{query}"),
])

_transform_chain = _transform_prompt | _transform_llm | StrOutputParser()

def expand_queries(query: str) -> list[str]:
    try:
        response = _transform_chain.invoke({"query": query})
        raw: str = response.strip()
        variations = [line.strip() for line in raw.splitlines() if line.strip()]
        variations = variations[:3]
        return [query] + variations
    except Exception as e:
        print(f"[WARNING] Query expansion failed: {e}")
        return [query]


# ===========================================================================
# 5. HYBRID RETRIEVAL + RRF
# ===========================================================================
RRF_K = 60

def reciprocal_rank_fusion(dense_results: list[dict], sparse_results: list[dict]) -> list[dict]:
    scores: dict[str, float] = {}
    meta_map: dict[str, dict] = {}
    text_map: dict[str, str] = {}

    for result_list in (dense_results, sparse_results):
        for rank, hit in enumerate(result_list):
            key = hit["text"][:100]
            rrf_contribution = 1.0 / (RRF_K + rank + 1)
            scores[key] = scores.get(key, 0.0) + rrf_contribution
            if key not in meta_map:
                meta_map[key] = hit["meta"]
                text_map[key] = hit["text"]

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {"text": text_map[key], "meta": meta_map[key], "rrf_score": score}
        for key, score in fused
    ]

def hybrid_retrieve(query: str, top_k: int = 20) -> list[dict]:
    query_emb = embed_query(query)
    dense  = vector_search(query_emb, top_k=5)
    sparse = bm25_search(query, top_k=top_k)
    return reciprocal_rank_fusion(dense, sparse)


# ===========================================================================
# 6. RERANKER
# ===========================================================================
_reranker: CrossEncoder | None = None

def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        print("[INFO] Loading Reranker model...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    reranker = _get_reranker()
    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    reranked = []
    for score, chunk in ranked[:top_k]:
        chunk["rerank_score"] = float(score)
        reranked.append(chunk)
    print(f"[INFO] Reranked to top {len(reranked)} chunks.")
    return reranked


# ===========================================================================
# 7. CONTRACT CLASSIFIER
# يعرف نوع العقد تلقائياً
# ===========================================================================
_classify_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

_classify_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "أنت محامٍ خبير. حدد نوع العقد من النص المعطى.\n"
        "أجب بكلمة أو كلمتين فقط من القائمة:\n"
        "عقد عمل، عقد إيجار، عقد بيع، عقد خدمات، عقد شراكة، NDA، عقد توريد، غير محدد"
    ),
    HumanMessagePromptTemplate.from_template("النص:\n{text}"),
])

_classify_chain = _classify_prompt | _classify_llm | StrOutputParser()

def classify_contract(full_text: str) -> str:
    print("[INFO] Classifying contract type...")
    try:
        result = _classify_chain.invoke({"text": full_text[:2000]})
        contract_type = result.strip()
        print(f"[INFO] Contract type: {contract_type}")
        return contract_type
    except Exception as e:
        print(f"[WARNING] Classification failed: {e}")
        return "غير محدد"


# ===========================================================================
# 8. STRUCTURED EXTRACTION
# يستخرج المعلومات الأساسية كـ JSON
# ===========================================================================
_extract_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

_extract_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "أنت محامٍ خبير في تحليل العقود القانونية.\n"
        "استخرج المعلومات التالية من العقد وأجب بـ JSON فقط بدون أي نص إضافي:\n\n"
        "{{\n"
        "  \"parties\": [\"الطرف الأول\", \"الطرف الثاني\"],\n"
        "  \"start_date\": \"تاريخ البداية أو null\",\n"
        "  \"end_date\": \"تاريخ الانتهاء أو null\",\n"
        "  \"contract_value\": \"قيمة العقد أو null\",\n"
        "  \"subject\": \"موضوع العقد في جملة واحدة\",\n"
        "  \"termination_conditions\": [\"شرط الإنهاء 1\", \"شرط الإنهاء 2\"],\n"
        "  \"payment_terms\": \"شروط الدفع أو null\",\n"
        "  \"governing_law\": \"القانون المُطبَّق أو null\",\n"
        "  \"key_obligations\": [\"التزام 1\", \"التزام 2\"]\n"
        "}}\n\n"
        "مهم: أجب بـ JSON فقط — لا تضف أي شرح أو مقدمة."
    ),
    HumanMessagePromptTemplate.from_template("العقد:\n{text}"),
])

_extract_chain = _extract_prompt | _extract_llm | StrOutputParser()

def extract_contract_info(full_text: str) -> dict:
    print("[INFO] Extracting contract information...")
    try:
        response = _extract_chain.invoke({"text": full_text[:6000]})
        clean = response.strip()
        clean = re.sub(r"```json|```", "", clean).strip()
        result = json.loads(clean)
        print("[INFO] Contract info extracted successfully.")
        return result
    except Exception as e:
        print(f"[WARNING] Extraction failed: {e}")
        return {
            "parties": [],
            "start_date": None,
            "end_date": None,
            "contract_value": None,
            "subject": "غير محدد",
            "termination_conditions": [],
            "payment_terms": None,
            "governing_law": None,
            "key_obligations": []
        }


# ===========================================================================
# 9. RISK DETECTION
# يكتشف البنود الخطيرة ويصنفها
# ===========================================================================

RISK_KEYWORDS = [
    # عربي
    "غرامة", "تعويض", "إنهاء فوري", "مسؤولية", "التزام مالي",
    "حجز", "توقيف", "فسخ", "خسارة", "ضرر", "عقوبة",
    # إنجليزي
    "penalty", "termination", "liable", "liability", "forfeit",
    "indemnify", "breach", "damages", "default", "fine", "liquidated"
]

_risk_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

_risk_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "أنت محامٍ خبير في تحليل مخاطر العقود.\n"
        "حلّل هذا البند وأجب بـ JSON فقط:\n\n"
        "{{\n"
        "  \"is_risky\": true أو false,\n"
        "  \"risk_type\": \"نوع الخطر (مالي / قانوني / تشغيلي / غير محدد)\",\n"
        "  \"severity\": \"high أو medium أو low\",\n"
        "  \"explanation\": \"شرح المخاطرة في جملة واحدة\",\n"
        "  \"recommendation\": \"توصية للتعامل مع هذا البند\"\n"
        "}}\n\n"
        "مهم: أجب بـ JSON فقط."
    ),
    HumanMessagePromptTemplate.from_template("البند:\n{clause}"),
])

_risk_chain = _risk_prompt | _risk_llm | StrOutputParser()

def detect_risks(chunks: list[Document]) -> list[dict]:
    print("[INFO] Detecting risks in contract clauses...")
    risks = []

    for chunk in chunks:
        text = chunk.page_content
        has_keyword = any(kw.lower() in text.lower() for kw in RISK_KEYWORDS)
        if not has_keyword:
            continue

        try:
            response = _risk_chain.invoke({"clause": text[:1000]})
            clean = response.strip()
            clean = re.sub(r"```json|```", "", clean).strip()
            risk_data = json.loads(clean)

            if risk_data.get("is_risky"):
                risks.append({
                    "chunk_id": chunk.metadata.get("chunk_id", "?"),
                    "page": chunk.metadata.get("page", "?"),
                    "text": text,
                    "risk_type": risk_data.get("risk_type", "غير محدد"),
                    "severity": risk_data.get("severity", "low"),
                    "explanation": risk_data.get("explanation", ""),
                    "recommendation": risk_data.get("recommendation", ""),
                })
        except Exception as e:
            print(f"[WARNING] Risk detection failed for chunk: {e}")
            continue

    severity_order = {"high": 0, "medium": 1, "low": 2}
    risks.sort(key=lambda x: severity_order.get(x["severity"], 3))

    print(f"[INFO] Found {len(risks)} risky clauses.")
    return risks


# ===========================================================================
# 10. LEGAL Q&A GENERATOR
# نفس الـ generator الأصلي بس بـ prompt قانوني
# ===========================================================================
_gen_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

_gen_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "أنت محامٍ خبير ومستشار قانوني.\n\n"
        "قواعد:\n"
        "- اعتمد فقط على نصوص العقد المُقدَّمة\n"
        "- لا تخترع معلومات قانونية\n"
        "- لو الإجابة غير موجودة قل: 'غير مذكور في العقد'\n"
        "- اذكر المادة أو البند المرتبط بالإجابة إن وُجد\n"
        "- نظّم الإجابة بنقاط واضحة\n"
        "- راعي سياق المحادثة السابقة إذا كان متعلقاً بالسؤال"
    ),
    HumanMessagePromptTemplate.from_template(
        "{history}"
        "نصوص العقد:\n{context}\n\n"
        "السؤال القانوني:\n{question}\n\n"
        "الإجابة:"
    ),
])

_gen_chain = _gen_prompt | _gen_llm | StrOutputParser()

def generate_answer(query: str, context_chunks: list[dict], history: list[dict] = []) -> dict:
    chunks = context_chunks[:5]
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        page = chunk["meta"].get("page", "?")
        context_parts.append(f"[Chunk {i} | صفحة {page}]\n{chunk['text']}")
    context_text = "\n---\n".join(context_parts)

    history_text = ""
    if history:
        history_text = "سياق المحادثة السابقة:\n"
        for turn in history[-3:]:
            history_text += f"سؤال سابق: {turn['question']}\n"
            history_text += f"إجابة سابقة: {turn['answer'][:200]}...\n\n"

    response = _gen_chain.invoke({
        "history": history_text,
        "context": context_text,
        "question": query
    })
    answer: str = response.strip()

    sources: list[dict] = [
        {
            **chunk["meta"],
            "text": chunk["text"][:200] + "...",
            "page": chunk["meta"].get("page", "?"),
        }
        for chunk in chunks
    ]
    return {"answer": answer, "sources": sources}


# ===========================================================================
# 11. FULL PIPELINE — INDEX + ANALYZE
# ===========================================================================
def analyze_contract(file_path: str) -> dict:
    print("=" * 60)
    print("     Legal Contract Analyzer — Phase 1: Analysis")
    print("=" * 60)

    chunks = load_and_chunk(file_path)
    full_text = " ".join([c.page_content for c in chunks])

    build_index(chunks)
    build_bm25_index(chunks)
    _get_reranker()

    contract_type = classify_contract(full_text)

    contract_info = extract_contract_info(full_text)
    contract_info["contract_type"] = contract_type

    risks = detect_risks(chunks)

    print("\n✅ Analysis Complete!")
    return {
        "chunks": chunks,
        "contract_info": contract_info,
        "risks": risks,
    }


# ===========================================================================
# 12. Q&A — ASK
# ===========================================================================
def ask(query: str, chat_history: list[dict] = []) -> dict:
    print(f"\n{'='*60}")
    print(f"❓ السؤال: {query}")
    print("=" * 60)

    print("[INFO] Expanding query...")
    expanded = expand_queries(query)
    print("Expanded queries:")
    for i, q in enumerate(expanded, 1):
        print(f"  {i}. {q}")

    print("[INFO] Running Hybrid Retrieval (Dense + BM25)...")
    all_hits: list[dict] = []
    seen_texts = set()
    for q in expanded:
        hits = hybrid_retrieve(q, top_k=20)
        for hit in hits:
            key = hit["text"][:100]
            if key not in seen_texts:
                seen_texts.add(key)
                all_hits.append(hit)
    print(f"[INFO] Retrieved {len(all_hits)} unique chunks.")

    print("[INFO] Reranking...")
    reranked = rerank(query, all_hits, top_k=5)

    print("[INFO] Generating legal answer...")
    result = generate_answer(query, reranked, chat_history)

    print(f"\n{'='*60}")
    print("⚖️ Answer:")
    print("=" * 60)
    print(result["answer"])
    print("\n📚 Sources:")
    print("-" * 60)
    for i, source in enumerate(result["sources"], 1):
        print(f"  [{i}] Chunk ID: {source.get('chunk_id', '?')} | 📄 صفحة: {source.get('page', '?')}")
        print(f"      {source['text']}\n")

    chat_history.append({"question": query, "answer": result["answer"]})
    if len(chat_history) > 5:
        chat_history.pop(0)

    return result
