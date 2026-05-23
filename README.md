# ⚖️ Arabic Legal RAG Analyzer

> A production-ready Arabic/English Legal Contract Analysis system powered by **Retrieval-Augmented Generation (RAG)**.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-LLM-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-ready-blue?style=flat-square&logo=docker)

---

## 🎯 What It Does

Upload any legal contract PDF (Arabic or English) and get:

- 📋 **Automatic Contract Report** — parties, dates, value, obligations
- ⚠️ **Risk Detection** — risky clauses classified as 🔴 High / 🟡 Medium / 🟢 Low
- 💬 **Legal Q&A Chat** — ask anything about the contract with source citations
- 📊 **Visual Analytics** — risk distribution charts powered by Plotly

---

## 🏗️ Architecture

```
PDF Upload
      ↓
┌─────────────────────┐
│   Legal Chunker     │  ← Splits on Articles & Clauses
└─────────────────────┘
      ↓
┌──────────────────────────────────────┐
│         Index Building               │
│   ChromaDB (Dense) + BM25 (Sparse)  │
└──────────────────────────────────────┘
      ↓                    ↓
┌──────────────┐   ┌──────────────────────┐
│   Q&A Chat   │   │   Legal Analysis      │
│              │   │  ● Classify Contract  │
│  Expand ──►  │   │  ● Extract Info JSON  │
│  Retrieve ──►│   │  ● Detect Risks       │
│  Rerank ──►  │   └──────────────────────┘
│  Generate    │              ↓
└──────────────┘   ┌──────────────────────┐
        ↓          │    Streamlit UI       │
        └─────────►│  Tab 1: 📊 Report    │
                   │  Tab 2: ⚠️ Risks     │
                   │  Tab 3: 💬 Chat      │
                   └──────────────────────┘
```

---

## 🚀 RAG Pipeline Components

| Component | Technology | Details |
|-----------|-----------|---------|
| **Loader** | PyMuPDF | Legal-aware chunking on Articles & Clauses |
| **Embedder** | `paraphrase-multilingual-mpnet-base-v2` | 768-dim, Arabic + English |
| **Vector Store** | ChromaDB | Cosine HNSW similarity search |
| **Sparse Search** | BM25Okapi | Arabic tokenizer `\u0600-\u06FF` |
| **Fusion** | RRF (k=60) | `score = Σ 1/(60 + rank)` |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-Encoder for precision |
| **Query Expansion** | Groq LLM | Generates 2 alternative queries |
| **Generator** | Groq `gpt-oss-120b` | Instruction-based prompting |

---

## 📁 Project Structure

```
arabic-legal-rag-analyzer/
├── legal_core.py      ← Full RAG + Legal Pipeline
├── legal_app.py       ← Streamlit UI (3 Tabs + Charts)
├── rag_core.py        ← Base RAG Pipeline
├── chatbot.py         ← Terminal Chatbot
├── chatbot_app.py     ← Streamlit Chatbot
├── Dockerfile         ← Docker Container
├── requirements.txt   ← Dependencies
├── .env               ← API Keys (not committed)
└── .gitignore
```

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/your-username/arabic-legal-rag-analyzer.git
cd arabic-legal-rag-analyzer
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your API Key
Create a `.env` file:
```
rag_system=gsk_your_groq_api_key_here
```
Get a free key at [console.groq.com](https://console.groq.com)

### 4. Run

**Legal Analyzer (Streamlit):**
```bash
streamlit run legal_app.py
```

**Simple Chatbot (Streamlit):**
```bash
streamlit run chatbot_app.py
```

**Terminal Chatbot:**
```bash
python chatbot.py
```

**Docker:**
```bash
docker build -t legal-analyzer .
docker run -p 8501:8501 legal-analyzer
```

---

## 🖥️ Usage

1. Open `http://localhost:8501`
2. Upload your contract PDF from the Sidebar
3. Click **تحليل العقد** (Analyze Contract)
4. Explore the 3 tabs:
   - **📊 Report** — structured contract information
   - **⚠️ Risks** — risky clauses with recommendations
   - **💬 Chat** — ask questions in Arabic or English

---

## 📊 Prompt Techniques Used

| Technique | Where | Purpose |
|-----------|-------|---------|
| **Role-based Prompting** | All LLM calls | System + Human message separation |
| **Instruction-based Prompting** | Generator | Rules to prevent hallucination |
| **Constrained Output** | Extraction | Forces JSON output format |
| **Query Expansion Prompting** | expand_queries() | Generates alternative phrasings |

---

## 🔬 Key Technical Decisions

**Why Hybrid Retrieval?**
Dense search (ChromaDB) understands meaning but misses exact terms.
BM25 catches exact keywords but misses semantics.
Combined via RRF → best of both worlds.

**Why Cross-Encoder Reranking?**
Bi-Encoders compute query and document separately.
Cross-Encoders analyze them together → significantly more accurate.

**Why Query Expansion?**
User asks "كم عدد الـ heads" — PDF says "8 attention heads".
Expansion generates Arabic/English variations → higher recall.

---

## 🛠️ Tech Stack

```
Python 3.12
LangChain + LangChain-Groq
Groq API (openai/gpt-oss-120b)
ChromaDB
sentence-transformers
rank-bm25
PyMuPDF (fitz)
Streamlit
Plotly
Docker
python-dotenv
```

---

## 📄 Supported Contract Types

- ✅ عقد عمل (Employment Contract)
- ✅ عقد إيجار (Rental Contract)
- ✅ عقد خدمات (Service Contract)
- ✅ NDA (Non-Disclosure Agreement)
- ✅ عقد شراكة (Partnership Contract)
- ✅ عقد توريد (Supply Contract)

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## 📝 License

MIT License — free to use and modify.

---

> Built with ❤️ using LangChain, Groq, and sentence-transformers
