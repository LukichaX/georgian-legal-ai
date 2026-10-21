# ⚖️ Georgian Legal AI — Neuro-Symbolic Agentic RAG System

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.45-orange.svg)](https://github.com/langchain-ai/langgraph)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5.15-red.svg)](https://www.trychroma.com/)
[![SQLite FTS5](https://img.shields.io/badge/SQLite-FTS5_Trigram-lightgrey.svg)](https://sqlite.org/fts5.html)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA_12.8-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade legal research and case-analysis engine built over the **Constitution of Georgia and 20 active statutory codes (5,624 consolidated articles)** scraped directly from the official legislative gazette [`matsne.gov.ge`](https://matsne.gov.ge).

Designed specifically around **Georgian/Germanic Continental Legal Dogmatics**, the system unifies deterministic SQL lookups, multi-query legal issue decomposition, multi-tier dense/sparse retrieval with single-pass vector reuse, cross-encoder reranking, and deterministic citation verification.

---

## 📑 Table of Contents

- [System Architecture](#-system-architecture)
- [Key Engineering Highlights](#-key-engineering-highlights)
- [Statutory Corpus (21 Codified Acts)](#-statutory-corpus-21-codified-acts)
- [Hardware & Model Profiles](#-hardware--model-profiles)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
  - [1. Environment Setup](#1-environment-setup)
  - [2. ETL & Indexing Pipeline](#2-etl--indexing-pipeline)
  - [3. Running the Server](#3-running-the-server)
  - [4. Adversarial Stress-Test Suite](#4-adversarial-stress-test-suite)
- [REST API Contract](#-rest-api-contract)
- [Dogmatic Methodology & Anti-Hallucination](#-dogmatic-methodology--anti-hallucination)
- [Telemetry & Crowdsourced Feedback](#-telemetry--crowdsourced-feedback)

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    User([User Query / Legal Case]) --> Server[FastAPI Server :8000]
    Server --> HotSwap{Zero-Downtime Hot-Swap Engine}
    HotSwap --> Agent[Stateful LangGraph Agent]

    subgraph LangGraph ["Stateful LangGraph Pipeline (legal_agent.py)"]
        RouterNode[1. Router & Multi-Query Decomposer\nCloud Gemini / Local GGUF]
        RetrieverNode[2. Multi-Tier Hybrid Retriever]
        ReuseContextNode[2b. Reuse Active Context\nFollow-up / Calculation]
        GeneratorNode[3. Adaptive 3-Mode Generator\nContinental Legal Dogmatics]
        GuardNode[4. Deterministic Citation Guard\nStrict [E1]-[E6] & Injection Shield]
        MemoryNode[5. State & Memory Checkpointer\nLangGraph MemorySaver]

        RouterNode -->|hybrid_search / exact_lookup| RetrieverNode
        RouterNode -->|follow_up_existing| ReuseContextNode
        RouterNode -->|direct_chat / out_of_scope| GeneratorNode

        RetrieverNode --> GeneratorNode
        ReuseContextNode --> GeneratorNode
        GeneratorNode --> GuardNode
        GuardNode --> MemoryNode
    end

    subgraph RetrievalEngine ["Modular Hybrid Retrieval Engine (hybrid_retriever.py)"]
        Tier0[Tier 0: Deterministic SQL Article Lookup\n<1ms B-Tree Index]
        Tier1A[Tier 1A: Sparse FTS5 Trigram BM25\nCustom Georgian Stemmer]
        Tier1B[Tier 1B: Dense ChromaDB Cosine Search\nSingle-Pass Query Vectorization]
        Tier1C[Tier 1C: Statutory Title & Breadcrumb Matcher\nHeader & Definitional Bonus]
        RRF[Reciprocal Rank Fusion - RRF\nGlobal + Primary + Adjacent Codes]
        Tier2[Tier 2: Cross-Encoder Reranker\nQwen/Qwen3-Reranker-0.6B FP16]

        Tier0 -.-> RRF
        Tier1A --> RRF
        Tier1B --> RRF
        Tier1C --> RRF
        RRF --> Tier2
    end

    RetrieverNode <--> RetrievalEngine
    GuardNode --> ClientResp([Verified Response + Evidence Pack])
    MemoryNode --> Telemetry[(Telemetry Store\nSQLite + JSONL Logs)]
```

---

## ⚡ Key Engineering Highlights

### 1. Lossless DOM-First Legal AST (`src/data_pipeline.py`)
- **Superscript & Prime Preservation:** Handles statutory numbering variations (`144^1`, `126¹`, `(1)`) flawlessly without text corruption.
- **Hierarchical Breadcrumbs:** Preserves full legislative ancestry (`Book > Part > Chapter > Subchapter > Article`).
- **Repealed Article Pruning:** Automatically detects and filters out 814 repealed/inactive articles (`ძალადაკარგულია`), preserving 5,624 active statutory articles.
- **Dual-Storage Indexing:**
  - **SQLite B-Tree & FTS5 Trigram (`nodes` + `nodes_fts`):** Sub-millisecond direct lookup by article identifier + fast morphological full-text search with title weighting (`bm25(nodes_fts, 0.0, 10.0, 1.0, 6.0)`).
  - **ChromaDB (`georgian_legal_dense`):** Multi-chunk mean-pooled dense embeddings preventing truncation on long statutory provisions.

### 2. Modular Multi-Tier Hybrid Retriever (`src/hybrid_retriever.py`)
- **Single-Pass Query Vectorization:** Generates embedding for each decomposed sub-query once and shares it across global, primary-code, and secondary-code vector partitions.
- **Georgian Morphological Stemmer:** Preserves 3-letter semantic roots (e.g., `ვად`, `წილ`, `სესხ`, `ნივთ`, `ზიან`, `ბრალ`, `ქირ`) precisely aligning with SQLite FTS5 3-character trigram tokenization.
- **Statutory Breadcrumb Matcher (Tier 1C):** Instantaneous SQL header match scoring with definitional article bonuses (`ცნება`, `განაკვეთი`, `ზოგადი დებულებები`).
- **Reciprocal Rank Fusion (RRF):** Fuses dense, sparse, and header ranks across primary and adjacent codes.
- **Tier 2 Cross-Encoder Reranking:** Applies `Qwen/Qwen3-Reranker-0.6B` (FP16 CUDA) with round-robin slot allocation across decomposed legal sub-queries.

### 3. Stateful LangGraph Legal Agent (`src/legal_agent.py`)
- **Multi-Query Issue Spotting:** Decomposes complex disputes into 1–3 orthogonal statutory queries:
  1. *Specific Part Offense / Institution* (e.g., compulsory inheritance share, fraud vs. embezzlement).
  2. *General Part Doctrine* (e.g., limitation period, attempt, complicity, good-faith acquisition).
  3. *Exceptions / Procedures / Sanctions* (e.g., exclusion of evidence, emergency searches, formal notice).
- **Homonym & Trap Disambiguation:** Distinguishes bodily homonyms (*"მუხლი დამიზიანდა"* - knee injured) and contract clauses (*"ხელშეკრულების მე-5 მუხლი"*) from statutory codes.
- **Adaptive 3-Mode Generator:**
  - **Mode A (Case Resolution):** Three-tier Continental methodology (*Tatbestand* $\rightarrow$ *Rechtswidrigkeit* $\rightarrow$ *Schuld*).
  - **Mode B (Direct Query / Calculation):** Concise 2–4 sentence answers with exact statutory deadlines and mathematical computation.
  - **Mode C (Legal Drafting):** Formal legal templates (notices, claims, applications).
- **Deterministic Citation Guard:** Verifies `[E1]`–`[E6]` statutory markers against retrieved context. Hallucinated markers and prompt injection tags (e.g., `[E77]`) are stripped deterministically.

### 4. Zero-Downtime GPU Hot-Swap FastAPI Server (`src/server.py`)
- **VRAM Pinning:** Keeps heavy PyTorch models (Embedder, Cross-Encoder, Local GGUF Router, ChromaDB) pinned in GPU memory during startup.
- **Zero-Downtime Hot-Reload:** Listens to filesystem modifications in `src/`. When `hybrid_retriever.py`, `legal_agent.py`, or `telemetry_store.py` are modified, modules reload via `importlib.reload()` in **<50ms** without restarting the process or dropping Cloudflare tunnels.

---

## 📚 Statutory Corpus (21 Codified Acts)

The database consolidates **5,624 clean active articles** across 21 foundational legal acts of Georgia:

| Code Slug | Official Statutory Title (ქართულად) | Matsne Doc ID | Active Articles |
|:---|:---|:---:|:---:|
| `constitution` | საქართველოს კონსტიტუცია | `30346` | 78 |
| `criminal` | საქართველოს სისხლის სამართლის კოდექსი | `16426` | 516 |
| `criminal_proc` | საქართველოს სისხლის სამართლის საპროცესო კოდექსი | `90034` | 358 |
| `civil` | საქართველოს სამოქალაქო კოდექსი | `31702` | 1,409 |
| `civil_proc` | საქართველოს სამოქალაქო საპროცესო კოდექსი | `29962` | 583 |
| `labor` | საქართველოს შრომის კოდექსი | `1155567` | 88 |
| `gen_admin` | საქართველოს ზოგადი ადმინისტრაციული კოდექსი | `16270` | 172 |
| `admin_proc` | საქართველოს ადმინისტრაციული საპროცესო კოდექსი | `16492` | 154 |
| `admin_offenses` | საქართველოს ადმინისტრაციულ სამართალდარღვევათა კოდექსი | `28216` | 620 |
| `tax` | საქართველოს საგადასახადო კოდექსი | `1043717` | 303 |
| `customs` | საქართველოს საბაჟო კოდექსი | `4598501` | 214 |
| `election` | საქართველოს საარჩევნო კოდექსი | `1557168` | 218 |
| `imprisonment` | პატიმრობის კოდექსი | `91634` | 155 |
| `juvenile` | არასრულწლოვანთა მართლმსაჯულების კოდექსი | `2877281` | 105 |
| `child_rights` | ბავშვის უფლებათა კოდექსი | `4613854` | 101 |
| `spatial_building` | სივრცითი დაგეგმარების, არქიტექტურული და სამშენებლო კოდექსი | `4276845` | 158 |
| `product_safety` | პროდუქტის უსაფრთხოებისა და თავისუფალი მიმოქცევის კოდექსი | `1659419` | 122 |
| `forest` | საქართველოს ტყის კოდექსი | `4874066` | 98 |
| `env_assessment` | გარემოსდაცვითი შეფასების კოდექსი | `3691981` | 52 |
| `air` | საქართველოს საჰაერო კოდექსი | `33298` | 106 |
| `maritime` | საქართველოს საზღვაო კოდექსი | `32476` | 14 |
| **Total** | **21 Statutory Acts** | — | **5,624** |

---

## 🎛️ Hardware & Model Profiles

The system is fully hardware-agnostic and configured via `.env`:

| Profile | Hardware Requirement | `EMBED_PROVIDER` | `RERANKER_PROVIDER` | `ROUTER_PROVIDER` |
|:---|:---|:---|:---|:---|
| **Full GPU (Recommended)** | 16GB+ VRAM (RTX 3090/4090) | `gguf` (`Qwen3-Embedding-8B-Q8_0`) | `cross_encoder` (`Qwen3-Reranker-0.6B`) | `hybrid` (`Qwen3.5-4B` fallback) |
| **Mid GPU** | 6–8GB VRAM (RTX 3060/4060) | `huggingface` (`Qwen3-Embedding-0.6B`) | `cross_encoder` (`Qwen3-Reranker-0.6B`) | `cloud_only` (Gemini API) |
| **Zero VRAM / Cloud** | 0GB VRAM (CPU or VPS) | `google` (`text-embedding-004`) | `none` (Fast RRF + Header Match) | `cloud_only` (Gemini API) |

---

## 📂 Project Structure

```text
georgian-legal-ai/
├── src/
│   ├── server.py               # FastAPI Server + REST API + Zero-Downtime GPU Hot-Swap
│   ├── legal_agent.py          # Stateful LangGraph Agent, Issue Decomposer & Citation Guard
│   ├── hybrid_retriever.py     # Multi-Tier Modular Retriever (SQL + FTS5 + Chroma + Reranker)
│   ├── telemetry_store.py      # SQLite Multi-User Session Isolation, 1-10 Ratings & JSONL
│   └── data_pipeline.py        # Unified ETL CLI (Scraper + AST Parser + Dual Indexer)
├── utils/
│   ├── stress_test_runner.py   # Automated CSV-Driven Adversarial & Multi-Turn Benchmark Runner
│   ├── custom_stress_cases.csv # 30 Complex Adversarial Test Scenarios (Chains, Traps & Dogmatics)
│   └── stress_test_cases.csv   # 55 Auto-Bootstrapped Benchmark Test Cases
├── data/
│   ├── raw_html/               # Raw HTML gazettes from matsne.gov.ge
│   ├── processed_ast/          # Clean AST JSON files + ast_audit_report.json
│   ├── legal_database.db       # Consolidated SQLite database (nodes + FTS5 Trigram)
│   └── chroma_db/              # ChromaDB vector store (georgian_legal_dense)
├── logs/
│   ├── crowdsourced_eval.db    # Live session and turn telemetry database
│   ├── stress_test_report.md   # Benchmark evaluation report with full LLM answers
│   └── stress_test_summary.json# Statistical metrics & pass/clean percentages
├── frontend/                   # React + Vite + TypeScript Web UI (Spatial Glassmorphism)
├── .env.example                # Modular hardware & model configuration template
├── requirements.txt            # Python dependencies with CUDA wheel indices
└── start_live_server.ps1       # 1-Click Production Launcher (Vite Build + Tunnel + Server)
```

---

## 🚀 Quick Start

### 1. Environment Setup

```powershell
# Clone the repository
git clone https://github.com/LukichaX/georgian-legal-ai.git
cd georgian-legal-ai

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install PyTorch with CUDA 12.8 support
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install project dependencies
pip install -r requirements.txt

# Create .env configuration
Copy-Item .env.example .env
# Edit .env and insert your GOOGLE_API_KEY
```

### 2. ETL & Indexing Pipeline

Run the end-to-end pipeline to download, parse, and index all 21 codes:

```powershell
# Run Scrape -> Parse -> Dual Index in one command
python src/data_pipeline.py --step all

# Or run individual stages:
python src/data_pipeline.py --step scrape --codes criminal,civil,labor
python src/data_pipeline.py --step parse
python src/data_pipeline.py --step index --rebuild --batch-size 16
```

### 3. Running the Server

#### Option A: Direct FastAPI Server
```powershell
python src/server.py
```
The server will start on `http://localhost:8000`.

#### Option B: 1-Click Production Launcher with Cloudflare Tunnel
```powershell
.\start_live_server.ps1
```
This launcher:
1. Verifies/downloads `cloudflared.exe`.
2. Automatically builds the Vite frontend (if present).
3. Launches a secure Cloudflare Tunnel pointing to port 8000.
4. Starts the FastAPI server with auto-restart protection.

### 4. Adversarial Stress-Test Suite

Execute automated adversarial evaluations across multi-turn conversation chains, homonym traps, and edge cases:

```powershell
# Run full evaluation (Router + Retriever + Gemini Generator + Citation Guard)
python utils/stress_test_runner.py --cases utils/custom_stress_cases.csv

# Run retrieval-only benchmark (0 Gemini API calls, fast local validation)
python utils/stress_test_runner.py --cases utils/custom_stress_cases.csv --only-retrieval
```

Results and reports are generated in `logs/stress_test_report.md` and `logs/stress_test_summary.json`.

---

## 🔌 REST API Contract

The FastAPI server exposes clean endpoints designed for high-concurrency client interaction:

### `POST /api/chat`
Dispatches a turn through the LangGraph pipeline with automatic session tracking.

**Request:**
```json
{
  "thread_id": "session_user_01_abc",
  "tester_id": "usr_948",
  "tester_name": "დავითი",
  "message": "რა სასჯელია გათვალისწინებული ქურდობაზე ბინაში უკანონო შეღწევით?"
}
```

**Response:**
```json
{
  "turn_id": "turn_3a8f10b2c1",
  "answer": "საქართველოს სისხლის სამართლის კოდექსის 177-ე მუხლის მე-3 ნაწილის „გ“ ქვეპუნქტის თანახმად [E1], ქურდობა, ჩადენილი ბინაში უკანონო შეღწევით, ისჯება თავისუფლების აღკვეთით ვადით ოთხიდან შვიდ წლამდე.",
  "router_decision": {
    "intent": "hybrid_search",
    "code_slug": "criminal",
    "secondary_codes": [],
    "sub_queries": [
      "ქურდობა ბინაში უკანონო შეღწევა",
      "ქურდობის კვალიფიკაცია სასჯელი"
    ]
  },
  "retrieved_articles": [
    {
      "node_id": "criminal:a177",
      "code_slug": "criminal",
      "article_no": "177",
      "title": "მუხლი 177. ქურდობა",
      "breadcrumb": "კარი IX > თავი XX > მუხლი 177",
      "retrieval_tier": "TIER_1_HYBRID_RRF",
      "rerank_score": 0.942
    }
  ],
  "citation_status": {
    "is_valid": true,
    "marker_integrity": true,
    "cited_markers": ["E1"],
    "valid_markers": ["E1", "E2", "E3", "E4", "E5", "E6"],
    "message": "დადასტურებულია 1 საკანონმდებლო წყარო (E1)."
  },
  "timings": {
    "retrieval_seconds": 0.145,
    "generation_seconds": 1.320,
    "guard_seconds": 0.002,
    "total_seconds": 1.467
  }
}
```

### Other Core Endpoints
- `GET /api/articles/{node_id}` — Returns the complete article AST, including breadcrumbs and text.
- `GET /api/sessions?tester_id={id}` — Fetches all historical threads for a given user.
- `POST /api/sessions` — Creates an isolated dialogue session.
- `GET /api/sessions/{thread_id}/turns` — Fetches full message history and evaluations for a session.
- `POST /api/feedback` — Records user evaluation (1–10 rating, issue tags, expected article hints, notes).
- `GET /api/admin/export` — Exports evaluation database to JSONL for model fine-tuning (Requires `X-Admin-Password`).
- `POST /api/admin/reload` — Forces instant zero-downtime hot-swap of core Python modules.

---

## 🏛️ Dogmatic Methodology & Anti-Hallucination

Georgian Legal AI operates on rigorous Continental legal tenets:

1. **Three-Tier Criminal Structure (*Dreistufiger Deliktsaufbau*):**
   - Criminal evaluation analyzes (1) Statutory Tatbestand (Objective & Subjective), (2) Illegality (*Rechtswidrigkeit* - justifying grounds like Necessity Art. 30 and Self-Defense Art. 28), and (3) Guilt (*Schuld*).
2. **Civil Property & Vindication Dogmatics (Art. 187 Civil Code):**
   - Strict legal differentiation between voluntary entrustment and involuntary loss (theft). If an owner voluntarily lends property, a bona fide purchaser acquires legitimate ownership under Art. 187(1). If stolen, good-faith acquisition is precluded under Art. 187(2) and vindication (Art. 172) remains with the victim.
3. **Strict Citation Verification Guard:**
   - Every claim is anchored by citations `[E1]`–`[E6]`. The citation guard deterministically intercepts fabricated citations, strips unsourced brackets, and prevents prompt injection attacks attempting to force false legal assertions.

---

## 📊 Telemetry & Crowdsourced Feedback

Every user turn, retrieval decision, latency breakdown, and evaluation metric is logged to SQLite (`logs/crowdsourced_eval.db`) and automatically formatted for JSONL export (`logs/feedback_for_ai_analysis.jsonl`).

Testers can rate responses on a 1–10 scale and attach standardized tags:
- `WRONG_CODE` / `MISSED_ARTICLE`
- `HALLUCINATED_DEADLINE` / `BAD_CALCULATION`
- `OUTDATED_STATUTE` / `PROCEDURAL_CONFUSION`

This provides an immediate feedback loop for continuous prompt tuning, retrieval threshold refinement, and downstream Direct Preference Optimization (DPO).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
Statutory text and legislative data are sourced from the Legislative Herald of Georgia ([`matsne.gov.ge`](https://matsne.gov.ge)).
