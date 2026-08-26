# Sales FAQ Bot

![python-image] ![fastapi-image] ![Qdrant-image] ![HTML-image] ![HuggingFace-image]

> An internal RAG (Retrieval-Augmented Generation) chatbot for sales FAQ scenarios. The system combines dense retrieval, BM25 retrieval, hybrid fusion, reranking, and Azure OpenAI response generation.

![5kome-utkoy](https://github.com/user-attachments/assets/73d561b0-fa34-4160-9587-0cc17f15a4db)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Prerequisites](#-prerequisites)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Contact](#-contact)

---

## 🎯 Overview

Sales FAQ Bot is a proof-of-concept Q&A assistant for internal sales operations. It supports:

- **Dense Retrieval**: Semantic vector search using `BAAI/bge-m3`
- **BM25 Retrieval**: Sparse keyword-based retrieval
- **Hybrid Retrieval**: Reciprocal Rank Fusion (RRF) of dense and BM25 results
- **Reranking**: Cross-encoder reranking using `BAAI/bge-reranker-base`
- **RAG Answering**: Final response generation through Azure OpenAI

The backend is built with FastAPI, local vector storage is provided by Qdrant (file-based), and model assets are cached under `models/`.

---

## 📦 Prerequisites

Ensure your development environment meets the following requirements:

| Requirement | Version | Description                                |
| ----------- | ------- | ------------------------------------------ |
| Python      | 3.12+   | Core runtime environment                   |
| uv          | 0.9.17+ | Package and virtual environment management |

### Environment Variables

Runtime secrets are loaded from `config/.env`.

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_DEPLOYMENT`

---

## 🚀 Getting Started

### 1. Clone the Project

```bash
git clone https://github.com/xinhuang0716/salesFAQBot.git
cd salesFAQBot
```

### 2. Configure Environment Variables

Copy and edit the environment template:

```powershell
Copy-Item config/.env.example config/.env
```

Then update `config/.env`:

```bash
AZURE_OPENAI_API_KEY=your_azure_openai_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=your_deployment_name
```

### 3. Prepare Knowledge Base

Put your Excel knowledge file in `knowledgeDoc/`.

Required columns:

- `id`: Unique identifier for each knowledge chunk
- `source`: The source document or reference
- `topic`: The main topic of the knowledge chunk
- `subtype`: The subcategory or type of the knowledge chunk
- `relevance`: The knowledge content or answer text

### 4. Start the Service

```powershell
uv run main.py
```

After startup, service runs at:

```text
http://localhost:8000
```

Then access the following endpoints:

- Chat UI: `http://localhost:8000/`
- API Docs (Swagger): `http://localhost:8000/docs`

---

## 💻 Usage

### Web Interface Usage

1. Open `http://localhost:8000/`
2. Enter your question in the chat box
3. The frontend calls `/rag-response/` and renders markdown output

### API Usage

All major API responses use a unified envelope:

```json
{
  "status": "success",
  "data": {}
}
```

#### 1. Health Check

```bash
curl -X GET "http://localhost:8000/health/"
```

Response example:

```json
{
  "status": "success",
  "data": {
    "service": "ok"
  }
}
```

#### 2. Dense Retrieval

```bash
curl -X POST "http://localhost:8000/dense-retrieve/" \
  -H "Content-Type: application/json" \
  -d '{"message":"如何進行線上開戶？"}'
```

Response example:

```json
{
  "status": "success",
  "data": [
    {
      "point_id": 12,
      "dense_rank": 1,
      "dense_score": 0.83,
      "topic": "開戶流程",
      "subtype": "線上開戶",
      "relevance": "..."
    }
  ]
}
```

#### 3. RAG Response

```bash
curl -X POST "http://localhost:8000/rag-response/" \
  -H "Content-Type: application/json" \
  -d '{"message":"如何進行線上開戶？"}'
```

Response example:

```json
{
  "status": "success",
  "data": {
    "response": "### 回答\n...",
    "references": [
      {
        "point_id": 12,
        "hybrid_rank": 1,
        "rerank_rank": 1,
        "topic": "開戶流程",
        "subtype": "線上開戶",
        "relevance": "..."
      }
    ]
  }
}
```

#### 4. Other Retrieval Endpoints

- `POST /bm25-retrieve/`
- `POST /hybrid-retrieve/`
- `POST /reranker/`

These endpoints also accept:

```json
{
  "message": "your query"
}
```

and return:

```json
{
  "status": "success",
  "data": []
}
```

---

## 📂 Project Structure

```text
salesFAQBot/
|
|-- config/                         # Runtime settings and environment files
|   |-- .env                        # Local secrets (not committed)
|   |-- .env.example                # Environment variable template
|   `-- config.yaml                 # Retrieval and reranker parameters
|
|-- core/                           # Core retrieval and generation logic
|   |-- aoai.py                     # Azure OpenAI request/response client
|   |-- bm25.py                     # BM25 index build and sparse retrieval
|   |-- context.py                  # Prompt construction and document formatting
|   |-- dense_search.py             # Dense vector retrieval over Qdrant
|   |-- embedder.py                 # Sentence-transformer embedding wrapper
|   |-- hybrid_search.py            # RRF fusion between dense and BM25
|   `-- reranker.py                 # Cross-encoder reranking
|
|-- infra/                          # Infrastructure and bootstrapping utilities
|   |-- database.py                 # Local Qdrant initialization helpers
|   |-- indexer.py                  # Excel loading and vectorization pipeline
|   `-- settings.py                 # Typed settings loader from yaml + env
|
|-- routers/                        # FastAPI route handlers
|   |-- bm25_retrieve.py            # BM25 retrieval endpoint
|   |-- dense_retrieve.py           # Dense retrieval endpoint
|   |-- health.py                   # Health check endpoint
|   |-- hybrid_retrieve.py          # Hybrid retrieval endpoint
|   |-- pages.py                    # Landing page route
|   |-- rag_response.py             # End-to-end RAG answer endpoint
|   |-- reranker_retrieve.py        # Dense + reranker endpoint
|   `-- schemas.py                  # Shared request/response models
|
|-- knowledgeDoc/                   # Source Excel knowledge files (*.xlsx)
|-- models/                         # Downloaded embedding/reranker model assets
|-- static/                         # Frontend assets (css/js)
|-- template/                       # HTML templates
|-- db/                             # Local Qdrant data directory
|
|-- main.py                         # FastAPI app entry and lifecycle wiring
|-- pyproject.toml                  # Project metadata and dependencies
|-- uv.lock                         # Locked dependency versions for uv
`-- README.md                       # Project documentation
```

---

## ⚙️ Configuration

### `config/config.yaml`

```yaml
retrieval:
  top_k: 16
  score_threshold: 0.5
  hybrid_top_k: 8
  rrf_k: 60

reranker:
  top_k: 3
  score_threshold: 0.3
```

### Parameter Description

#### `retrieval`

- `top_k`: Number of top documents retrieved from each base retriever (dense and BM25) before fusion.
- `score_threshold`: Minimum similarity score for dense retrieval. Set `null` to disable threshold filtering.
- `hybrid_top_k`: Final number of documents returned after hybrid fusion ranking.
- `rrf_k`: Smoothing constant used by Reciprocal Rank Fusion. Larger values reduce rank-gap impact.

#### `reranker`

- `top_k`: Number of documents kept after cross-encoder reranking.
- `score_threshold`: Minimum reranker score to keep a candidate. Set `null` to keep by rank only.

---

## 📞 Contact

For any questions or suggestions, please contact:

**Project Team**

- **Email**: tom.h.huang@fubon.com, kris.yj.chen@fubon.com
- **Phone**: 02-87716888 #69175, 02-66080879 #69194
- **Department**: Securities Data Science Department, Data Service Division (5F)

**Issue Reporting**

- Please submit bugs or feature requests via GitHub Issues
- Pull Request contributions are welcome

---

## 📄 License

This project is for internal use only. Copyright belongs to Fubon Securities Data Science Department.

<!-- Markdown link & img dfn's -->

[python-image]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[fastapi-image]: https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi
[Qdrant-image]: https://img.shields.io/badge/Qdrant-Vector%20DB-FF6B6B?style=for-the-badge
[HuggingFace-image]: https://img.shields.io/badge/-HuggingFace-3B4252?style=for-the-badge&logo=huggingface&logoColor=
[HTML-image]: https://img.shields.io/badge/html-%23E34F26?style=for-the-badge&logo=html5&logoColor=%23fff
