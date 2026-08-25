# Sales FAQ Bot

![python-image] ![fastapi-image] ![Qdrant-image] ![HTML-image] ![HuggingFace-image]

> An intelligent Q&A system built on RAG (Retrieval-Augmented Generation) architecture, designed to provide fast and accurate answers to sales' FAQ. The system integrates vector retrieval, semantic search, and generative AI, offering a user-friendly web interface for real-time queries.

![5kome-utkoy](https://github.com/user-attachments/assets/73d561b0-fa34-4160-9587-0cc17f15a4db)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Prerequisites](#-prerequisites)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Development](#-development)
- [Testing](#-testing)
- [Contact](#-contact)

---

## 🎯 Overview

Sales FAQ Bot is an POC Q&A system designed to solve various questions that sales encounter in their daily work. Through advanced RAG technology, the system can:

- **Fast Retrieval**: Accurately find relevant documents from the knowledge base
- **Semantic Understanding**: Use LLM models to understand user query intent
- **Intelligent Answers**: Combine retrieval results with LLM to generate natural and fluent responses
- **Real-time Interaction**: Provide a demo web chat interface

The system uses FastAPI as the backend framework, Qdrant as the vector database, and integrates the BGE-M3 embedding model with Azure OpenAI to achieve high-performance semantic search and intelligent Q&A.

---

## 📦 Prerequisites

Ensure your development environment meets the following requirements:

| Requirement | Version | Description |
|---------|------|------|
| Python | 3.12+ | Core runtime environment |
| uv | 0.9.17+ | Lightweight package manager for environment and dependency building |

### API Keys

The system requires the following API Keys (configure in `.env` file):

- `AZURE_OPENAI_API_KEY` Azure OpenAI API key (for Azure OpenAI)
- `AZURE_OPENAI_ENDPOINT` Azure OpenAI endpoint URL
- `AZURE_OPENAI_API_VERSION` Azure OpenAI API version (optional, defaults to 2024-12-01-preview)
- `HUGGINGFACE_LLM_Model` HuggingFace access token (if using private models)

---

## 🚀 Getting Started

### 1. Clone the Project

```bash
# Clone the project (or download the project archive)
git clone https://github.com/xinhuang0716/salesFAQBot.git
cd salesFAQBot
```

### 2. Configure Environment Variables

Create a `.env` file in the project root directory:

```bash
# .env
# For Azure OpenAI API
AZURE_OPENAI_API_KEY=your_azure_openai_key_here
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint_here
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# For HuggingFace (if using private models)
HUGGINGFACE_LLM_Model=your_huggingface_token_here
```

### 3. Prepare Knowledge Base

Place your FAQ knowledge base Excel file in the `knowledgeDoc/` folder.

**File Format Requirements:**
- File type: `.xlsx`
- Required columns: `id`, `source`, `topic`, `subtype`, `relevance`

Example:
| id | source | topic | subtype | relevance |
|----|--------|-------|---------|-----------|
| 1  | Internal Document | Account Opening Process | Online Account Opening | How to apply for online account opening? |

### 4. Configure System Parameters

Edit `config/config.yaml` to adjust system settings (optional):

```yaml
embedder:
  repo: "BAAI/bge-m3"  # Can switch to other embedding models

retriever:
  top_k: 3              # Number of documents to retrieve
  score_threshold: 0.5  # Similarity threshold
```

### 5. Start the Service

By default, the virtual ennvironment and dependencies will be automatically created when starting the server for the first time.

```powershell
# Start FastAPI server using uv
uv run main.py
```

After successful startup, you will see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8010
```

### 6. Access the Application

Open your browser and visit:
```
http://localhost:8010
```

---

## 💻 Usage

### Web Interface Usage

1. **Open Application**: Visit `http://localhost:8010` in your browser
2. **Enter Question**: Type your question in the chat box
3. **Get Answer**: The system will automatically retrieve relevant documents and generate answers

### API Usage

#### 1. Retrieve Relevant Documents

```bash
curl -X POST "http://localhost:8010/retrieveDocs" \
  -H "Content-Type: application/json" \
  -d '{"message": "如何進行線上開戶？"}'
```

**Response Example:**
```json
{
  "response": "[{'rank': 1, 'doc_id': 1, 'score': 0.85, 'topic': '開戶流程', ...}]",
  "status": "success"
}
```

#### 2. Get Intelligent Answer

```bash
curl -X POST "http://localhost:8010/response" \
  -H "Content-Type: application/json" \
  -d '{"message": "如何進行線上開戶？"}'
```

**Response Example:**
```json
{
  "response": "線上開戶流程如下：\n1. 準備身分證與...",
  "status": "success"
}
```

---

## 📂 Project Structure

```
salesFAQBot/
│
├── config/                          # Configuration files
│   └── config.yaml                  # Main system configuration
│
├── core/                            # Core functionality modules
│   ├── __init__.py
│   │
│   ├── init/                        # Initialization module
│   │   ├── builder.py               # System initialization builder
│   │   └── database.py              # Qdrant database initialization
│   │
│   ├── embedder/                    # Embedding module
│   │   ├── base_embedder.py         # Embedder base class (abstract)
│   │   ├── sentence_transformer_embedder.py  # ST embedder implementation
│   │   └── bm25.py                  # BM25 sparse embedder
│   │
│   ├── reranker/                    # Reranking module
│   │   ├── base_reranker.py         # Reranker base class (abstract)
│   │   └── sentence_transformer_reranker.py  # ST reranker implementation
│   │
│   ├── retrieve/                    # Retrieval module
│   │   ├── dense_search.py          # Dense vector retrieval
│   │   ├── bm25_search.py           # BM25 sparse retrieval
│   │   └── rerank_search.py         # Reranking retrieval
│   │
│   └── response/                    # Response generation module
│       ├── aoai.py                  # Azure OpenAI API integration
│       └── prompt.py                # Prompt constructor
│
├── db/                              # Vector database (auto-generated)
│   ├── meta.json                    # Qdrant metadata
│   └── collection/                  # Vector collection storage
│       └── FAQ/                     # FAQ collection data
│
├── knowledgeDoc/                    # Knowledge base source files
│   └── *.xlsx                       # FAQ data in Excel format
│
├── logs/                            # Log files (auto-generated)
│   └── app.log                      # Application logs
│
├── models/                          # AI model files (auto-downloaded)
│   ├── bge-m3/                      # BGE-M3 embedding model
│   ├── jina-reranker-v2-base-multilingual/  # Jina reranker
│   └── ...                          # Other models
│
├── static/                          # Static resources
│   ├── css/
│   │   └── style.css                # Web styles
│   └── js/
│       └── script.js                # Frontend interaction logic
│
├── template/                        # HTML templates
│   └── index.html                   # Main chat interface
│
├── test/                            # Test scripts
│   ├── test_bm25.py                 # BM25 tests
│   └── test_reranker.py             # Reranker tests
│
├── test_records/                    # Test records
│
├── utils/                           # Utility functions
│   └── corpus.py                    # Text processing utilities
│
├── .env                             # Environment variables (create manually)
├── requirements.txt                 # Python dependencies
├── main.py                          # FastAPI main program
└── README.md                        # Project documentation
```

### Core Module Descriptions

#### `core/init/`
- **builder.py**: Handles initialization flow during system startup, including loading knowledge base, creating embedding vectors, and initializing database
- **database.py**: Encapsulates Qdrant database creation and configuration logic

#### `core/embedder/`
- **base_embedder.py**: Defines the abstract interface for embedders
- **sentence_transformer_embedder.py**: Implementation based on Sentence Transformers
- **bm25.py**: Implements BM25 sparse embedding

#### `core/retrieve/`
- **dense_search.py**: Semantic vector retrieval (primary method)
- **bm25_search.py**: Keyword retrieval
- **rerank_search.py**: Retrieval result reranking

#### `core/response/`
- **aoai.py**: Interacts with Azure OpenAI API to generate final answers
- **prompt.py**: Constructs RAG prompt templates using class-based approach with cached properties

---

## ⚙️ Configuration

### config.yaml Detailed Description

```yaml
# FastAPI server settings
server:
  cors_origins: ["*"]          # CORS allowed origins (restrict in production)
  max_message_length: 1024     # Maximum message length
  cors_max_age: 3600           # CORS preflight request cache time (seconds)

# Vector database settings
db:
  collection_name: "FAQ"       # Qdrant collection name

# Embedding model settings
embedder:
  type: "sentence_transformer"            # Embedder type
  repo: "BAAI/bge-m3"                    # HuggingFace model repository

# Retrieval settings
retriever:
  top_k: 3                     # Number of documents to retrieve
  score_threshold: 0.5         # Minimum similarity score threshold

# Reranking settings (optional)
reranker:
  apply: False                 # Whether to enable reranking
  type: "sentence_transformer" # Reranker type
  repo: "jinaai/jina-reranker-v2-base-multilingual"
  top_k: 3                     # Number to keep after reranking
  score_threshold: 0.5         # Reranking score threshold
```

---

## 🔧 Development

### LLM Provider

The system uses **Azure OpenAI API** via [aoai.py](core/response/aoai.py).

It uses the unified `prompt_template` class from [prompt.py](core/response/prompt.py):

```python
from core.response.prompt import prompt_template

# System prompt is cached for efficiency using @cached_property
system_prompt = prompt_template().system_prompt

# Construct user prompt with retrieved documents
user_prompt = prompt_template.construct(query, top_k_docs)
```

### Extending Embedding Models

1. Inherit from `BaseEmbedder` base class:

```python
# core/embedder/my_embedder.py
from core.embedder.base_embedder import BaseEmbedder

class MyEmbedder(BaseEmbedder):
    def __init__(self, config):
        # Initialization logic
        pass
    
    def encode(self, texts, encode_type):
        # Embedding logic
        return embeddings
```

2. Configure in `config.yaml`:

```yaml
embedder:
  type: "my_embedder"
  # Other configurations...
```

### Extending Retrieval Strategies

Add new retrieval classes in the `core/retrieve/` directory:

```python
# core/retrieve/my_search.py
class MySearcher:
    def search(self, query, k=3, score=0.4):
        # Retrieval logic
        return results
```

### Log Management

Logs are output to:
- **Console**: Real-time viewing
- **File**: `logs/app.log`

Log level: INFO (can be adjusted in [main.py](main.py))

---

## 🧪 Testing

### Running Tests

```powershell
# Test BM25 retrieval
uv run test\test_bm25.py

# Test reranker
uv run test\test_reranker.py

# Test API responses
uv run test\test_response.py
```
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
[HuggingFace-image]: https://img.shields.io/badge/-HuggingFace-3B4252?style=flat&logo=huggingface&logoColor=
[HTML-image]: https://img.shields.io/badge/html-%23E34F26?style=for-the-badge&logo=html5&logoColor=%23fff
