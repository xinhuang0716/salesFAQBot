# Sales FAQ Bot

<img width="892" height="686" alt="圖片" src="https://github.com/user-attachments/assets/9784355e-c962-4644-ab93-83789ab00a23" />

## Prerequisites

| Requirement | Version |
| :---------: | :-----: |
|   Python    |  3.12   |
|     pip     | 25.0.1  |

## Project Strucure

```
salesFAQBot/
├── backend/
│   ├── pipeline.py             # RAG 文件檢索流程
│   ├── main.py
│   ├── corpus_builder.py       # 文本處理
│   ├── database_builder.py     # 初始化QDrant、建立資料庫 client instance
│   ├── index_builder.py        # 向量化文件
│   ├── dense_search.py         # 向量檢索
│   ├── bm25_search.py          # BM25檢索
│   ├── hybrid_search.py        # Hybrid Search檢索
│   └── embedder/
│       ├── base_embedder.py    
│       └── bge_embedder.py     # BAAI/bge-m3 模型
│   └── utils/
│       ├── database.py         # 初始化QDrant、建立資料庫 client instance
│       ├── denseVec.py         # 向量化文件
│       └── BM25.py             # BM25初始化、斷詞、BM25檢索
├── frontend/
│   └── index.html
├── .env
├── requirements.txt
└── README.md
```

## Installation

- Create and activate a virtual environment (optional but recommended):

  ```bash
  # powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```

- Install the required packages:

  ```bash
  # powershell
  pip install -r requirements.txt
  ```

## Usage

- activate the virtual environment (if created):

  ```bash
  # powershell
  .venv\Scripts\Activate.ps1
  ```

- run the RAG pipeline

  ```bash
  # powershell
  cd backend
  python pipeline.py
  ```

## Help

Shall you have any problem, please let me knows. Look forward to your feedbacks and suggestions!

```
Email: tom.h.huang@fubon.com, kris.yj.chen@fubon.com
Tel:   02-87716888 #69175, 02-66080879 #69194
Dept:  證券 數據科學部 資料服務處(5F)
```



