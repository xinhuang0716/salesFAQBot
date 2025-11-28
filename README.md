# Sales FAQ Bot

<img width="650" height="500" alt="demo" src="https://github.com/user-attachments/assets/9784355e-c962-4644-ab93-83789ab00a23" />

## Prerequisites

| Requirement | Version |
| :---------: | :-----: |
|   Python    |  3.12   |
|     pip     | 25.0.1  |

## Project Strucure

```
salesFAQBot/
├── core/
│   └── init/
│       ├── model.py
│       ├── bm25.py
│       ├── db.py 
│       └── embedding.py
│   └── embedder/
│       ├── base_embedder.py
│       ├── xxx_embedder.py 
│       └── yyy_embedder.py
│   └── reranker/
│       ├── base_reranker.py    
│       └── xxx_reranker.py
│   └── retrieve/ 
│       └── pipeline.py
│   └── response/
│       ├── prompt.py    
│       └── aoai.py
├── utils/
│   ├── datapreprocessing.py
│   └── *.py
├── template/
│   └── index.html
├── static/
│   └── css/
│       └── style.css
│   └── js/
│       └── script.js   
├── server.py
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




