# Environment and Dependency Compatibility

## Verified Environment

- Operating system: Windows, 64-bit
- Python: 3.13.5, 64-bit
- Virtual environment: `.venv`
- Dependency manifest: `requirements.txt`
- Local LLM runtime: Ollama
- Local generation model: `qwen2.5:7b`

## Pinned Direct Dependencies

| Package | Version |
|---|---:|
| fastapi | 0.149.0 |
| uvicorn | 0.54.0 |
| streamlit | 1.66.0 |
| pydantic | 2.14.1 |
| chromadb | 1.5.9 |
| sentence-transformers | 6.1.0 |
| transformers | 5.20.2 |
| tokenizers | 0.23.3 |
| torch | 2.14.1 |
| pypdf | 6.21.1 |
| python-docx | 1.2.0 |
| rank-bm25 | 0.2.2 |
| nltk | 3.10.3 |
| python-dotenv | 1.2.4 |
| python-multipart | 0.0.32 |
| pytest | 9.1.1 |
| httpx | 0.28.1 |

## Checks Completed

- Dependency installation completed successfully in `.venv`.
- `pip check` returned `No broken requirements found.`
- Imports for all 17 direct dependencies succeeded.
- `app/core/config.py` passed Python syntax compilation.
- Default configuration loaded with chunk size 200 and overlap 40.
- The Ollama API responded during environment discovery.
- The local `qwen2.5:7b` model was available.

These checks verify the environment and basic configuration only. The RAG pipeline, API, UI, guardrails, and evaluation suite have not been implemented or tested yet.

## Design Constraints

- Configure ChromaDB explicitly to use cosine distance.
- Count chunks using the selected embedding model's tokenizer.
- Start with 200-token chunks and 40-token overlap; compare alternatives during evaluation.
- Keep reranking optional.
- Calibrate abstention thresholds separately for each retrieval mode.
- Use `pypdf` as the initial PDF parser.
- Report benchmark results only after actual runs.

## Reproducibility

Direct dependency versions are pinned in `requirements.txt`. To verify a future installation, run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

The installed environment can be recorded with:

```powershell
.\.venv\Scripts\python.exe -m pip freeze
```

A fresh environment should be tested before claiming reproducibility on another machine.