# Engineering Onboarding and Architecture Guidelines

## Architecture Overview
The Enterprise Knowledge Copilot uses a modular Retrieval-Augmented Generation (RAG) architecture. Document ingestion parses PDFs, DOCX, TXT, and Markdown files into structured chunks. These chunks are converted into dense vector embeddings using sentence-transformers and indexed in ChromaDB with cosine distance metric.

## Environment and Dependency Setup
All dependencies are pinned in requirements.txt. Developers must use Python 3.13.5 with an isolated virtual environment located at .venv. Test automation is executed using pytest.

## Coding and Contribution Standards
Direct modifications to ingestion or retrieval interfaces require prior review and regression testing. All unit tests must complete with zero failures before committing code. Async methods must be tested using pytest-asyncio.
