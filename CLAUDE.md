# CLAUDE.md — AI Chatbot with RAG on Azure

## Project Overview

An AI-powered chatbot with two modes:
- **General Query**: Answers questions directly via Llama 3 (8B) on Azure AI
- **Document RAG**: Users upload PDF/TXT → Cohere embeddings stored in ChromaDB → LlamaIndex retrieves context → Llama 3 answers

**Microservices architecture**: FastAPI backend + Streamlit frontend, both containerized and deployed on Azure Container Apps via GitHub Actions CI/CD.

---

## Architecture

```
Streamlit Frontend  (port 8501)
        ↓  HTTP REST
FastAPI Backend     (port 8000)
        ↓
ChromaDB (./chroma_db/) ← Cohere embeddings (Azure AI)
        ↓
Llama 3 8B (Azure AI)
```

### Backend Endpoints
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/upload-document/` | Upload PDF/TXT, create ChromaDB vector index |
| POST | `/document-query/` | RAG-based Q&A against uploaded document |
| POST | `/general-query/` | Direct LLM Q&A (no document context) |
| GET | `/status/` | Check if a document is currently loaded |
| GET | `/clear-index/` | Delete ChromaDB collection and reset state |

---

## Running Locally

### Prerequisites
- Docker and Docker Compose installed
- A `.env` file at the project root (see Environment Variables below)

### Start everything with Docker Compose
```bash
docker-compose up --build
```
- Backend: http://localhost:8000
- Frontend: http://localhost:8501

The frontend waits for the backend health check to pass before starting.

### Run backend only (without Docker)
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Run frontend only (without Docker)
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```
> Note: When running locally without Docker, update `FASTAPI_BASE_URL` in [frontend/app.py:31](frontend/app.py#L31) to `http://127.0.0.1:8000`.

---

## Environment Variables

Create a `.env` file at the project root:
```env
AZURE_META_ENDPOINT=<your-llama3-azure-endpoint>
AZURE_META_API=<your-llama3-api-key>
AZURE_COHERE_ENDPOINT=<your-cohere-azure-endpoint>
AZURE_COHERE_API=<your-cohere-api-key>
```

These are injected into the backend container via `docker-compose.yml`. The frontend has no env vars — the backend URL is hardcoded in [frontend/app.py:31](frontend/app.py#L31).

---

## Key Files

| File | Role |
|------|------|
| [backend/main.py](backend/main.py) | FastAPI app, endpoint definitions, global index state |
| [backend/chat.py](backend/chat.py) | LLM + embedding model init, ChromaDB connection |
| [backend/query_type.py](backend/query_type.py) | General and document query handlers (LlamaIndex) |
| [backend/gunicorn.conf.py](backend/gunicorn.conf.py) | Production server config (1 Uvicorn worker) |
| [frontend/app.py](frontend/app.py) | Streamlit UI, file upload, chat history, session state |
| [docker-compose.yml](docker-compose.yml) | Local orchestration with health checks |
| [.github/workflows/](.github/workflows/) | CI/CD: push to `main` → build image → push to GHCR → deploy to Azure Container Apps |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend framework | FastAPI (Python 3.11) |
| Frontend framework | Streamlit |
| LLM | Llama 3 8B via Azure AI (`llama-index-llms-azure-inference`) |
| Embeddings | Cohere via Azure AI (`llama-index-embeddings-azure-inference`) |
| RAG framework | LlamaIndex (`llama-index-core`) |
| Vector store | ChromaDB (persistent at `./chroma_db/`) |
| Production server | Gunicorn + Uvicorn workers |
| Containerization | Docker (python:3.11-slim) |
| Container registry | GitHub Container Registry (`ghcr.io/magesh-babu`) |
| Hosting | Azure Container Apps — Sweden Central |
| CI/CD | GitHub Actions |

---

## Azure Resources

| Resource | Value |
|----------|-------|
| Resource Group | `resource_apr_2026` |
| Container Apps Environment | `app-environment` |
| Backend Container App | `ai-rag-backend` |
| Frontend Container App | `ai-rag-frontend` |
| Container Registry | GitHub Container Registry — `ghcr.io/magesh-babu` |
| LLM Model | Llama 3 (8B) on Azure AI Services |
| Embedding Model | Cohere on Azure AI Services |

**Important:** After first deploy, update the backend URL in [frontend/app.py:31](frontend/app.py#L31) to the new `*.azurecontainerapps.io` URL shown in the portal.

---

## Known Limitations & Technical Debt

- **Hardcoded backend URL** — `FASTAPI_BASE_URL` in [frontend/app.py:31](frontend/app.py#L31) is hardcoded to the production Azure URL. Should be an env variable.
- **Global in-memory state** — `global_index` and `global_document_name` in [backend/main.py:32-33](backend/main.py#L32-L33) are process-level globals. This means multi-worker deployments would have inconsistent state. Currently safe because Gunicorn is configured to 1 worker ([backend/gunicorn.conf.py:12](backend/gunicorn.conf.py#L12)).
- **No selective document clearing** — `/clear-index/` deletes the entire `given_doc` ChromaDB collection. There is no way to swap documents without a full reset.
- **Single worker bottleneck** — 1 Gunicorn worker handles all requests sequentially. Fine for low traffic; will bottleneck under concurrent users.
- **Azure Container Apps env vars** — `AZURE_META_ENDPOINT`, `AZURE_META_API`, `AZURE_COHERE_ENDPOINT`, `AZURE_COHERE_API` must be added manually in portal: Container App → Settings → Environment variables. The workflow does not set these.

---

## CI/CD Pipeline

Push to `main` branch triggers GitHub Actions workflows:
1. Build Docker image for backend or frontend
2. Push to GitHub Container Registry (`ghcr.io/magesh-babu/backend:<sha>` or `frontend:<sha>`)
3. Azure login using `AZURE_CREDENTIALS` service principal
4. Deploy to Azure Container Apps in `resource_apr_2026` using `azure/container-apps-deploy-action@v2`

**GitHub Secrets required:** `AZURE_CREDENTIALS` (service principal JSON), `GHCR_PAT` (PAT with `read:packages` scope)

Workflows: [.github/workflows/main_ai-backend.yml](.github/workflows/main_ai-backend.yml) and [.github/workflows/main_ai-frontend.yml](.github/workflows/main_ai-frontend.yml)
