# Loan-Product-Assistant

## Overview
A RAG-based Loan Product Assistant for Bank of Maharashtra that answers questions about loan products using a knowledge base built from scraped data.

## Architecture
The system is now refactored into modular components:

- **KnowledgeBase** (`scripts/knowledge_base.py`): Handles FAISS index operations and text chunking
- **LLMService** (`scripts/llm_service.py`): Manages Ollama LLM interactions with fallback support
- **RAGService** (`scripts/rag_service.py`): Coordinates knowledge base and LLM for query processing
- **API Server** (`scripts/api_server.py`): FastAPI server with proper error handling and logging

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Build Knowledge Base
```bash
python scripts/build_knowledge_base.py --processed_dir data/processed --output_dir knowledge_base
```

### 3. Setup Ollama
```bash
# Install Ollama from https://ollama.com/
ollama pull llama3
```

### 4. Run the API
```bash
# Using the main API server (recommended)
uvicorn scripts.api_server:app --host 0.0.0.0 --port 8000 --reload

# Or using the legacy version
uvicorn scripts.serve_api:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check with service status
- `POST /reload` - Reload knowledge base
- `POST /ask` - Ask questions about loan products
- `GET /docs` - Interactive API documentation

## Example Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Ask a Question
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Tell me about the Maha Super Flexi Housing Loan Scheme.","top_k":5}'
```

## Configuration

Set environment variables:
```bash
OLLAMA_MODEL=llama3                    # Ollama model to use
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2  # Embedding model
KB_DIR=knowledge_base                  # Knowledge base directory
```

## Features

- **Modular Architecture**: Clean separation of concerns
- **Error Handling**: Comprehensive error handling with logging
- **Fallback Support**: Automatic fallback to alternative models
- **Health Monitoring**: Detailed health checks
- **Interactive Docs**: FastAPI automatic documentation
- **Free LLM**: Uses local Ollama models (no API costs)

## File Structure

```
scripts/
 build_knowledge_base.py      # Build FAISS knowledge base
 knowledge_base.py            # Knowledge base operations
 llm_service.py              # LLM service with Ollama
 rag_service.py              # RAG coordination service
 serve_api.py                # Legacy API server
 api_server.py               # Main API server (recommended)
```
