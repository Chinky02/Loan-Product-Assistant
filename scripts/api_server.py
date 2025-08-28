import os
import logging
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from scripts.knowledge_base import KnowledgeBase
from scripts.llm_service import LLMService
from scripts.rag_service import RAGService, AskRequest, AskResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Configuration
DEFAULT_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
KB_DIR = Path(os.environ.get("KB_DIR", "knowledge_base"))
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

# Initialize services
knowledge_base = KnowledgeBase(KB_DIR, DEFAULT_EMBEDDING_MODEL)
llm_service = LLMService(OLLAMA_MODEL)
rag_service = RAGService(knowledge_base, llm_service)

# FastAPI app
app = FastAPI(
    title="Loan Product Assistant RAG API", 
    version="0.4.0",
    description="A RAG-based API for answering questions about Bank of Maharashtra loan products"
)


@app.on_event("startup")
def startup_event():
    """Initialize services on startup."""
    try:
        knowledge_base.load()
        logger.info("Knowledge base loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load knowledge base: {e}")


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "kb_ready": knowledge_base.is_ready(),
        "kb_dir": str(KB_DIR),
        "provider": "ollama",
        "model": OLLAMA_MODEL,
        "embedding_model": DEFAULT_EMBEDDING_MODEL
    }


@app.post("/reload")
def reload_knowledge_base() -> Dict[str, Any]:
    """Reload the knowledge base."""
    try:
        knowledge_base.load()
        logger.info("Knowledge base reloaded successfully")
        return {
            "reloaded": True,
            "kb_ready": knowledge_base.is_ready(),
            "provider": "ollama",
            "model": OLLAMA_MODEL
        }
    except Exception as e:
        logger.error(f"Failed to reload knowledge base: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload: {e}")


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    """Ask a question and get an answer."""
    try:
        if not knowledge_base.is_ready():
            raise HTTPException(
                status_code=503, 
                detail="Knowledge base not loaded. Please reload it first."
            )
        
        response = rag_service.process_query(request)
        logger.info(f"Processed query: {request.question[:50]}...")
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "Loan Product Assistant RAG API",
        "version": "0.4.0",
        "docs": "/docs",
        "health": "/health"
    }
