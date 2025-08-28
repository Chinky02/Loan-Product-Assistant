import os
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import ollama

load_dotenv()

DEFAULT_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
KB_DIR = Path(os.environ.get("KB_DIR", "knowledge_base"))
INDEX_PATH = KB_DIR / "index.faiss"
META_PATH = KB_DIR / "metadata.jsonl"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

app = FastAPI(title="Loan Product Assistant RAG API", version="0.3.1")


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    max_tokens: int = 500
    temperature: float = 0.2


class AskResponse(BaseModel):
    answer: str
    contexts: List[Dict[str, Any]]
    provider: str


class KBState:
    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.metadata: List[Dict[str, Any]] = []
        self.embedder: SentenceTransformer | None = None
        self.dim: int | None = None

    def load(self, model_name: str = DEFAULT_MODEL) -> None:
        if not INDEX_PATH.exists() or not META_PATH.exists():
            raise FileNotFoundError(f"Knowledge base not found in {KB_DIR}. Build it first.")
        self.index = faiss.read_index(str(INDEX_PATH))
        self.metadata = []
        with META_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    self.metadata.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        self.embedder = SentenceTransformer(model_name)
        self.dim = self.embedder.get_sentence_embedding_dimension()

    def is_ready(self) -> bool:
        return self.index is not None and self.embedder is not None and len(self.metadata) > 0

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.is_ready():
            raise RuntimeError("Knowledge base not loaded")
        q_emb = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, idxs = self.index.search(q_emb, min(top_k, len(self.metadata)))
        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            meta = self.metadata[idx].copy()
            meta["score"] = float(score)
            results.append(meta)
        return results


def call_llm_ollama(prompt: str, max_tokens: int = 500, temperature: float = 0.2) -> str:
    model = OLLAMA_MODEL
    sys_msg = (
        "You are a helpful assistant specializing in Bank of Maharashtra loan products. "
        "Answer succinctly using only the provided context. If the answer isn't in the context, say you don't know."
    )
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": prompt},
    ]
    try:
        resp = ollama.chat(model=model, messages=messages, options={"temperature": temperature})
        return resp.get("message", {}).get("content", "")
    except Exception as e:
        msg = str(e).lower()
        if "not found" in msg or "status code: 404" in msg:
            # Fallback to a widely available default model
            fallback = "llama3"
            if model != fallback:
                try:
                    resp = ollama.chat(model=fallback, messages=messages, options={"temperature": temperature})
                    return resp.get("message", {}).get("content", "")
                except Exception:
                    pass
        raise


kb = KBState()


@app.on_event("startup")
def _startup() -> None:
    try:
        kb.load(DEFAULT_MODEL)
    except Exception as e:
        print(f"[WARN] Failed to load KB on startup: {e}")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "kb_ready": kb.is_ready(), "kb_dir": str(KB_DIR), "provider": "ollama", "model": OLLAMA_MODEL}


@app.post("/reload")
def reload_kb() -> Dict[str, Any]:
    kb.load(DEFAULT_MODEL)
    return {"reloaded": True, "kb_ready": kb.is_ready(), "provider": "ollama", "model": OLLAMA_MODEL}


def build_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    ctx_lines: List[str] = []
    for i, c in enumerate(contexts):
        src = c.get("source", "unknown")
        key = c.get("key", "")
        chunk_index = c.get("chunk_index", 0)
        ctx_text = c.get("text", "")
        ctx_lines.append(f"[Context {i+1}] (src={src} | key={key} | chunk={chunk_index})\n{ctx_text}")
    joined_ctx = "\n\n".join(ctx_lines)
    prompt = (
        f"Context:\n{joined_ctx}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    return prompt


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if not kb.is_ready():
        raise HTTPException(status_code=503, detail="Knowledge base not loaded. Build and/or reload it.")
    hits = kb.search(req.question, req.top_k)

    enriched: List[Dict[str, Any]] = []
    for h in hits:
        src_path = Path(h.get("source", ""))
        key_path = h.get("key", "")
        chunk_index = int(h.get("chunk_index", 0))
        text = ""
        try:
            with src_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            def _flatten(obj, parent_key="", sep="."):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        nk = f"{parent_key}{sep}{k}" if parent_key else str(k)
                        if isinstance(v, (dict, list)):
                            yield from _flatten(v, nk, sep)
                        else:
                            yield nk, str(v)
                elif isinstance(obj, list):
                    for i, v in enumerate(obj):
                        nk = f"{parent_key}{sep}{i}" if parent_key else str(i)
                        if isinstance(v, (dict, list)):
                            yield from _flatten(v, nk, sep)
                        else:
                            yield nk, str(v)
            flat = dict(_flatten(data))
            full_text = flat.get(key_path, "")
            from scripts.build_knowledge_base import make_chunks
            chunks = make_chunks(full_text, 800, 120)
            if 0 <= chunk_index < len(chunks):
                text = chunks[chunk_index]
            else:
                text = full_text
        except Exception:
            pass
        enriched.append({**h, "text": text})

    prompt = build_prompt(req.question, enriched)
    answer = call_llm_ollama(prompt, max_tokens=req.max_tokens, temperature=req.temperature)
    return AskResponse(answer=answer, contexts=enriched, provider="ollama") 