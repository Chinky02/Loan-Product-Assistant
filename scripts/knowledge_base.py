import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class KnowledgeBase:
    """Handles FAISS index operations and text chunking."""
    
    def __init__(self, kb_dir: Path, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.kb_dir = Path(kb_dir)
        self.index_path = self.kb_dir / "index.faiss"
        self.metadata_path = self.kb_dir / "metadata.jsonl"
        self.embedding_model = embedding_model
        
        self.index: faiss.Index | None = None
        self.metadata: List[Dict[str, Any]] = []
        self.embedder: SentenceTransformer | None = None
        self.dim: int | None = None

    def load(self) -> None:
        """Load the FAISS index, metadata, and embedding model."""
        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError(f"Knowledge base not found in {self.kb_dir}. Build it first.")
        
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = []
        
        with self.metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    self.metadata.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        self.embedder = SentenceTransformer(self.embedding_model)
        self.dim = self.embedder.get_sentence_embedding_dimension()

    def is_ready(self) -> bool:
        """Check if the knowledge base is loaded and ready."""
        return (
            self.index is not None 
            and self.embedder is not None 
            and len(self.metadata) > 0
        )

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks in the knowledge base."""
        if not self.is_ready():
            raise RuntimeError("Knowledge base not loaded")
        
        q_emb = self.embedder.encode(
            [query], 
            convert_to_numpy=True, 
            normalize_embeddings=True
        ).astype("float32")
        
        scores, idxs = self.index.search(q_emb, min(top_k, len(self.metadata)))
        results: List[Dict[str, Any]] = []
        
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            meta = self.metadata[idx].copy()
            meta["score"] = float(score)
            results.append(meta)
        
        return results

    def get_chunk_text(self, metadata: Dict[str, Any]) -> str:
        """Retrieve the actual text content for a chunk."""
        src_path = Path(metadata.get("source", ""))
        key_path = metadata.get("key", "")
        chunk_index = int(metadata.get("chunk_index", 0))
        
        try:
            with src_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            flat = self._flatten_json(data)
            full_text = flat.get(key_path, "")
            
            chunks = self._make_chunks(full_text, 800, 120)
            if 0 <= chunk_index < len(chunks):
                return chunks[chunk_index]
            else:
                return full_text
        except Exception:
            return ""

    @staticmethod
    def _flatten_json(obj: Dict, parent_key: str = "", sep: str = ".") -> Dict[str, str]:
        """Flatten nested JSON structure."""
        items: List[Tuple[str, str]] = []
        
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
                if isinstance(v, (dict, list)):
                    flat_child = KnowledgeBase._flatten_json(v, new_key, sep)
                    for ck, cv in flat_child.items():
                        items.append((ck, cv))
                else:
                    if v is not None:
                        items.append((new_key, str(v)))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
                if isinstance(v, (dict, list)):
                    flat_child = KnowledgeBase._flatten_json(v, new_key, sep)
                    for ck, cv in flat_child.items():
                        items.append((ck, cv))
                else:
                    if v is not None:
                        items.append((new_key, str(v)))
        else:
            if obj is not None:
                items.append((parent_key or "value", str(obj)))
        
        return dict(items)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Normalize whitespace in text."""
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _make_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split text into overlapping chunks."""
        if not text:
            return []
        
        text = KnowledgeBase._normalize_whitespace(text)
        if len(text) <= chunk_size:
            return [text]
        
        chunks: List[str] = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = max(end - chunk_overlap, start + 1)
        
        return chunks
