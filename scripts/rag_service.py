from typing import Dict, List, Any
from pydantic import BaseModel


class AskRequest(BaseModel):
    """Request model for asking questions."""
    question: str
    top_k: int = 5
    max_tokens: int = 500
    temperature: float = 0.2


class AskResponse(BaseModel):
    """Response model for answers."""
    answer: str
    contexts: List[Dict[str, Any]]
    provider: str


class RAGService:
    """Main RAG service that coordinates knowledge base and LLM."""
    
    def __init__(self, knowledge_base, llm_service):
        self.kb = knowledge_base
        self.llm = llm_service

    def build_prompt(self, question: str, contexts: List[Dict[str, Any]]) -> str:
        """Build a prompt from question and contexts."""
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

    def process_query(self, request: AskRequest) -> AskResponse:
        """Process a query and return a response."""
        if not self.kb.is_ready():
            raise RuntimeError("Knowledge base not loaded")
        
        # Search for relevant chunks
        hits = self.kb.search(request.question, request.top_k)
        
        # Enrich hits with actual text content
        enriched: List[Dict[str, Any]] = []
        for h in hits:
            text = self.kb.get_chunk_text(h)
            enriched.append({**h, "text": text})
        
        # Build prompt and generate response
        prompt = self.build_prompt(request.question, enriched)
        answer = self.llm.generate_response(prompt, request.temperature)
        
        return AskResponse(
            answer=answer, 
            contexts=enriched, 
            provider="ollama"
        )
