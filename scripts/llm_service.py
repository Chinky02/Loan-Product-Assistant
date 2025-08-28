import logging
from typing import Dict, List, Any
import ollama

logger = logging.getLogger(__name__)


class LLMService:
    """Handles LLM interactions using Ollama."""
    
    def __init__(self, model: str = "llama3"):
        self.model = model
        self.system_message = (
            "You are a helpful assistant specializing in Bank of Maharashtra loan products. "
            "Answer succinctly using only the provided context. If the answer isn't in the context, say you don't know."
        )

    def generate_response(self, prompt: str, temperature: float = 0.2) -> str:
        """Generate a response using the LLM."""
        try:
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt},
            ]
            
            response = ollama.chat(
                model=self.model, 
                messages=messages, 
                options={"temperature": temperature}
            )
            
            return response.get("message", {}).get("content", "")
            
        except Exception as e:
            error_msg = str(e).lower()
            if "not found" in error_msg or "status code: 404" in error_msg:
                logger.warning(f"Model {self.model} not found, trying fallback")
                return self._try_fallback(prompt, temperature)
            else:
                logger.error(f"LLM error: {e}")
                raise

    def _try_fallback(self, prompt: str, temperature: float) -> str:
        """Try with a fallback model if the primary model fails."""
        fallback_model = "llama3"
        if self.model == fallback_model:
            raise RuntimeError(f"Both primary model {self.model} and fallback {fallback_model} failed")
        
        try:
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt},
            ]
            
            response = ollama.chat(
                model=fallback_model, 
                messages=messages, 
                options={"temperature": temperature}
            )
            
            logger.info(f"Successfully used fallback model: {fallback_model}")
            return response.get("message", {}).get("content", "")
            
        except Exception as e:
            logger.error(f"Fallback model also failed: {e}")
            raise RuntimeError(f"Both primary and fallback models failed: {e}")
