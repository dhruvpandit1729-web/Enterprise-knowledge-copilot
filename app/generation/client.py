from __future__ import annotations
import httpx
from app.core.config import settings

class OllamaClient:
    def __init__(self, host: str = settings.ollama_host, model: str = settings.llm_model):
        self.host = host
        self.model = model

    async def generate(self, prompt: str) -> str:
        async with httpx.AsyncClient(base_url=self.host, timeout=60.0) as client:
            response = await client.post(
                "/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
            )
            response.raise_for_status()
            return response.json().get("response", "")
