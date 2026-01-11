"""Ollama API client for making LLM requests."""

import httpx
from typing import List, Dict, Any, Optional

async def ollama_chat(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 420.0,
) -> Optional[str]:
    """
    Appelle l'API Ollama /api/chat et retourne le texte assistant.
    Doc officielle: POST /api/chat. (cf docs Ollama)
    """
    url = f"{base_url.rstrip('/')}/api/chat"
    # Payload conforme à l'API Ollama
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()

    # Réponse Ollama = {"message": {"role":"assistant","content":"..."} , ...}
    return data.get("message", {}).get("content")
