import logging

import requests

logger = logging.getLogger(__name__)


class OllamaLLM:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        try:
            r = requests.post(url, json=payload, timeout=120)
            r.raise_for_status()
        except requests.ConnectionError:
            logger.error("Cannot connect to Ollama at %s", self.base_url)
            raise RuntimeError(f"Cannot connect to Ollama at {self.base_url}")
        except requests.Timeout:
            logger.error("Ollama request timed out after 120s")
            raise RuntimeError("Ollama request timed out after 120s")
        except requests.HTTPError as e:
            logger.error("Ollama returned HTTP %s: %s", e.response.status_code, e.response.text[:200])
            raise
        return r.json().get("response", "").strip()
