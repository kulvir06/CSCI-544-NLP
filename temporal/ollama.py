"""Small standard-library client for a locally running Ollama instance."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from api import config
from temporal.schema import ClaimExtraction


class OllamaUnavailableError(RuntimeError):
    pass


class OllamaClient:
    def __init__(
        self,
        base_url: str = config.OLLAMA_BASE_URL,
        temporal_model: str = config.OLLAMA_TEMPORAL_MODEL,
        embedding_model: str = config.OLLAMA_EMBEDDING_MODEL,
        timeout: float = config.OLLAMA_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.temporal_model = temporal_model
        self.embedding_model = embedding_model
        self.timeout = timeout

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise OllamaUnavailableError(
                f"Ollama returned HTTP {error.code}: {detail}"
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise OllamaUnavailableError(
                f"Cannot reach Ollama at {self.base_url}: {error}"
            ) from error

    def extract_claims(self, prompt: str) -> ClaimExtraction:
        response = self._post(
            "/api/chat",
            {
                "model": self.temporal_model,
                "stream": False,
                "format": ClaimExtraction.model_json_schema(),
                "options": {"temperature": 0, "num_predict": 1200},
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You extract atomic regulatory claims and their temporal scope. "
                            "Return only schema-valid JSON. Never invent a date. Use null "
                            "when the supplied text does not support a boundary."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            },
        )
        try:
            content = response["message"]["content"]
            return ClaimExtraction.model_validate_json(content)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid temporal extraction response: {response}") from error

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._post(
            "/api/embed",
            {"model": self.embedding_model, "input": texts},
        )
        embeddings = response.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise ValueError("Ollama returned an invalid embedding response")
        return embeddings

    def available_models(self) -> list[str]:
        request = Request(f"{self.base_url}/api/tags", method="GET")
        try:
            with urlopen(request, timeout=min(self.timeout, 5.0)) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            raise OllamaUnavailableError(
                f"Cannot reach Ollama at {self.base_url}: {error}"
            ) from error
        return [item.get("name", "") for item in payload.get("models", [])]
