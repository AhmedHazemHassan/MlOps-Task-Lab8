"""Configuration for AutoGen LLM access using a local Ollama model.
"""

from __future__ import annotations

import os
from typing import Dict, List


def get_llm_config() -> Dict:
    """Return an AutoGen ``llm_config`` dictionary for Ollama mistral.

    Environment variables
    ---------------------
    OLLAMA_BASE_URL:
        Optional base URL for the Ollama server. Defaults to
        ``"http://localhost:11434/v1"``.

    The configuration uses ``model="mistral:latest"`` and a dummy API key,
    which is sufficient for OpenAI-compatible Ollama endpoints.
    """

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

    config_list: List[Dict[str, str]] = [
        {
            "model": "mistral:latest",
            "api_key": "NA",
            "base_url": base_url,
        }
    ]

    return {"config_list": config_list}
