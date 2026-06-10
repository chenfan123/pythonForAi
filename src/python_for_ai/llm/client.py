from typing import Optional

from openai import OpenAI

from python_for_ai.config import get_api_key, get_base_url

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=get_api_key(), base_url=get_base_url())
    return _client


def reset_client() -> None:
    """Reset cached client (used in tests)."""
    global _client
    _client = None
