from collections.abc import Iterator

import pytest

from python_for_ai.llm.client import reset_client


@pytest.fixture(autouse=True)
def _reset_llm_client() -> Iterator[None]:
    reset_client()
    yield
    reset_client()
