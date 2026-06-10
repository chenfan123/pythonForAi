from unittest.mock import MagicMock, patch

import pytest

from python_for_ai.llm.service import get_llm_response


def test_get_llm_response_returns_content() -> None:
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = "Hello"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion

    with patch("python_for_ai.llm.service.get_client", return_value=mock_client):
        result = get_llm_response("Hi")

    assert result == "Hello"
    mock_client.chat.completions.create.assert_called_once()


def test_get_llm_response_rejects_non_string() -> None:
    with pytest.raises(ValueError, match="Input must be a string"):
        get_llm_response(123)  # type: ignore[arg-type]
