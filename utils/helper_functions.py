import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(_ENV_PATH, override=True)

MODEL_NAME = "qwen3.7-max"
DEFAULT_BASE_URL = "https://api.302ai.cn/v1"

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "未找到 DASHSCOPE_API_KEY。请在 AIPython/.env 中配置 API Key。"
            )
        base_url = os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL)
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


def get_llm_response(prompt):
    """Call Qwen and return the model response as a string."""
    if not isinstance(prompt, str):
        raise ValueError("Input must be a string enclosed in quotes.")

    completion = _get_client().chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful but terse AI assistant who gets straight to the point.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return completion.choices[0].message.content


def print_llm_response(prompt):
    """Call Qwen and print the model response."""
    try:
        response = get_llm_response(prompt)
        print("*" * 100)
        print(response)
        print("*" * 100)
        print("\n")
    except TypeError as error:
        print("Error:", str(error))
