import os
from pathlib import Path

from dotenv import load_dotenv

MODEL_NAME = "qwen3.7-max"
DEFAULT_BASE_URL = "https://api.302ai.cn/v1"


def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


PROJECT_ROOT = find_project_root()
ENV_PATH = PROJECT_ROOT / ".env"


def load_config() -> None:
    load_dotenv(ENV_PATH, override=True)


def get_api_key() -> str:
    load_config()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError(
            "未找到 DASHSCOPE_API_KEY。请在项目根目录 .env 中配置 API Key。"
        )
    return api_key


def get_base_url() -> str:
    load_config()
    return os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL)
