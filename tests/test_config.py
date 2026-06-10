from python_for_ai.config import ENV_PATH, PROJECT_ROOT, find_project_root


def test_project_root_contains_pyproject() -> None:
    root = find_project_root()
    assert (root / "pyproject.toml").is_file()
    assert PROJECT_ROOT == root


def test_env_path_is_in_project_root() -> None:
    assert ENV_PATH == PROJECT_ROOT / ".env"
