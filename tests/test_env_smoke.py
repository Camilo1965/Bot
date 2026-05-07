from __future__ import annotations

from pathlib import Path


def test_env_example_exists() -> None:
    env_example = Path(__file__).resolve().parents[1] / ".env.example"
    assert env_example.exists(), ".env.example must exist for setup"
