from __future__ import annotations

import os
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_model_path() -> Path:
    # GitHub-friendly default path inside repository.
    return project_root() / "weights" / "hf" / "Qwen3.5-0.8B"


def get_model_path() -> str:
    return os.environ.get("QWEN_MODEL_PATH", str(default_model_path()))


def add_model_path_to_sys() -> str:
    model_path = get_model_path()
    if model_path not in sys.path:
        sys.path.insert(0, model_path)
    return model_path
