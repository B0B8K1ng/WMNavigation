"""
Dataclass-based configuration for VLM backends with YAML loading support.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from api import BaseVLM


@dataclass
class GeminiConfig:
    model: str = "gemini-2.0-flash"
    system_instruction: Optional[str] = None
    api_key_env: str = "GEMINI_API_KEY"
    base_url_env: str = "GEMINI_BASE_URL"


@dataclass
class QwenConfig:
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    system_instruction: Optional[str] = None
    api_key_env: str = "QWEN_API_KEY"
    base_url_env: str = "QWEN_BASE_URL"


def _load_yaml(path: str) -> dict:
    try:
        import yaml  # optional dependency
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        raise ImportError("PyYAML is required for YAML config loading: pip install pyyaml")


def gemini_config_from_yaml(path: str) -> GeminiConfig:
    data = _load_yaml(path).get("gemini", {})
    return GeminiConfig(**{k: v for k, v in data.items() if k in GeminiConfig.__dataclass_fields__})


def qwen_config_from_yaml(path: str) -> QwenConfig:
    data = _load_yaml(path).get("qwen", {})
    return QwenConfig(**{k: v for k, v in data.items() if k in QwenConfig.__dataclass_fields__})


def build_gemini_vlm(config: Optional[GeminiConfig] = None) -> "BaseVLM":
    from api import GeminiVLM
    cfg = config or GeminiConfig()
    return GeminiVLM(model=cfg.model, system_instruction=cfg.system_instruction)


def build_qwen_vlm(config: Optional[QwenConfig] = None) -> "BaseVLM":
    from api import QwenVLM
    cfg = config or QwenConfig()
    return QwenVLM(model=cfg.model, system_instruction=cfg.system_instruction)


def build_vlm_from_yaml(path: str) -> "BaseVLM":
    """
    Load VLM config from a YAML file and return the appropriate VLM instance.

    Expected YAML structure::

        backend: gemini   # or "qwen"
        gemini:
          model: gemini-2.0-flash
        qwen:
          model: Qwen/Qwen2.5-VL-7B-Instruct
    """
    data = _load_yaml(path)
    backend = data.get("backend", "gemini").lower()
    if backend == "qwen":
        return build_qwen_vlm(qwen_config_from_yaml(path))
    return build_gemini_vlm(gemini_config_from_yaml(path))
