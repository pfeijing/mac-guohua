from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def load_config(path: str | Path) -> dict[str, Any]:
    load_dotenv()

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    generation = cfg.setdefault("generation", {})

    env_mapping = {
        "MAC_GUOHUA_BASE_MODEL": "base_model",
        "MAC_GUOHUA_CONTROLNET": "controlnet_model",
        "MAC_GUOHUA_VAE": "vae_model",
        "MAC_GUOHUA_LORA": "lora_path",
    }

    for env_name, config_name in env_mapping.items():
        value = os.getenv(env_name)
        if value:
            generation[config_name] = value

    return cfg