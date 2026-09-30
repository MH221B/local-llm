"""Layer-aware Kitty KV cache for the Qwen3.5 hybrid architecture."""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a plain script (`python src/cache.py`), not only as a module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers.cache_utils import DynamicLayer

from src.quant import build_promote_mask, fake_quant_groupwise_lastdim

FULL_ATTENTION = "full_attention"


def full_attention_layers(config) -> list[int]:
    """Indices of the layers that actually hold a KV cache."""
    text = getattr(config, "text_config", config)
    return [i for i, t in enumerate(text.layer_types) if t == FULL_ATTENTION]


def load_model(model_id: str, dtype=torch.float16):
    """Load the text side of a Qwen3.5 checkpoint. Vision is ignored."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.eval()
    return model, tokenizer


if __name__ == "__main__":
    from transformers import AutoConfig

    MODEL_ID = "RBergBauer/Qwen3.5-4B-MTP-Heretic"
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    layers = full_attention_layers(cfg)
    print("layer_types:", cfg.text_config.layer_types)
    print("full attention:", layers)
    assert layers == [3, 7, 11, 15, 19, 23, 27, 31], layers
    assert len(layers) == 8
    assert cfg.text_config.num_hidden_layers == 32
    print("head_dim:", cfg.text_config.head_dim)
    print("num_key_value_heads:", cfg.text_config.num_key_value_heads)
    print("mtp_num_hidden_layers:", cfg.text_config.mtp_num_hidden_layers)
    print("cache.py structural ok")
