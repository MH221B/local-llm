"""Checks 1-3 for the Kitty port.

1. Cache-level golden trace: Kitty's reference KittyKVCache and our
   QuantizingDynamicLayer are driven through the same scripted prefill + decode on
   synthetic K/V, and must return identical tensors at every step.
2. Structural: the layer map and head dimensions.
3. Quantization actually exercised: the cache must have quantized something, and a
   quantized run must differ from an fp16 run.

Runs offline on CPU. No model weights are downloaded.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.cache import QuantizingDynamicLayer, full_attention_layers  # noqa: E402

MODEL_ID = "RBergBauer/Qwen3.5-4B-MTP-Heretic"

# Kitty's defaults, which our port must reproduce.
CONFIG = dict(
    sink_length=32,
    buffer_length=128,
    group_size=128,
    kbits=2,
    vbits=2,
    promote_ratio=0.1,
    promote_bit=4,
    channel_selection=1,
)


def load_kitty_reference(reference_dir: Path):
    """Load Kitty's modules without executing kitty_sim/__init__.py.

    The package __init__ eagerly imports a module that needs CacheConfig, which
    upstream removed after v4.55. Loading by file path under a synthetic package
    avoids the __init__ while still satisfying kitty_simulate's relative import.
    """
    from transformers import cache_utils

    if not hasattr(cache_utils, "CacheConfig"):
        class CacheConfig:  # minimal stand-in: Kitty only uses it as a base class
            def __init__(self, cache_implementation: str = ""):
                self.cache_implementation = cache_implementation

        cache_utils.CacheConfig = CacheConfig

    package = types.ModuleType("kitty_sim")
    package.__path__ = [str(reference_dir)]
    sys.modules["kitty_sim"] = package

    def load(name: str):
        path = reference_dir / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"kitty_sim.{name}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"kitty_sim.{name}"] = module
        spec.loader.exec_module(module)
        return module

    return load("utils_quant"), load("kitty_simulate")


def check_structural() -> None:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(MODEL_ID)
    layers = full_attention_layers(cfg)
    assert layers == [3, 7, 11, 15, 19, 23, 27, 31], layers
    assert len(layers) == 8 and cfg.text_config.num_hidden_layers == 32
    assert cfg.text_config.head_dim == 256, cfg.text_config.head_dim
    assert cfg.text_config.num_key_value_heads == 4, cfg.text_config.num_key_value_heads
    print("check 2 (structural): ok  layers =", layers)


def check_golden_trace(reference_dir: Path, prefill: int, decode: int) -> None:
    _, kitty_sim = load_kitty_reference(reference_dir)

    reference = kitty_sim.KittyKVCache(kitty_sim.KittyKVCacheConfig(**CONFIG))
    # transformers 5.17's DynamicCache no longer carries the legacy key_cache /
    # value_cache lists Kitty's update() mutates (they moved to cache.layers[i].
    # keys/.values). Kitty touches only those two attributes inside update(), so the
    # reference instance gets plain lists here. No third-party line is edited; the
    # shim is recorded in bench/port-results.md.
    reference.key_cache = []
    reference.value_cache = []
    ours = QuantizingDynamicLayer(**CONFIG)

    torch.manual_seed(0)
    shape = (1, 4, prefill, 256)
    k, v = torch.randn(*shape, dtype=torch.float16), torch.randn(*shape, dtype=torch.float16)

    def compare(step: int) -> None:
        if reference.PostQuant:
            ref_k, ref_v = reference.key_cache[0], reference.value_cache[0]
        else:
            raise AssertionError("reference must run PostQuant")
        assert torch.equal(ref_k, ours.keys), f"step {step}: stored K diverged"
        assert torch.equal(ref_v, ours.values), f"step {step}: stored V diverged"

    rk, rv = reference.update(k, v, layer_idx=0)
    ok, ov = ours.update(k, v)
    assert torch.equal(rk, ok), "prefill returned K diverged"
    assert torch.equal(rv, ov), "prefill returned V diverged"
    compare(0)

    for step in range(1, decode + 1):
        k1 = torch.randn(1, 4, 1, 256, dtype=torch.float16)
        v1 = torch.randn(1, 4, 1, 256, dtype=torch.float16)
        rk, rv = reference.update(k1, v1, layer_idx=0)
        ok, ov = ours.update(k1, v1)
        assert torch.equal(rk, ok), f"decode step {step}: returned K diverged"
        assert torch.equal(rv, ov), f"decode step {step}: returned V diverged"
        compare(step)

    assert ours.quantized_pages > 0, "trace must have quantized keys"
    assert ours.quantized_value_blocks > 0, "trace must have quantized values"
    print(f"check 1 (golden trace): ok  prefill={prefill} decode={decode} "
          f"pages={ours.quantized_pages} value_blocks={ours.quantized_value_blocks}")


def check_quantization_exercised() -> None:
    torch.manual_seed(0)
    prefill = 200  # > sink(32) + buffer(128), the 160-token dead zone
    k = torch.randn(1, 4, prefill, 256, dtype=torch.float16)
    v = torch.randn(1, 4, prefill, 256, dtype=torch.float16)

    quantizing = QuantizingDynamicLayer(**CONFIG)
    quantizing.update(k, v)
    assert quantizing.quantized_pages > 0, "nothing was quantized"

    fp16_cfg = {**CONFIG, "kbits": 16, "vbits": 16, "promote_ratio": 0.0}
    fp16 = QuantizingDynamicLayer(**fp16_cfg)
    fp16.update(k, v)

    quantized_region = slice(CONFIG["sink_length"], prefill)
    assert not torch.equal(
        quantizing.keys[:, :, quantized_region, :], fp16.keys[:, :, quantized_region, :]
    ), "quantized cache is identical to fp16 - quantization is a no-op"

    # Below the dead zone nothing may be quantized.
    tiny = QuantizingDynamicLayer(**CONFIG)
    tiny.update(torch.randn(1, 4, 160, 256, dtype=torch.float16),
                torch.randn(1, 4, 160, 256, dtype=torch.float16))
    assert tiny.quantized_pages == 0, "160 tokens is the dead zone boundary, nothing to quantize"
    print("check 3 (quantization exercised): ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True,
                        help="directory holding Kitty's utils_quant.py and kitty_simulate.py")
    parser.add_argument("--prefill", type=int, default=200)
    parser.add_argument("--decode", type=int, default=300)
    args = parser.parse_args()

    check_structural()
    check_golden_trace(args.reference, args.prefill, args.decode)
    check_quantization_exercised()
    print("all checks passed")


if __name__ == "__main__":
    main()
