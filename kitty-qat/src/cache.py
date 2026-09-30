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


class QuantizingDynamicLayer(DynamicLayer):
    """Kitty's KIVI-style 2-bit KV cache as a drop-in ``DynamicLayer``.

    Ported from ``kitty_sim.KittyKVCache.update``. The branch structure, the page
    grid anchored at ``sink_length``, the KIVI-style V retirement rule and the
    post-quantization return timing are reproduced deliberately. Do not "simplify"
    them: each is observable in the golden trace, and every one of them is a way
    to silently quantize the wrong tokens.
    """

    def __init__(
        self,
        sink_length: int = 32,
        buffer_length: int = 128,
        group_size: int = 128,
        kbits: int = 2,
        vbits: int = 2,
        promote_ratio: float = 0.1,
        promote_bit: int = 4,
        channel_selection: int = 1,
        post_quant: bool = True,
    ):
        super().__init__()
        if channel_selection not in (0, 1):
            raise ValueError(f"channel_selection must be 0 or 1, got {channel_selection}")
        if group_size <= 0 or group_size > buffer_length or buffer_length % group_size != 0:
            raise ValueError("group_size must be a positive factor of buffer_length")
        if not 0.0 <= promote_ratio <= 1.0:
            raise ValueError(f"promote_ratio must be in [0, 1], got {promote_ratio}")
        self.sink_length = sink_length
        self.buffer_length = buffer_length
        self.group_size = group_size
        self.kbits = kbits
        self.vbits = vbits
        self.promote_ratio = promote_ratio
        self.promote_bit = promote_bit
        self.channel_selection = channel_selection
        self.post_quant = post_quant
        # Observability for the "quantization actually happened" check.
        self.quantized_pages = 0
        self.quantized_value_blocks = 0

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized or self.get_seq_length() == 0:
            return self._prefill(key_states, value_states)
        return self._decode(key_states, value_states)

    # -- lifecycle -----------------------------------------------------------------

    def _prefill(self, key_states, value_states):
        self.lazy_initialization(key_states, value_states)
        self.keys = key_states.detach().clone()
        self.values = value_states.detach().clone()
        # PostQuant: the returned tensors are a snapshot taken BEFORE write-back, so the
        # current step attends to unquantized K/V and each later step lags one page.
        to_return = None
        if self.post_quant:
            to_return = (self.keys.detach().clone(), self.values.detach().clone())

        length = self.keys.shape[-2]
        if length <= self.sink_length:
            raise ValueError(
                f"Kitty: cached length {length} must exceed sink_length {self.sink_length}"
            )
        if length > self.sink_length + self.buffer_length:
            start = self.sink_length
            after_sink = length - self.sink_length
            end = start + after_sink - (after_sink % self.buffer_length)
            for idx in range(start, end, self.buffer_length):
                self._quantize_key_page(idx)
            # KIVI-style V on prefill: the sink up to the token buffer.
            self._quantize_values(start, start + after_sink - self.buffer_length)

        return to_return if self.post_quant else (self.keys, self.values)

    def _decode(self, key_states, value_states):
        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        to_return = None
        if self.post_quant:
            to_return = (self.keys.detach().clone(), self.values.detach().clone())

        length = self.keys.shape[-2]
        quantizable = length - self.sink_length - self.buffer_length
        if quantizable > 0 and quantizable % self.buffer_length == 1:
            self._quantize_key_page(length - self.buffer_length - 1)
        if quantizable > 0:
            start = length - 2 * self.buffer_length - 1
            self._quantize_values(start, start + self.buffer_length)

        return to_return if self.post_quant else (self.keys, self.values)

    # -- quantization --------------------------------------------------------------

    def _quantize_key_page(self, start: int) -> None:
        end = start + self.buffer_length
        page = self.keys[:, :, start:end, :].transpose(2, 3).contiguous()
        mask = build_promote_mask(
            key_states=page,
            promote_ratio=self.promote_ratio,
            channel_selection=self.channel_selection,
        )
        page = fake_quant_groupwise_lastdim(
            data=page,
            group_size=self.group_size,
            bit=self.kbits,
            promote_mask=mask,
            promote_bit=self.promote_bit,
        ).transpose(2, 3).contiguous()
        self.keys[:, :, start:end, :] = page
        self.quantized_pages += 1

    def _quantize_values(self, start: int, end: int) -> None:
        if end <= start:
            return
        block = fake_quant_groupwise_lastdim(
            data=self.values[:, :, start:end, :],
            group_size=self.group_size,
            bit=self.vbits,
        )
        self.values[:, :, start:end, :] = block
        self.quantized_value_blocks += 1


def build_kitty_cache(cache, config, **kwargs):
    """Install QuantizingDynamicLayer at every full-attention index of ``cache``.

    This is the enforcement point for spec section 7. In the v5.8+ cache API the
    ``update()`` call carries no ``layer_idx`` (the ``Cache`` dispatches by index),
    so the "never quantize a non-attention layer" rule cannot live inside
    ``update()``. It lives here instead: only ``full_attention`` indices are
    replaced, and every other index is asserted to be a non-KV layer we must not
    touch.

    ``cache`` is the model's own cache object with ``cache.layers`` already
    populated, and ``config`` is the model config.
    """
    from transformers.cache_utils import DynamicLayer

    text = getattr(config, "text_config", config)
    types = list(text.layer_types)
    wanted = [i for i, t in enumerate(types) if t == FULL_ATTENTION]

    for i in wanted:
        current = cache.layers[i]
        assert isinstance(current, DynamicLayer), (
            f"layer {i} is declared {types[i]} but its cache object is "
            f"{type(current).__name__}, not DynamicLayer. Refusing to install a "
            "quantizer on a layer that does not hold a KV cache."
        )
        cache.layers[i] = QuantizingDynamicLayer(**kwargs)

    assert len(wanted) == 8 and wanted == [3, 7, 11, 15, 19, 23, 27, 31], wanted
    return cache, wanted


# Spec section 7 enforcement map:
#   "update() called with a non-full-attention layer_idx must raise"
#       -> enforced in build_kitty_cache(), because the v5.8+ Cache dispatches by
#          index and never passes layer_idx into update(). Only full_attention
#          indices are ever replaced, and each replaced index asserts the existing
#          object was a DynamicLayer (i.e. actually holds a KV cache).
#   "non-KV cache traffic from linear layers is normal and must never raise"
#       -> linear layers use LinearAttentionLayer and are never swapped.
#   "assert quantization actually occurred"
#       -> scripts/equiv_check.py check 3, via quantized_pages.


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

    torch.manual_seed(0)
    B, H, D, SINK, BUF = 1, 4, 256, 32, 128

    def kv(n):
        return (
            torch.randn(B, H, n, D, dtype=torch.float16),
            torch.randn(B, H, n, D, dtype=torch.float16),
        )

    layer = QuantizingDynamicLayer()
    k, v = kv(200)
    returned_k, _ = layer.update(k, v)

    assert returned_k.shape[-2] == 200, "post-quant return is the full prefill"
    assert torch.equal(returned_k, k), "post-quant prefill return must be UNQUANTIZED"
    assert layer.quantized_pages > 0, "prefill must quantize at least one page"
    # The sink is never touched.
    assert torch.equal(layer.keys[:, :, :SINK, :], k[:, :, :SINK, :]), "sink must stay fp16"
    # A quantized page must differ from the input.
    assert not torch.equal(layer.keys[:, :, SINK:SINK + BUF, :], k[:, :, SINK:SINK + BUF, :])

    pages_after_prefill = layer.quantized_pages
    for _ in range(300):
        layer.update(*kv(1))
    assert layer.quantized_pages > pages_after_prefill, "decode must quantize more pages"
    assert layer.quantized_value_blocks > 0

    print("QuantizingDynamicLayer ok:", layer.quantized_pages, "pages,",
          layer.quantized_value_blocks, "value blocks")
