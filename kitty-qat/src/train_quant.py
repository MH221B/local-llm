"""Straight-through-estimator wrapper around the vendored Kitty fake-quant.

Forward IS the vendored function (bit-equality by construction); backward
returns identity across the round() boundary. Scales and promote masks are
non-differentiable by design and are computed by the caller (detached).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a plain script (`python src/train_quant.py`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.quant import fake_quant_groupwise_lastdim


class SteFakeQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, group_size, bit, promote_mask, promote_bit):
        with torch.no_grad():
            return fake_quant_groupwise_lastdim(
                data=data,
                group_size=group_size,
                bit=bit,
                promote_mask=promote_mask,
                promote_bit=promote_bit,
            )

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


def ste_fake_quant(data, group_size, bit, promote_mask=None, promote_bit=4):
    """Same kwarg shape as the vendored call, so call sites stay identical."""
    return SteFakeQuant.apply(data, group_size, bit, promote_mask, promote_bit)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1, 4, 8, 256, dtype=torch.float16)

    from src.quant import build_promote_mask

    mask = build_promote_mask(key_states=x, promote_ratio=0.5, channel_selection=1)

    ref = fake_quant_groupwise_lastdim(data=x, group_size=128, bit=2,
                                       promote_mask=mask, promote_bit=4)
    ste = ste_fake_quant(x, 128, 2, promote_mask=mask, promote_bit=4)
    assert torch.equal(ref, ste), "STE forward must be bit-equal to the vendored quantizer"

    assert torch.equal(ste_fake_quant(x, 128, 16), x), "bit>=16 must short-circuit through STE"

    xt = torch.randn(1, 4, 8, 256, dtype=torch.float32, requires_grad=True)
    ste_fake_quant(xt, 128, 2).sum().backward()
    assert xt.grad is not None and torch.equal(xt.grad, torch.ones_like(xt)), \
        "backward must return identity"

    print("train_quant.py ok")
