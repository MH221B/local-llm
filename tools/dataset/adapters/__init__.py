"""Adapter registry. Keys match the adapter column in sources.py."""
from __future__ import annotations

from .base import Adapter
from . import cauldron, codefeedback, plainmath, sharegpt, smoltalk, toolcalls

ADAPTERS: dict[str, Adapter] = {
    "smoltalk": Adapter("smoltalk", smoltalk.build),
    "sharegpt_openhermes": Adapter("sharegpt_openhermes", sharegpt.build_openhermes),
    "sharegpt_toolace": Adapter("sharegpt_toolace", sharegpt.build_toolace),
    "numina": Adapter("numina", plainmath.build_numina),
    "gsm8k": Adapter("gsm8k", plainmath.build_gsm8k),
    "codefeedback": Adapter("codefeedback", codefeedback.build),
    "toolcalls": Adapter("toolcalls", toolcalls.build),
    "cauldron": Adapter("cauldron", cauldron.build),
}


def get(key: str) -> Adapter:
    if key not in ADAPTERS:
        raise KeyError(f"unknown adapter {key!r}; known: {sorted(ADAPTERS)}")
    return ADAPTERS[key]
