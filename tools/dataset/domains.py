"""Domain assignment. Sources are single-domain; the table classifies mixed sources."""
from __future__ import annotations

from .canonical import DOMAINS

KEYWORDS = {
    "reasoning": ("solve", "prove", "theorem", "derive", "calculate", "equation", "integral"),
    "coding": ("function", "def ", "class ", "compile", "refactor", "traceback", "import "),
    "roleplay": ("pretend", "roleplay", "character", "story", "persona", "*smiles*"),
    "uncensored": ("jailbreak", "bypass", "unrestricted", "no restrictions", "as an ai without"),
}


def assign_domain(text: str, declared: str) -> str:
    """Trust a source's declared domain; keyword-classify only mixed/unmapped sources.

    Spec section 9.5: rule-based per source. Substring keyword hits must never silently
    move a single-domain source (e.g. a CodeFeedback prompt mentioning "solve"), so the
    fallback applies only when the source declares something outside DOMAINS.
    """
    if declared in DOMAINS:
        return declared
    low = text.lower()
    for domain, words in KEYWORDS.items():
        if any(w in low for w in words):
            return domain
    return declared


def prompt_text(messages: list[dict]) -> str:
    for m in messages:
        if m["role"] == "user":
            content = m["content"]
            if isinstance(content, str):
                return content
            return " ".join(p.get("text", "") for p in content if p.get("type") == "text")
    return ""


if __name__ == "__main__":
    msgs = [{"role": "user", "content": [{"type": "text", "text": "Solve x^2 = 4"}]}]
    print("declared kept:", assign_domain(prompt_text(msgs), "reasoning"))
    print("mixed classified:", assign_domain("write a def and refactor", "mixed"))
    print("mixed fallback:", assign_domain("pretend you are a prince", "mixed"))
