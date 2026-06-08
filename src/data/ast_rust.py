"""Optional tree-sitter Rust helpers for AST-aware FIM + snippet extraction.

Decision: AST is a *refinement, not critical path*. We use tree-sitter-rust (Python
bindings) — NOT `syn` (Rust-only, needs a helper binary). Everything here degrades
gracefully: if tree-sitter isn't installed, callers fall back to char/line FIM and
regex snippet extraction, so the pipeline still runs end-to-end on the MVP.

Uses:
  * span_for_fim()      — pick a complete top-level item (fn/impl/struct/...) to mask,
                          so FIM trains on whole semantic units, not random char spans.
  * extract_functions() — function-level snippets to seed OSS-Instruct SFT data.
"""
from __future__ import annotations

_ITEM_KINDS = {
    "function_item", "impl_item", "struct_item", "enum_item", "trait_item", "mod_item",
}


def available() -> bool:
    try:
        import tree_sitter_rust  # noqa: F401
        import tree_sitter  # noqa: F401
        return True
    except ImportError:
        return False


def _parse(code: str):
    import tree_sitter_rust
    from tree_sitter import Language, Parser

    parser = Parser(Language(tree_sitter_rust.language()))
    return parser.parse(bytes(code, "utf-8"))


def top_level_items(code: str) -> list[tuple[int, int, str]]:
    """Return [(start_byte, end_byte, kind)] for top-level items. Empty if unavailable."""
    if not available():
        return []
    tree = _parse(code)
    out: list[tuple[int, int, str]] = []
    for child in tree.root_node.children:
        if child.type in _ITEM_KINDS:
            out.append((child.start_byte, child.end_byte, child.type))
    return out


def extract_functions(code: str, min_len: int = 40) -> list[str]:
    """Function/impl source snippets for SFT instruction seeding (fallback: regex)."""
    items = top_level_items(code)
    if items:
        raw = code.encode("utf-8")
        return [raw[s:e].decode("utf-8", "ignore") for s, e, k in items
                if k in ("function_item", "impl_item") and (e - s) >= min_len]
    # fallback: crude `fn ... { ... }` slices
    import re
    return [m.group(0) for m in re.finditer(r"(?s)fn\s+\w+.*?\n}\n", code)
            if len(m.group(0)) >= min_len]


def span_for_fim(code: str) -> tuple[int, int] | None:
    """Pick a complete item span to use as the FIM 'middle'. None => caller uses char FIM."""
    items = top_level_items(code)
    if not items:
        return None
    # choose the largest item (most informative middle); deterministic, no RNG
    s, e, _ = max(items, key=lambda it: it[1] - it[0])
    return (s, e)
