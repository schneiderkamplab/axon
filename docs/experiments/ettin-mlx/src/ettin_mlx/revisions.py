"""Pinned Hugging Face revisions: the single place that fixes which weights are used.

These are the ``main`` commits of the official repos as of 2026-10-28. Tests, the
benchmark and ``load("ettin-...")`` all resolve short names through this table.
"""

HF_ORG = "jhu-clsp"

PINNED_REVISIONS: dict[str, str] = {
    "ettin-encoder-17m": "987607455c61e7a5bbc85f7758e0512ea6d0ae4c",
    "ettin-encoder-32m": "1b8ba06455dd44f80fc9c1ca9e22806157a57379",
    "ettin-encoder-68m": "ac19ae4bc51093b31c475665ac872a936d056cc2",
    "ettin-encoder-150m": "45d08642849e5c5701b162671ac811b7654bfd9f",
    "ettin-encoder-400m": "7662476d60abb071a5bd319c9f3074f3072c062d",
    "ettin-encoder-1b": "befd76be43d08b89ff9957012f3ff29d0842780b",
    "ettin-decoder-17m": "728fb5b6b3dc5916aa20829c027143ae8eca4eeb",
    "ettin-decoder-32m": "a6ba5b89e2d8a5b71c819ac6c47ba81423019d3e",
    "ettin-decoder-68m": "5d704138dd3e12a32ca7a357d25b09c618a9fb02",
    "ettin-decoder-150m": "b099375cba8b9045a87a9e6ede0b7ab32fd2f571",
    "ettin-decoder-400m": "ac53c21d2e9284ec150975cfc68f4e09a9c6afa6",
    "ettin-decoder-1b": "7a839081bb4c6f7e10cfb6334f6d9e1e1a76a980",
}


def resolve(name: str, revision: str | None = None) -> tuple[str, str | None]:
    """Map a short name (``ettin-encoder-17m``) or repo id to ``(repo_id, revision)``."""
    short = name.split("/", 1)[1] if name.startswith(HF_ORG + "/") else name
    if short in PINNED_REVISIONS:
        return f"{HF_ORG}/{short}", revision or PINNED_REVISIONS[short]
    return name, revision
