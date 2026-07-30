"""Chunking strategies.

`pack_sentences` packs by character count. `pack_units` packs by token count and can stop at
10-K Item boundaries, which is what the indexer uses — character budgets do not line up with
what the encoder actually consumes, and all-MiniLM-L6-v2 truncates at 256 tokens, so a chunk
built to a larger budget has its tail silently excluded from the dense vector.
"""

from dataclasses import dataclass, field


@dataclass
class SentenceUnit:
    """One source sentence plus the metadata needed to assemble and cite a chunk."""

    sentence: str
    n_tokens: int
    section: str = "unknown"
    sentence_id: str | None = None
    meta: dict = field(default_factory=dict)


def pack_units(
    units: list[SentenceUnit],
    max_tokens: int = 256,
    overlap_tokens: int = 32,
    respect_sections: bool = True,
) -> list[list[SentenceUnit]]:
    """Greedily group sentences into chunks under a token budget.

    Overlap is carried as whole sentences rather than a raw token window: the source data is
    sentence-segmented, and cutting mid-sentence hands the encoder a fragment whose meaning
    depends on text that is no longer there.

    `max_tokens=0` emits one chunk per sentence. `respect_sections` starts a new chunk whenever
    the Item changes, so a chunk never straddles two Items and its `section` stays unambiguous.
    """
    chunks: list[list[SentenceUnit]] = []
    buf: list[SentenceUnit] = []
    buf_tokens = 0
    current_section: str | None = None

    for unit in units:
        if respect_sections and current_section is not None and unit.section != current_section:
            if buf:
                chunks.append(buf)
            buf, buf_tokens = [], 0
        current_section = unit.section

        if buf and buf_tokens + unit.n_tokens > max_tokens:
            chunks.append(buf)
            tail: list[SentenceUnit] = []
            tail_tokens = 0
            for prev in reversed(buf):
                if tail_tokens + prev.n_tokens > overlap_tokens:
                    break
                tail.insert(0, prev)
                tail_tokens += prev.n_tokens
            buf, buf_tokens = list(tail), tail_tokens

        buf.append(unit)
        buf_tokens += unit.n_tokens

    if buf:
        chunks.append(buf)
    return chunks


def pack_sentences(sentences: list[str], max_chars: int = 1800) -> list[str]:
    chunks: list[str] = []
    buf: list[str] = []
    size = 0

    for s in sentences:
        s = (s or "").strip()
        if not s:
            continue

        if size + len(s) + 1 > max_chars and buf:
            chunks.append(" ".join(buf))
            buf = [s]
            size = len(s)
        else:
            buf.append(s)
            size += len(s) + 1

    if buf:
        chunks.append(" ".join(buf))

    return chunks
