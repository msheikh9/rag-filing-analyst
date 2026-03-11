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