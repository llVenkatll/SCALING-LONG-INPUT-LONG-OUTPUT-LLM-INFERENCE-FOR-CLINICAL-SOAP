def chunk_text_by_words(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    chunks = []
    start = 0
    step = max(1, chunk_words - overlap_words)
    while start < len(words):
        end = start + chunk_words
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return chunks
