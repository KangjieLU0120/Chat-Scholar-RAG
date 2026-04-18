import re
from typing import Any, List, Dict

TARGET_CHUNK_SIZE = 1000
MAX_CHUNK_SIZE = 1200
MIN_CHUNK_SIZE = 800


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'([a-zA-Z]{2,})-\n\s*([a-z]{2,})', r'\1\2', text)
    text = re.sub(r'(?<=[a-z,])\n(?=[a-z])', ' ', text)
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


HEADING_RE = re.compile(
    r"^([A-Z]|[1-9]\d*(?:\.[1-9]\d*)*)\.?\s+[A-Za-z].{2,60}$"
)
PLAIN_HEADING_RE = re.compile(
    r"^(abstract|introduction|related work|method|methods|experiments|results|conclusion|references)",
    re.I
)


def _is_title(text: str) -> bool:
    text = text.strip()
    if len(text) > 80 or "\n" in text:
        return False
    if text.endswith("."):
        return False
    if HEADING_RE.match(text):
        return True
    if PLAIN_HEADING_RE.match(text):
        return True
    return False


def _heading_level(title: str):
    m = HEADING_RE.match(title)
    if m:
        return len(m.group(1).split("."))
    return 1


def _update_title_path(path: List[str], title: str):
    level = _heading_level(title)
    new_path = path[: level - 1]
    new_path.append(title)
    return new_path


def _is_table(text: str) -> bool:
    lines = text.split("\n")
    if len(lines) < 3:
        return False

    score = 0
    for l in lines:
        digits = sum(c.isdigit() for c in l)
        if len(l) > 0 and digits / len(l) > 0.2:
            score += 1
    return score > len(lines) * 0.4


def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def _semantic_chunk(text: str) -> List[str]:
    sentences = _split_sentences(text)

    chunks = []
    current = []
    length = 0

    for sent in sentences:
        sent_len = len(sent)

        if sent_len > MAX_CHUNK_SIZE:
            if current:
                chunks.append(" ".join(current))
                current = []
                length = 0

            chunks.append(sent)
            continue

        if length + sent_len > TARGET_CHUNK_SIZE:
            if length >= MIN_CHUNK_SIZE:
                chunks.append(" ".join(current))
                current = [sent]
                length = sent_len
            else:
                current.append(sent)
                length += sent_len
        else:
            current.append(sent)
            length += sent_len

    if current:
        chunks.append(" ".join(current))

    return chunks

def _build_chunks(pages: List[Dict[str, Any]]):
    chunks = []
    chunk_id = 0

    buffer = ""
    current_title_path = []

    for page in pages:
        page_num = page.get("page", 1)
        blocks = page.get("blocks", [])

        for block in blocks:
            text = _normalize_text(block.get("text", ""))
            if not text:
                continue
            
            block_type = block.get("block_type", "")

            if block_type == "title" or _is_title(text):
                current_title_path = _update_title_path(
                    current_title_path, text
                )
                continue 

            if block_type == "table_like" or _is_table(text):
                if len(text) < TARGET_CHUNK_SIZE:
                    if buffer:
                        buffer += "\n\n" + text
                    else:
                        buffer = text
                        
                    if len(buffer) >= TARGET_CHUNK_SIZE:
                        sub_chunks = _semantic_chunk(buffer)
                        for sc in sub_chunks[:-1]:
                            chunks.append({
                                "chunk_id": chunk_id,
                                "text": sc,
                                "page": page_num,
                                "section": current_title_path[-1] if current_title_path else None,
                                "title_path": current_title_path[:],
                            })
                            chunk_id += 1
                        buffer = sub_chunks[-1]
                    continue
                else:
                    if buffer:
                        for c in _semantic_chunk(buffer):
                            chunks.append({
                                "chunk_id": chunk_id,
                                "text": c,
                                "page": page_num,
                                "section": current_title_path[-1] if current_title_path else None,
                                "title_path": current_title_path[:],
                            })
                            chunk_id += 1
                        buffer = ""

                    for tc in _semantic_chunk(text):
                        chunks.append({
                            "chunk_id": chunk_id,
                            "text": tc,
                            "page": page_num,
                            "section": current_title_path[-1] if current_title_path else None,
                            "title_path": current_title_path[:],
                            "chunk_type": "table_like"
                        })
                        chunk_id += 1
                    continue

            if buffer:
                buffer += " " + text
            else:
                buffer = text

            if len(buffer) >= TARGET_CHUNK_SIZE:
                sub_chunks = _semantic_chunk(buffer)

                for sc in sub_chunks[:-1]:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": sc,
                        "page": page_num,
                        "section": current_title_path[-1] if current_title_path else None,
                        "title_path": current_title_path[:],
                    })
                    chunk_id += 1

                buffer = sub_chunks[-1]

    if buffer:
        for sc in _semantic_chunk(buffer):
            chunks.append({
                "chunk_id": chunk_id,
                "text": sc,
                "page": page_num,
                "section": current_title_path[-1] if current_title_path else None,
                "title_path": current_title_path[:],
            })
            chunk_id += 1

    return chunks


def split_text_into_chunks(text, chunk_size=500, overlap=50, return_metadata=False):
    if return_metadata:
        if isinstance(text, list):
            if len(text) > 0 and isinstance(text[0], dict):
                return _build_chunks(text)
            else:
                text = "\n\n".join([str(t) for t in text if t])
        pages = [{
            "page": 1,
            "blocks": [{"text": text}]
        }]
        return _build_chunks(pages)

    if isinstance(text, list):
        if len(text) > 0 and isinstance(text[0], dict):
            text = "\n\n".join(p.get("text", "") for p in text if p.get("text"))
        else:
            text = "\n\n".join([str(t) for t in text if t])

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks